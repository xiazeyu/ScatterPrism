"""Generative models for ScatterPrism using PyTorch Lightning.

Includes:
    * :class:`CFM`  — (Conditional) Flow Matching with torchdyn ODE solvers.
      Used for both unconditional generation and detector unfolding.
    * :class:`DDPM` — Denoising Diffusion Probabilistic Model (experimental
      baseline; retained for comparison and future work).
"""

import logging

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.stats import ks_2samp as scipy_ks_2samp
from scipy.stats import wasserstein_distance as scipy_wasserstein
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchdyn.core import NeuralODE

from scatterprism.metric import (
    chi2_metric,
    correlation_matrix_distance,
    covariance_frobenius_distance,
    pairwise_chi2_2d,
)
from scatterprism.networks import (
    ConditionalFlowMatchingMLP,
    ConditionalFlowMatchingResNet,
    DiffusionMLP,
    FlowMatchingMLP,
    FlowMatchingResNet,
)

log = logging.getLogger(__name__)


# Valid DDPM training targets; validated in DDPM.__init__ so an invalid value
# fails at construction time rather than during the first sampling call.
_DDPM_PREDICTION_TYPES = ("epsilon", "x0", "v")


class _NFECounter(nn.Module):
    """Wraps a velocity net to count the number of forward evaluations.

    Used during validation to log NFE alongside distribution metrics for ODE
    solvers (Flow Matching).
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self._inner = net
        self.count = 0

    def forward(self, *args, **kwargs):
        self.count += 1
        return self._inner(*args, **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def sample_conditional_pt(
    x0: torch.Tensor, 
    x1: torch.Tensor, 
    t: torch.Tensor, 
    sigma: float = 0.0
) -> torch.Tensor:
    """
    Sample from the conditional probability path p_t(x|x0, x1).
    
    x_t = t * x1 + (1 - t) * x0 + sigma * noise
    
    Args:
        x0: Source samples [batch, dim]
        x1: Target samples [batch, dim]
        t: Time values [batch]
        sigma: Standard deviation of Gaussian noise
    
    Returns:
        Interpolated samples x_t [batch, dim]
    """
    t = t.view(-1, 1)  # [batch, 1]
    mu_t = t * x1 + (1 - t) * x0
    if sigma > 0:
        epsilon = torch.randn_like(x0)
        return mu_t + sigma * epsilon
    return mu_t


def compute_conditional_vector_field(
    x0: torch.Tensor, 
    x1: torch.Tensor
) -> torch.Tensor:
    """
    Compute conditional vector field u_t(x|x0, x1) = x1 - x0.
    
    Args:
        x0: Source samples [batch, dim]
        x1: Target samples [batch, dim]
    
    Returns:
        Vector field [batch, dim]
    """
    return x1 - x0


class ODEWrapper(nn.Module):
    """Wraps a velocity model for use with torchdyn NeuralODE."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, args=None) -> torch.Tensor:
        """
        Args:
            t: Scalar time
            x: State [batch, dim]
            args: Optional arguments (required by torchdyn, unused here)
        Returns:
            Velocity dx/dt [batch, dim]
        """
        batch_size = x.shape[0]
        t_batch = t.expand(batch_size)
        return self.model(x, t_batch)


class ConditionalODEWrapper(nn.Module):
    """Wraps a conditional velocity model for use with torchdyn NeuralODE.
    
    The conditioning is stored as a buffer and used during forward pass.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._cond = None
    
    def set_condition(self, cond: torch.Tensor):
        """Set the conditioning tensor before ODE integration."""
        self._cond = cond
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, args=None) -> torch.Tensor:
        """
        Args:
            t: Scalar time
            x: State [batch, dim]
            args: Optional arguments (required by torchdyn, unused here)
        Returns:
            Velocity dx/dt [batch, dim]
        """
        if self._cond is None:
            raise RuntimeError("Conditioning not set. Call set_condition() before ODE integration.")
        batch_size = x.shape[0]
        t_batch = t.expand(batch_size)
        return self.model(x, t_batch, self._cond)


# =============================================================================
# Base Model
# =============================================================================

class BaseGenerativeModel(pl.LightningModule):
    """
    Base class for generative models using PyTorch Lightning.
    
    Provides common functionality for training, validation, and sampling.
    """
    
    def __init__(
        self,
        data_dim: int,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        scheduler: str = "plateau",  # "plateau", "cosine", or "none"
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        grad_clip_val: float | None = None,
        metric_sample_size: int = 5000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dim = data_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.grad_clip_val = grad_clip_val
        self.metric_sample_size = metric_sample_size

        # For epoch-level loss aggregation
        self.training_step_outputs: list[torch.Tensor] = []
        self.validation_step_outputs: list[torch.Tensor] = []
        # Buffer to accumulate true validation data for distribution metrics
        self._val_data_buffer: list[torch.Tensor] = []
        # Buffer to accumulate source data (detector-level) for paired/denoise metrics
        self._val_source_buffer: list[torch.Tensor] = []
        # Last train epoch loss, used to compute overfit gap during validation
        self._last_train_loss_epoch: float = 0.0
        
        log.info(f"BaseGenerativeModel initialized with data_dim={data_dim}")

    def on_fit_start(self):
        """Register epoch-level WandB metrics to use epoch (not global_step) as x-axis."""
        try:
            if wandb.run is not None:
                wandb.define_metric("epoch")
                for metric in [
                    "train/loss_epoch",
                    "val/loss_epoch",
                    "val/overfit_gap",
                    "val/chi2_mean",
                    "val/ks_statistic_mean",
                    "val/wasserstein_mean",
                    "val/correlation_distance",
                    "val/covariance_distance",
                    "val/chi2_2d_mean",
                    "val/nfe",
                ]:
                    wandb.define_metric(metric, step_metric="epoch")
        except Exception as e:
            log.debug(f"WandB metric definition skipped: {e}")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        if self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=1e-7,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_epoch",  # Use epoch-level loss
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": False,  # Don't crash if metric not available
                },
            }
        elif self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs if self.trainer else 100
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm (clipping is handled exclusively by pl.Trainer via gradient_clip_val)."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.log('train/grad_norm', total_norm, prog_bar=False)
    
    def on_train_epoch_end(self):
        """Aggregate and log epoch-level training loss."""
        if self.training_step_outputs:
            epoch_loss = torch.stack(self.training_step_outputs).mean()
            self._last_train_loss_epoch = epoch_loss.item()
            self.log('epoch', float(self.current_epoch), prog_bar=False, sync_dist=False)
            self.log('train/loss_epoch', epoch_loss, prog_bar=True, sync_dist=True)
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Aggregate and log epoch-level validation loss and distribution metrics."""
        if self.validation_step_outputs:
            epoch_loss = torch.stack(self.validation_step_outputs).mean()
            self.log('epoch', float(self.current_epoch), prog_bar=False, sync_dist=False)
            self.log('val/loss_epoch', epoch_loss, prog_bar=True, sync_dist=True)
            overfit_gap = epoch_loss.item() - self._last_train_loss_epoch
            self.log('val/overfit_gap', overfit_gap, prog_bar=False, sync_dist=True)
            self.validation_step_outputs.clear()

        # Compute distribution quality metrics (chi2, KS, Wasserstein) on every validation run.
        # Frequency is already controlled by check_val_every_n_epoch (= trainer.val_interval).
        if self._val_data_buffer:
            self._compute_and_log_distribution_metrics()
        self._val_data_buffer.clear()
        self._val_source_buffer.clear()
    
    def _compute_and_log_distribution_metrics(self) -> None:
        """Generate samples and log per-feature chi2 / KS / Wasserstein metrics.

        Paired (denoise) mode reconstructs from source data; generation mode
        samples from noise.
        """
        # Concatenate buffered validation data on CPU
        val_data = torch.cat(self._val_data_buffer, dim=0)
        n = min(len(val_data), self.metric_sample_size)
        val_data = val_data[:n].numpy()
        
        # Check if we have source data (paired/denoise mode)
        has_source_data = len(self._val_source_buffer) > 0
        if has_source_data:
            source_data = torch.cat(self._val_source_buffer, dim=0)[:n]

        # Generate samples in eval mode (temporarily switch back after).
        # Wrap self.net (if present) with an NFE counter to measure ODE function
        # evaluations during sampling.
        nfe_counter = None
        original_net = None
        if hasattr(self, 'net'):
            nfe_counter = _NFECounter(self.net)
            original_net = self.net
            self.net = nfe_counter

        was_training = self.training
        self.eval()
        try:
            device = self.device
            # Use a fixed seed so the metric is reproducible across epochs and
            # does not add Monte-Carlo variance on top of genuine model changes.
            _rng_state = torch.get_rng_state()
            _cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            
            # For validation, use fast rk4 solver instead of dopri5 (avoids hanging with random weights)
            original_solver = getattr(self, 'solver', None)
            if original_solver is not None and original_solver not in ('euler', 'midpoint', 'rk4'):
                self.solver = 'rk4'
            
            try:
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                with torch.no_grad():
                    if has_source_data and hasattr(self, 'reconstruct'):
                        # Check if this is conditional CFM (starts from Gaussian, conditions on detector)
                        is_conditional = getattr(self, 'conditional', False)
                        if is_conditional:
                            # Conditional mode: start from Gaussian, condition on detector data
                            x0 = torch.randn(n, self.data_dim, device=device)
                            samples = self.reconstruct(x0, cond=source_data.to(device)).cpu().numpy()
                        else:
                            # Non-conditional denoise mode: reconstruct from source (detector) data
                            samples = self.reconstruct(source_data.to(device)).cpu().numpy()
                    else:
                        # Generation mode: sample from noise
                        samples = self.sample(n, device=device).cpu().numpy()
            finally:
                torch.set_rng_state(_rng_state)
                if _cuda_state is not None:
                    torch.cuda.set_rng_state_all(_cuda_state)
                # Restore original solver
                if original_solver is not None:
                    self.solver = original_solver
        except Exception as e:
            log.warning(f"Skipping distribution metrics — sample/reconstruct failed: {e}")
            return
        finally:
            # Always restore training mode and the original net wrapper.
            if was_training:
                self.train()
            if original_net is not None:
                self.net = original_net

        if samples.ndim == 1:
            samples = samples[:, None]
        if val_data.ndim == 1:
            val_data = val_data[:, None]

        num_features = val_data.shape[1]
        chi2_vals, ks_stats, wd_vals = [], [], []
        for i in range(num_features):
            truth_feat = val_data[:, i]
            gen_feat = samples[:, i]
            try:
                chi2_vals.append(float(chi2_metric(truth_feat, gen_feat)))
                ks_result = scipy_ks_2samp(truth_feat, gen_feat)
                ks_stats.append(float(ks_result.statistic))
                wd_vals.append(float(scipy_wasserstein(truth_feat, gen_feat)))
            except Exception as e:
                log.debug(f"Metric computation failed for feature {i}: {e}")

        if chi2_vals:
            self.log('epoch', float(self.current_epoch), prog_bar=False, sync_dist=False)
            self.log('val/chi2_mean', float(np.mean(chi2_vals)), prog_bar=True, sync_dist=True)
            self.log('val/ks_statistic_mean', float(np.mean(ks_stats)), prog_bar=True, sync_dist=True)
            self.log('val/wasserstein_mean', float(np.mean(wd_vals)), prog_bar=True, sync_dist=True)
            if nfe_counter is not None:
                self.log('val/nfe', float(nfe_counter.count), prog_bar=False, sync_dist=True)
            log.info(
                f"[Epoch {self.current_epoch}] Distribution metrics — "
                f"chi2={np.mean(chi2_vals):.4f}, "
                f"ks={np.mean(ks_stats):.4f}, "
                f"wasserstein={np.mean(wd_vals):.4f}"
                + (f", nfe={nfe_counter.count}" if nfe_counter is not None else "")
            )

        # Joint distribution metrics (correlation structure across all channels)
        if num_features > 1:
            try:
                corr_dist = correlation_matrix_distance(val_data, samples)
                cov_dist = covariance_frobenius_distance(val_data, samples)
                chi2_2d_results = pairwise_chi2_2d(val_data, samples)
                self.log('val/correlation_distance', corr_dist, prog_bar=False, sync_dist=True)
                self.log('val/covariance_distance', cov_dist, prog_bar=False, sync_dist=True)
                self.log('val/chi2_2d_mean', chi2_2d_results['chi2_2d_mean'], prog_bar=False, sync_dist=True)
                log.info(
                    f"[Epoch {self.current_epoch}] Joint distribution — "
                    f"corr_dist={corr_dist:.6f}, "
                    f"cov_dist={cov_dist:.6f}, "
                    f"chi2_2d_mean={chi2_2d_results['chi2_2d_mean']:.4f}"
                )
            except Exception as e:
                log.debug(f"Joint distribution metrics failed: {e}")

    @torch.no_grad()
    def sample(
        self,
        n_generate: int,
        device: torch.device | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples from the model. Override in subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def reconstruct(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct from source distribution. Override in subclasses."""
        raise NotImplementedError


# =============================================================================
# DDPM - Denoising Diffusion Probabilistic Model
# =============================================================================

class DDPM(BaseGenerativeModel):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    
    Learns to denoise samples by predicting the noise added at each timestep.
    Uses a linear beta schedule by default.
    
    Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dims: list[int] | int = 256,
        time_embed_dim: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",  # "linear", "cosine"
        prediction_type: str = "epsilon",  # "epsilon", "x0", "v"
        # Network options
        activation: str = "silu",  # "relu", "silu", "gelu", "tanh"
        norm: str | None = None,  # "layer", "batch", or None
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(data_dim=data_dim, **kwargs)
        self.save_hyperparameters()

        if prediction_type not in _DDPM_PREDICTION_TYPES:
            raise ValueError(
                f"prediction_type must be one of {_DDPM_PREDICTION_TYPES}, "
                f"got {prediction_type!r}."
            )

        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # Build network
        self.net = DiffusionMLP(
            data_dim=data_dim,
            time_embed_dim=time_embed_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        
        # Setup noise schedule
        self._setup_schedule(beta_start, beta_end, beta_schedule)
        
        log.info(f"DDPM initialized: timesteps={num_timesteps}, prediction={prediction_type}, activation={activation}, norm={norm}")
    
    def _setup_schedule(self, beta_start: float, beta_end: float, schedule: str):
        """Setup the diffusion noise schedule."""
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif schedule == "cosine":
            steps = self.num_timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers (moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Compute training loss.
        
        Args:
            x0: Clean data samples
            t: Timesteps
        
        Returns:
            loss: MSE between predicted and target (epsilon, x0, or v)
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        
        pred = self.net(x_t, t.float())
        
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x0
        elif self.prediction_type == "v":
            # v-prediction: v = sqrt(alpha) * noise - sqrt(1-alpha) * x0
            sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return F.mse_loss(pred, target)
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Handle paired data
        else:
            x = batch
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        loss = self.p_losses(x, t)
        
        # Log step-level loss
        self.log('train/loss', loss, prog_bar=True)
        # Store for epoch-level aggregation
        self.training_step_outputs.append(loss.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        loss = self.p_losses(x, t)
        
        # Log step-level loss
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        # Store for epoch-level aggregation
        self.validation_step_outputs.append(loss.detach())
        # Accumulate true data for distribution metrics
        self._val_data_buffer.append(x.detach().cpu())
        return loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Single reverse diffusion step."""
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        pred = self.net(x, t_batch.float())
        
        if self.prediction_type == "epsilon":
            pred_noise = pred
        elif self.prediction_type == "x0":
            # Convert x0 prediction to noise
            sqrt_alpha = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
            pred_noise = (x - sqrt_alpha * pred) / sqrt_one_minus_alpha
        elif self.prediction_type == "v":
            # Convert v prediction to noise
            # v = sqrt(alpha) * noise - sqrt(1-alpha) * x0
            # x_t = sqrt(alpha) * x0 + sqrt(1-alpha) * noise
            # Solving for noise: noise = sqrt(alpha) * v + sqrt(1-alpha) * x_t
            sqrt_alpha = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
            pred_noise = sqrt_alpha * pred + sqrt_one_minus_alpha * x
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Compute x_{t-1}
        beta_t = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha * pred_noise)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean
    
    @torch.no_grad()
    def sample(
        self,
        n_generate: int,
        device: torch.device | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion. ``cond`` is unused for DDPM."""
        device = device if device is not None else self.device
        x = torch.randn(n_generate, self.data_dim, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)
        
        return x


# =============================================================================
# CFM - Conditional Flow Matching
# =============================================================================

class CFM(BaseGenerativeModel):
    """
    Conditional Flow Matching (CFM).
    
    Learns a velocity field that transports samples from a source distribution
    (typically Gaussian) to the target data distribution.
    
    When conditional=True (for denoising/unfolding):
    - Network receives (x_t, t, cond) where cond is the detector-level data
    - Training: Gaussian → particle-level, conditioned on detector data
    - Inference: Generate from Gaussian, conditioned on detector data to unfold
    
    Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dims: list[int] | int = 256,
        time_embed_dim: int = 64,
        sigma: float = 0.0,  # Noise in interpolation path
        solver: str = "dopri5",  # ODE solver: "dopri5", "euler", "midpoint", "rk4"
        solver_atol: float = 1e-5,
        solver_rtol: float = 1e-5,
        solver_steps: int = 100,  # Fixed steps for euler/midpoint/rk4
        # Network options
        network_type: str = "mlp",  # "mlp" or "resnet"
        activation: str = "silu",  # "relu", "silu", "gelu", "tanh"
        norm: str | None = None,  # "layer", "batch", or None
        dropout: float = 0.0,
        time_embedding: str = "fourier",  # "fourier" or "learned" (legacy, for old checkpoints)
        # Conditional mode (for denoising/unfolding)
        conditional: bool = False,  # If True, use conditioning on detector data
        cond_dim: int | None = None,  # Conditioning dimension (defaults to data_dim)
        cond_embed_dim: int | None = None,  # Embedding dim for conditioning
        **kwargs,
    ):
        super().__init__(data_dim=data_dim, **kwargs)
        self.save_hyperparameters()
        
        self.sigma = sigma
        self.solver = solver
        self.solver_atol = solver_atol
        self.solver_rtol = solver_rtol
        self.solver_steps = solver_steps
        self.conditional = conditional
        self.cond_dim = cond_dim if cond_dim is not None else data_dim
        
        if conditional:
            # Use conditional networks for denoising/unfolding
            network_cls = {
                "mlp": ConditionalFlowMatchingMLP,
                "resnet": ConditionalFlowMatchingResNet,
            }[network_type]
            
            self.net = network_cls(
                data_dim=data_dim,
                cond_dim=self.cond_dim,
                time_embed_dim=time_embed_dim,
                cond_embed_dim=cond_embed_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                norm=norm,
                dropout=dropout,
                time_embedding=time_embedding,
            )
            log.info(f"CFM initialized (CONDITIONAL): network={network_type}, cond_dim={self.cond_dim}, activation={activation}, sigma={sigma}, solver={solver}, norm={norm}")
        else:
            # Standard unconditional networks
            network_cls = {
                "mlp": FlowMatchingMLP,
                "resnet": FlowMatchingResNet,
            }[network_type]
            
            self.net = network_cls(
                data_dim=data_dim,
                time_embed_dim=time_embed_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                norm=norm,
                dropout=dropout,
                time_embedding=time_embedding,
            )
            log.info(f"CFM initialized: network={network_type}, activation={activation}, sigma={sigma}, solver={solver}, norm={norm}")
    
    def compute_loss(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        cond: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        """
        Compute flow matching loss.
        
        Args:
            x0: Source samples (Gaussian noise)
            x1: Target samples (particle-level data)
            cond: Conditioning data (detector-level) - only used when self.conditional=True
            return_stats: If True, return additional statistics
        
        Returns:
            loss or (loss, stats_dict) if return_stats=True
        
        Loss components:
            - velocity_loss: MSE(v_θ(x_t, t, [cond]), u_t) where u_t = x1 - x0
        """
        batch_size = x1.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=x1.device)
        
        # Interpolate between noise (x0) and target (x1)
        x_t = sample_conditional_pt(x0, x1, t, self.sigma)
        
        # Target velocity (for linear interpolation)
        u_t = compute_conditional_vector_field(x0, x1)
        
        # Predicted velocity
        if self.conditional:
            if cond is None:
                raise ValueError("Conditioning required for conditional CFM")
            v_t = self.net(x_t, t, cond)
        else:
            v_t = self.net(x_t, t)
        
        # Velocity loss (main loss)
        velocity_loss = F.mse_loss(v_t, u_t)
        
        # Total loss
        total_loss = velocity_loss
        
        if return_stats:
            with torch.no_grad():
                v_t_norm = v_t.norm(dim=-1).mean()  # Mean velocity magnitude
                u_t_norm = u_t.norm(dim=-1).mean()  # Mean target velocity magnitude
                cos_sim = F.cosine_similarity(v_t, u_t, dim=-1).mean()  # Direction alignment
            stats = {
                'v_t_norm': v_t_norm,
                'u_t_norm': u_t_norm,
                'cos_sim': cos_sim,
            }
            return total_loss, stats
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        if self.conditional:
            # Conditional mode: Gaussian → particle_level, conditioned on detector_level
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x1, cond = batch[0], batch[1]  # particle_level, detector_level
                x0 = torch.randn_like(x1)      # Start from Gaussian
            else:
                raise ValueError("Conditional CFM requires paired data (particle, detector)")
            loss, stats = self.compute_loss(x0, x1, cond=cond, return_stats=True)
        else:
            # Non-conditional mode (original behavior)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Paired data: (particle_level, detector_level)
                x1, x0 = batch[0], batch[1]
            else:
                # Unpaired: sample from Gaussian
                x1 = batch if not isinstance(batch, (list, tuple)) else batch[0]
                x0 = torch.randn_like(x1)
            loss, stats = self.compute_loss(x0, x1, return_stats=True)
        
        # Log step-level loss and velocity stats
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/velocity_pred_norm', stats['v_t_norm'], prog_bar=False)
        self.log('train/velocity_target_norm', stats['u_t_norm'], prog_bar=False)
        self.log('train/velocity_cos_sim', stats['cos_sim'], prog_bar=False)
        if 'ot_cost' in stats:
            self.log('train/ot_cost', stats['ot_cost'], prog_bar=False)
        
        # Store for epoch-level aggregation
        self.training_step_outputs.append(loss.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.conditional:
            # Conditional mode: Gaussian → particle_level, conditioned on detector_level
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x1, cond = batch[0], batch[1]  # particle_level, detector_level
                x0 = torch.randn_like(x1)      # Start from Gaussian
            else:
                raise ValueError("Conditional CFM requires paired data (particle, detector)")
            loss, stats = self.compute_loss(x0, x1, cond=cond, return_stats=True)
            # Store detector data for validation metrics (sample generation)
            self._val_source_buffer.append(cond.detach().cpu())
        else:
            # Non-conditional mode (original behavior)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x1, x0 = batch[0], batch[1]
            else:
                x1 = batch if not isinstance(batch, (list, tuple)) else batch[0]
                x0 = torch.randn_like(x1)
            loss, stats = self.compute_loss(x0, x1, return_stats=True)
            # For paired/denoise mode, also buffer source data for correct metric computation
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                self._val_source_buffer.append(x0.detach().cpu())
        
        # Log step-level loss
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        
        # Store for epoch-level aggregation
        self.validation_step_outputs.append(loss.detach())
        # Accumulate true data for distribution metrics
        self._val_data_buffer.append(x1.detach().cpu())
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        n_generate: int,
        device: torch.device | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples by integrating the learned velocity field.

        Args:
            n_generate: Number of samples to generate.
            device:     Device to generate samples on.
            cond:       Conditioning data for conditional mode
                        ``[n_generate, cond_dim]``.
        """
        device = device if device is not None else self.device
        x0 = torch.randn(n_generate, self.data_dim, device=device)
        return self.reconstruct(x0, cond=cond)
    
    @torch.no_grad()
    def reconstruct(self, x0: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        Reconstruct/unfold from source samples.
        
        Memory-efficient: uses only start/end points, no intermediate trajectory storage.
        For fixed-step solvers (euler, midpoint, rk4), uses a manual loop which is
        much faster than torchdyn's NeuralODE.
        
        Args:
            x0: Source samples [batch, dim]
            cond: Conditioning data for conditional mode [batch, cond_dim]
        
        Returns:
            Reconstructed samples x1 [batch, dim]
        """
        # Validate conditioning for conditional mode
        if self.conditional:
            if cond is None:
                raise ValueError("Conditioning data required for conditional CFM")
            cond = cond.to(x0.device)
        
        # Use manual loop for fixed-step solvers (much faster than torchdyn)
        if self.solver in ("euler", "midpoint", "rk4"):
            n_steps = getattr(self, 'solver_steps', 100)
            dt = 1.0 / n_steps
            x = x0.clone()
            
            if self.solver == "euler":
                for step in range(n_steps):
                    t_val = float(step) / n_steps
                    t_b = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
                    if self.conditional:
                        x = x + dt * self.net(x, t_b, cond)
                    else:
                        x = x + dt * self.net(x, t_b)
            elif self.solver == "midpoint":
                for step in range(n_steps):
                    t_val = float(step) / n_steps
                    t_mid = t_val + dt / 2
                    t_b = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
                    t_b_mid = torch.full((x.shape[0],), t_mid, device=x.device, dtype=x.dtype)
                    if self.conditional:
                        k1 = self.net(x, t_b, cond)
                        x_mid = x + (dt / 2) * k1
                        x = x + dt * self.net(x_mid, t_b_mid, cond)
                    else:
                        k1 = self.net(x, t_b)
                        x_mid = x + (dt / 2) * k1
                        x = x + dt * self.net(x_mid, t_b_mid)
            elif self.solver == "rk4":
                for step in range(n_steps):
                    t_val = float(step) / n_steps
                    t_b = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
                    t_b_mid = torch.full((x.shape[0],), t_val + dt/2, device=x.device, dtype=x.dtype)
                    t_b_end = torch.full((x.shape[0],), t_val + dt, device=x.device, dtype=x.dtype)
                    if self.conditional:
                        k1 = self.net(x, t_b, cond)
                        k2 = self.net(x + (dt/2) * k1, t_b_mid, cond)
                        k3 = self.net(x + (dt/2) * k2, t_b_mid, cond)
                        k4 = self.net(x + dt * k3, t_b_end, cond)
                    else:
                        k1 = self.net(x, t_b)
                        k2 = self.net(x + (dt/2) * k1, t_b_mid)
                        k3 = self.net(x + (dt/2) * k2, t_b_mid)
                        k4 = self.net(x + dt * k3, t_b_end)
                    x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            return x
        
        # Use NeuralODE for adaptive solvers (dopri5, etc.)
        if self.conditional:
            wrapper = ConditionalODEWrapper(self.net)
            wrapper.set_condition(cond)
        else:
            wrapper = ODEWrapper(self.net)
            
        node = NeuralODE(
            wrapper,
            solver=self.solver,
            sensitivity="autograd",  # Lighter than adjoint for inference
            atol=self.solver_atol,
            rtol=self.solver_rtol,
            return_t_eval=False,  # Don't return time evaluations
        ).to(x0.device)
        
        # Only request start and end points
        t_span = torch.tensor([0.0, 1.0], device=x0.device)
        
        # Forward pass - node returns (t_eval, trajectory) or just trajectory
        result = node(x0, t_span)
        
        # Handle different return formats
        if isinstance(result, tuple):
            trajectory = result[1]
        else:
            trajectory = result
        
        # Return only final state, explicitly delete intermediate to free memory
        final_state = trajectory[-1].clone()
        del trajectory
        
        return final_state
    
    @torch.no_grad()
    def get_trajectory(
        self, 
        x0: torch.Tensor, 
        num_steps: int = 100,
        cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Get full trajectory from x0 to x1.
        
        WARNING: This stores all intermediate states and can use significant memory.
        Use reconstruct() for memory-efficient sampling.
        
        Args:
            x0: Source samples [batch, dim]
            num_steps: Number of trajectory points
            cond: Conditioning data for conditional mode [batch, cond_dim]
        
        Returns:
            Trajectory [num_steps, batch, dim]
        """
        # Validate conditioning for conditional mode
        if self.conditional:
            if cond is None:
                raise ValueError("Conditioning data required for conditional CFM")
            cond = cond.to(x0.device)
            wrapper = ConditionalODEWrapper(self.net)
            wrapper.set_condition(cond)
        else:
            wrapper = ODEWrapper(self.net)
            
        node = NeuralODE(
            wrapper,
            solver=self.solver,
            sensitivity="autograd",
            atol=self.solver_atol,
            rtol=self.solver_rtol,
        ).to(x0.device)
        
        t_span = torch.linspace(0.0, 1.0, num_steps, device=x0.device)
        result = node.trajectory(x0, t_span)
        
        # Handle different return formats from torchdyn
        if isinstance(result, tuple):
            trajectory = result[-1]  # Last element is typically the trajectory
        else:
            trajectory = result
        
        return trajectory
