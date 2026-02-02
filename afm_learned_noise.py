"""
Adversarial Flow Matching with Learned Noise Prior
===================================================

Extension of adversarial flow matching where the noise distribution (prior)
is learned rather than fixed to N(0, I). This allows the model to potentially
find a better matching between the prior and data manifold.

Key modifications from base adversarial flow:
- LearnedGaussianPrior: Parameterizes prior as N(mu, diag(exp(log_var)))
- KL regularization: Prevents prior from collapsing (KL to standard normal)
- Joint training: Prior parameters optimized alongside generator

The intuition is that optimal transport cost depends on the choice of source
distribution. A learned prior could reduce the total transport distance by
better aligning with the data geometry.
"""

from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange

# =============================================================================
# Device Configuration
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# Data Generation
# =============================================================================

def gen_data(n: int, device: torch.device = DEVICE) -> Tensor:
    """Generate 2D mixture of 8 Gaussians arranged in a circle."""
    scale = 4.0
    centers = torch.tensor([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=torch.float32, device=device) * scale

    x = 0.5 * torch.randn(n, 2, device=device)
    center_ids = torch.randint(0, 8, (n,), device=device)
    x = (x + centers[center_ids]) / np.sqrt(2)
    return x

# =============================================================================
# Flow Path Primitives
# =============================================================================
#
# We use the standard linear interpolation path:
#     z_t = (1 - t) * x_0 + t * x_T
#
# where x_0 ~ p_data and x_T ~ N(0, I).
# At t=0, z_0 = x_0 (data). At t=1, z_1 = x_T (noise).
#

def sample_noise(n: int, d: int, device: torch.device = DEVICE) -> Tensor:
    """Sample from standard prior N(0, I)."""
    return torch.randn(n, d, device=device)


def interpolate(x_0: Tensor, x_T: Tensor, t: Tensor) -> Tensor:
    """Linear interpolation: z_t = (1-t)*x_0 + t*x_T."""
    return x_0 * (1 - t) + x_T * t


# =============================================================================
# Time Sampling
# =============================================================================

def sample_times(
    n: int,
    nfe: int,
    device: torch.device = DEVICE
) -> tuple[Tensor, Tensor]:
    """
    Sample discrete (t_src, t_tgt) pairs for NFE-step generation.

    Args:
        n: Batch size
        nfe: Number of function evaluations (steps)
        device: Torch device

    Returns:
        t_src: Source time in [1/nfe, 1]
        t_tgt: Target time = t_src - 1/nfe
    """
    steps = torch.randint(1, nfe + 1, (n, 1), dtype=torch.float32, device=device)
    t_src = steps / nfe
    t_tgt = t_src - (1.0 / nfe)
    return t_src, t_tgt


# =============================================================================
# Learned Noise Prior via Normalizing Flow
# =============================================================================
#
# A normalizing flow transforms a simple base distribution (N(0,I)) through
# a sequence of invertible mappings. This allows learning arbitrary densities.
#
# In the adversarial setting, no KL regularization is needed - the generator's
# adversarial loss backpropagates through the flow, training it to produce
# noise that minimizes transport cost to the data.
#

class AffineCoupling(nn.Module):
    """
    Affine coupling layer: splits input, uses one half to predict
    scale/shift for the other half.

    For z = [z_a, z_b]:
        z'_a = z_a
        z'_b = z_b * exp(s(z_a)) + t(z_a)

    Invertible with tractable Jacobian: det = exp(sum(s(z_a)))
    """

    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.d_half = d // 2

        self.net = nn.Sequential(
            nn.Linear(self.d_half, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, (d - self.d_half) * 2),
        )
        # Initialize last layer near-zero for near-identity initialization
        # Small noise ensures non-zero gradients from the start
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Forward: base -> target. Returns (z', log_det)."""
        z_a, z_b = z[:, :self.d_half], z[:, self.d_half:]
        st = self.net(z_a)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)  # Bound scale for stability

        z_b_new = z_b * s.exp() + t
        z_new = torch.cat([z_a, z_b_new], dim=-1)
        log_det = s.sum(dim=-1)
        return z_new, log_det

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Inverse: target -> base. Returns (z_base, -log_det)."""
        z_a, z_b = z[:, :self.d_half], z[:, self.d_half:]
        st = self.net(z_a)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)

        z_b_base = (z_b - t) * (-s).exp()
        z_base = torch.cat([z_a, z_b_base], dim=-1)
        log_det = -s.sum(dim=-1)
        return z_base, log_det


class FlowPrior(nn.Module):
    """
    Normalizing flow prior: transforms N(0, I) into learned distribution.

    Uses alternating affine coupling layers with permutations. The flow
    can represent arbitrary continuous densities (universal approximation).

    In adversarial training, gradients from the generator loss flow back
    through sample() to train the prior - no explicit density matching needed.
    """

    def __init__(self, d: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.n_layers = n_layers

        # Build flow layers with alternating permutations
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(AffineCoupling(d, hidden_dim))

        # Fixed random permutations (alternating halves)
        self.register_buffer(
            "perm",
            torch.stack([torch.randperm(d) for _ in range(n_layers)])
        )
        self.register_buffer(
            "perm_inv",
            torch.stack([torch.argsort(self.perm[i]) for i in range(n_layers)])
        )

    def forward(self, z_base: Tensor) -> tuple[Tensor, Tensor]:
        """Transform base samples to prior samples. Returns (z_prior, log_det)."""
        z = z_base
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        for i, layer in enumerate(self.layers):
            z = z[:, self.perm[i]]  # Permute
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z_prior: Tensor) -> tuple[Tensor, Tensor]:
        """Transform prior samples back to base. Returns (z_base, -log_det)."""
        z = z_prior
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        for i in range(self.n_layers - 1, -1, -1):
            z, log_det = self.layers[i].inverse(z)
            total_log_det = total_log_det + log_det
            z = z[:, self.perm_inv[i]]  # Unpermute

        return z, total_log_det

    def sample(self, n: int, device: torch.device = DEVICE) -> Tensor:
        """Sample from the learned prior."""
        z_base = torch.randn(n, self.d, device=device)
        z_prior, _ = self.forward(z_base)
        return z_prior

    def log_prob(self, z_prior: Tensor) -> Tensor:
        """Compute log p(z) under the flow."""
        z_base, neg_log_det = self.inverse(z_prior)
        # log p(z_prior) = log p_base(z_base) + log |det dz_base/dz_prior|
        #                = log p_base(z_base) - log |det dz_prior/dz_base|
        log_p_base = -0.5 * (z_base.square().sum(dim=-1) + self.d * np.log(2 * np.pi))
        return log_p_base + neg_log_det

    def entropy_lower_bound(self, n_samples: int = 256) -> Tensor:
        """
        Monte Carlo estimate of entropy: H[q] = -E_q[log q(z)]

        Useful for regularization to prevent prior collapse.
        """
        z = self.sample(n_samples, device=self.perm.device)
        return -self.log_prob(z).mean()


class VAEEncoder(nn.Module):
    """
    VAE encoder: x → z via reparameterization.

    Maps data to (μ, log_σ²), samples z = μ + σ * ε where ε ~ N(0,I).
    KL divergence to N(0,I) regularizes the latent space.
    """

    def __init__(self, d: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode x → (z, μ, log_σ²)."""
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z, mu, logvar

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """KL divergence to N(0,I): 0.5 * sum(σ² + μ² - 1 - log σ²)."""
        return 0.5 * (logvar.exp() + mu.square() - 1 - logvar).sum(dim=-1).mean()


class ConditionalAffineCoupling(nn.Module):
    """
    Affine coupling conditioned on external signal c.

    Network takes [z_a, c] as input to predict scale/shift for z_b.
    """

    def __init__(self, d: int, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.d_half = d // 2
        self.cond_dim = cond_dim

        # Condition on both z_a and external conditioning c
        self.net = nn.Sequential(
            nn.Linear(self.d_half + cond_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, (d - self.d_half) * 2),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Forward: base -> target. Returns (z', log_det)."""
        z_a, z_b = z[:, :self.d_half], z[:, self.d_half:]
        st = self.net(torch.cat([z_a, c], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)

        z_b_new = z_b * s.exp() + t
        z_new = torch.cat([z_a, z_b_new], dim=-1)
        log_det = s.sum(dim=-1)
        return z_new, log_det

    def inverse(self, z: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Inverse: target -> base."""
        z_a, z_b = z[:, :self.d_half], z[:, self.d_half:]
        st = self.net(torch.cat([z_a, c], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)

        z_b_base = (z_b - t) * (-s).exp()
        z_base = torch.cat([z_a, z_b_base], dim=-1)
        log_det = -s.sum(dim=-1)
        return z_base, log_det


class ConditionalFlowPrior(nn.Module):
    """
    Conditional normalizing flow.

    Given conditioning c (e.g., VAE latent), transforms N(0,I) → learned prior.

    Training: c = VAE.encode(x), flow transforms noise conditioned on c
    Inference: c ~ N(0,I), flow transforms noise conditioned on c
    """

    def __init__(self, d: int, cond_dim: int, n_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        self.cond_dim = cond_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(d, cond_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # Fixed permutations
        self.register_buffer(
            "perm",
            torch.stack([torch.randperm(d) for _ in range(n_layers)])
        )
        self.register_buffer(
            "perm_inv",
            torch.stack([torch.argsort(self.perm[i]) for i in range(n_layers)])
        )

    def forward(self, z_base: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Transform base samples to prior samples given conditioning c."""
        z = z_base
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        for i, layer in enumerate(self.layers):
            z = z[:, self.perm[i]]
            z, log_det = layer(z, c)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z_prior: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Transform prior samples back to base given conditioning c."""
        z = z_prior
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        for i in range(self.n_layers - 1, -1, -1):
            z, log_det = self.layers[i].inverse(z, c)
            total_log_det = total_log_det + log_det
            z = z[:, self.perm_inv[i]]

        return z, total_log_det

    def sample(self, c: Tensor) -> Tensor:
        """Sample from conditional prior given c."""
        z_base = torch.randn(c.shape[0], self.d, device=c.device)
        z_prior, _ = self.forward(z_base, c)
        return z_prior


class VAEFlowPrior(nn.Module):
    """
    VAE + Conditional Flow prior.

    Training:
        z_latent, μ, logvar = vae.encode(x)  # VAE latent from data
        x_T = flow.sample(z_latent)           # conditional flow sample
        Generator trains on x_T

    Inference:
        z_latent ~ N(0,I)                     # sample from VAE prior
        x_T = flow.sample(z_latent)           # conditional flow sample
        x = G(x_T)                            # generate
    """

    def __init__(self, d: int, latent_dim: int = None, hidden_dim: int = 64, flow_layers: int = 4):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim if latent_dim is not None else d

        self.vae = VAEEncoder(d, self.latent_dim, hidden_dim)
        self.flow = ConditionalFlowPrior(d, self.latent_dim, flow_layers, hidden_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode data to VAE latent."""
        return self.vae(x)

    def sample(self, n: int, device: torch.device = DEVICE) -> Tensor:
        """Sample: z ~ N(0,I), then conditional flow."""
        z_latent = torch.randn(n, self.latent_dim, device=device)
        return self.flow.sample(z_latent)

    def sample_given_latent(self, z_latent: Tensor) -> Tensor:
        """Sample from flow given VAE latent."""
        return self.flow.sample(z_latent)

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """KL loss from VAE."""
        return self.vae.kl_loss(mu, logvar)


def interpolate(x_0: Tensor, x_T: Tensor, t: Tensor) -> Tensor:
    """Linear interpolation: z_t = (1-t)*x_0 + t*x_T."""
    return x_0 * (1 - t) + x_T * t

# =============================================================================
# Time Sampling
# =============================================================================

def sample_times(
    n: int,
    nfe: int,
    device: torch.device = DEVICE
) -> tuple[Tensor, Tensor]:
    """
    Sample discrete (t_src, t_tgt) pairs for NFE-step generation.

    Args:
        n: Batch size
        nfe: Number of function evaluations (steps)
        device: Torch device

    Returns:
        t_src: Source time in [1/nfe, 1]
        t_tgt: Target time = t_src - 1/nfe
    """
    # Sample integer steps from {1, 2, ..., nfe}
    steps = torch.randint(1, nfe + 1, (n, 1), dtype=torch.float32, device=device)
    t_src = steps / nfe
    t_tgt = t_src - (1.0 / nfe)
    return t_src, t_tgt

def sample_times_continuous(n: int, min_step: float = 0.05, device=DEVICE):
    """Sample continuous (t_src, t_tgt) pairs."""
    t_src = torch.rand(n, 1, device=device)  # Uniform [0, 1]
    step_size = torch.rand(n, 1, device=device) * min_step + min_step  # Random step
    t_tgt = (t_src - step_size).clamp(min=0)
    
    # Ensure t_src > t_tgt
    mask = t_src <= t_tgt
    t_src[mask] = t_tgt[mask] + step_size[mask]
    
    return t_src.clamp(max=1), t_tgt

def sample_times_mixed(n: int, max_nfe: int, device=DEVICE):
    """Sample with random NFE per batch."""
    # Random NFE from 1 to max_nfe for this batch
    nfe = torch.randint(1, max_nfe + 1, (1,)).item()
    
    steps = torch.randint(1, nfe + 1, (n, 1), dtype=torch.float32, device=device)
    t_src = steps / nfe
    t_tgt = t_src - (1.0 / nfe)
    return t_src, t_tgt
# =============================================================================
# Neural Network Architecture
# =============================================================================

class Net(nn.Module):
    """Simple MLP with SELU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Generator(nn.Module):
    """
    Generator for adversarial flow matching.

    Takes (x_src, t_src, t_tgt) and predicts x_tgt, the transported sample.
    """

    def __init__(self, d: int, hidden_dim: int = 256):
        super().__init__()
        # Input: [x_src (d), t_src (1), t_tgt (1)]
        self.net = Net(d + 2, d, hidden_dim)

    def forward(self, x_src: Tensor, t_src: Tensor, t_tgt: Tensor) -> Tensor:
        """Transport x_src from t_src to t_tgt."""
        inp = torch.cat([x_src, t_src, t_tgt], dim=1)
        return self.net(inp)


class Discriminator(nn.Module):
    """
    Discriminator for adversarial flow matching.

    Takes (x, t) and outputs a scalar logit indicating whether x
    is a real sample from the marginal q(z_t) at time t.
    """

    def __init__(self, d: int, hidden_dim: int = 256):
        super().__init__()
        # Input: [x (d), t (1)]
        self.net = Net(d + 1, 1, hidden_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Output logit for (x, t) pair."""
        inp = torch.cat([x, t], dim=1)
        return self.net(inp).squeeze(-1)

# =============================================================================
# Adversarial Flow Model
# =============================================================================

class AdversarialFlow(nn.Module):
    """
    Adversarial Flow Matching with Learned Noise Prior.

    Trains a generator to transport samples along the flow path using
    adversarial training. The discriminator learns to distinguish real
    marginal samples from transported ones.

    Key extension: The noise prior is learned rather than fixed to N(0,I).
    This allows the model to find a source distribution that minimizes
    transport cost to the data manifold.

    Loss components:
    - Relativistic adversarial loss: log(sigmoid(D_real - D_fake))
    - OT regularization: ||x_tgt - x_src||^2 / step_size (straight paths)
    - R1/R2 gradient penalty via finite differences
    - Logit centering: (D_real + D_fake)^2
    - KL regularization on prior: prevents degenerate priors
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 256,
        prior_type: str = "encoder",  # "encoder", "flow", or "fixed"
        prior_layers: int = 4,
        prior_hidden: int = 64,
    ):
        super().__init__()
        self.d = d
        self.prior_type = prior_type

        self.gen = Generator(d, hidden_dim)
        self.dis = Discriminator(d, hidden_dim)
        self.gen_ema = deepcopy(self.gen)

        # Prior options
        if prior_type == "vae":
            self.prior = VAEFlowPrior(d, latent_dim=d, hidden_dim=prior_hidden, flow_layers=prior_layers)
        elif prior_type == "flow":
            self.prior = FlowPrior(d, n_layers=prior_layers, hidden_dim=prior_hidden)
        else:
            self.prior = None

        # Freeze EMA model
        for p in self.gen_ema.parameters():
            p.requires_grad_(False)

    def update_ema(self, decay: float = 0.99):
        """Update EMA generator weights."""
        with torch.no_grad():
            for p_gen, p_ema in zip(self.gen.parameters(), self.gen_ema.parameters()):
                p_ema.data.lerp_(p_gen.data, 1 - decay)

    def compute_dis_loss(
        self,
        x_tgt_real: Tensor,
        x_tgt_fake: Tensor,
        t_tgt: Tensor,
        gp_scale: float = 0.01,
        gp_eps: float = 0.01,
        cp_scale: float = 0.01,
        step_weight: Tensor | None = None,
    ) -> Tensor:
        """
        Compute discriminator loss.

        Args:
            x_tgt_real: Real samples at t_tgt
            x_tgt_fake: Generator outputs (detached)
            t_tgt: Target timestep
            gp_scale: Gradient penalty scale
            gp_eps: Noise scale for finite-difference GP
            cp_scale: Logit centering scale
            step_weight: Per-sample weight based on step size
        """
        logits_real = self.dis(x_tgt_real, t_tgt)
        logits_fake = self.dis(x_tgt_fake.detach(), t_tgt)

        # Perturbed logits for approximated gradient penalty
        logits_real_r1 = self.dis(x_tgt_real + gp_eps * torch.randn_like(x_tgt_real), t_tgt)
        logits_fake_r2 = self.dis(x_tgt_fake.detach() + gp_eps * torch.randn_like(x_tgt_fake), t_tgt)

        if step_weight is None:
            step_weight = torch.ones_like(logits_real)

        # Relativistic adversarial loss
        adv_loss = F.softplus(-(logits_real - logits_fake)).mean()

        # Approximated R1 regularization (gradient penalty on real samples)
        r1_loss = ((logits_real - logits_real_r1).square() * step_weight / gp_eps ** 2).mean()

        # Approximated R2 regularization (gradient penalty on fake samples)
        r2_loss = ((logits_fake - logits_fake_r2).square() * step_weight / gp_eps ** 2).mean()

        # Logit centering regularization
        cp_loss = ((logits_real + logits_fake).square()).mean()

        return adv_loss + gp_scale * (r1_loss + r2_loss) + cp_scale * cp_loss

    def compute_gen_loss(
        self,
        x_src: Tensor,
        x_tgt_fake: Tensor,
        t_tgt: Tensor,
        logits_real: Tensor,
        ot_scale: float = 0.05,
        step_weight: Tensor | None = None,
    ) -> Tensor:
        """
        Compute generator loss.

        Args:
            x_src: Source samples
            x_tgt_fake: Generator outputs
            t_tgt: Target timestep
            logits_real: Discriminator logits on real samples
            ot_scale: Optimal transport regularization scale
            step_weight: Per-sample weight based on step size
        """
        logits_fake = self.dis(x_tgt_fake, t_tgt)

        if step_weight is None:
            step_weight = torch.ones_like(logits_fake)

        # Relativistic adversarial loss (reversed)
        adv_loss = F.softplus(-(logits_fake - logits_real.detach())).mean()

        # OT regularization: encourage straight paths
        # Weight inversely by step size (smaller steps should have smaller displacements)
        ot_loss = (((x_tgt_fake - x_src).square().sum(dim=1)) / step_weight.squeeze(-1)).mean()

        return adv_loss + ot_scale * ot_loss

    def sample_prior(self, n: int) -> Tensor:
        """Sample from prior (learned or standard)."""
        if self.prior is not None:
            return self.prior.sample(n, device=next(self.parameters()).device)
        return sample_noise(n, self.d, device=next(self.parameters()).device)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        nfe: int = 1,
        use_ema: bool = True
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Generate samples via multi-step transport.

        Args:
            n: Number of samples
            nfe: Number of function evaluations (steps)
            use_ema: Whether to use EMA generator

        Returns:
            Final samples and (timesteps, path) for visualization
        """
        gen = self.gen_ema if use_ema else self.gen
        device = next(self.parameters()).device

        # Start from learned prior at t=1
        z = self.sample_prior(n)

        # Timesteps from t=1 to t=0
        t_steps = torch.linspace(1.0, 0.0, nfe + 1, device=device)
        path = [z]

        for i in range(nfe):
            t_src = t_steps[i].expand(n, 1)
            t_tgt = t_steps[i + 1].expand(n, 1)
            z = gen(z, t_src, t_tgt)
            path.append(z)

        return z, (t_steps, torch.stack(path))

# =============================================================================
# Visualization
# =============================================================================

_plot_counter = [0]


def _get_filename(prefix: str, filename: str | None) -> str:
    if filename is None:
        _plot_counter[0] += 1
        return f"{prefix}_{_plot_counter[0]}.jpg"
    return filename


def viz_2d_data(data: Tensor, filename: str | None = None):
    """Save 2D scatter plot."""
    plt.figure()
    data = data.cpu()
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    plt.axis("scaled")
    plt.savefig(_get_filename("data_2d", filename), format="jpg", dpi=150, bbox_inches="tight")
    plt.close()


def viz_2d_path(
    t_steps: Tensor,
    path: Tensor,
    n_lines: int = -1,
    color: str | None = None,
    filename: str | None = None
):
    """Save 2D trajectory plot with time-offset visualization."""
    plt.figure(figsize=(12, 12))
    t_steps, path = t_steps.cpu(), path.cpu()

    plt.scatter(15 + path[0, :, 0], path[0, :, 1], s=1, label="t=1 (noise)")
    plt.scatter(path[-1, :, 0], path[-1, :, 1], s=1, label="t=0 (data)")
    plt.plot(
        15 * t_steps[:, None] + path[:, :n_lines, 0],
        path[:, :n_lines, 1],
        color=color, alpha=0.5
    )
    plt.axis("scaled")
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.savefig(_get_filename("path_2d", filename), format="jpg", dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# Training
# =============================================================================

def train(
    model: AdversarialFlow,
    data_fn: Callable[[int], Tensor],
    n_iter: int = 50000,
    batch_size: int = 1024,
    lr: float = 1e-4,
    prior_lr: float | None = None,
    betas: tuple[float, float] = (0.0, 0.9),
    nfe: int = 1,
    ot_scale: float = 0.05,
    gp_scale: float = 0.01,
    gp_eps: float = 0.01,
    cp_scale: float = 0.01,
    entropy_scale: float = 0.0,
    ema_decay: float = 0.99,
    sample_every: int = 1000,
    n_samples: int = 1024,
):
    """
    Train the adversarial flow model with learned prior.

    Uses alternating discriminator/generator updates. The prior is trained
    jointly with the generator - adversarial gradients flow through the
    flow prior, teaching it to produce noise that minimizes transport cost.

    Optional entropy regularization (entropy_scale > 0) prevents prior collapse
    by encouraging high entropy in the learned distribution.
    """
    if prior_lr is None:
        prior_lr = lr

    # Generator + prior optimized together
    gen_params = list(model.gen.parameters())
    if model.prior is not None:
        prior_params = list(model.prior.parameters())
        # Use separate param groups for potentially different LR
        gen_optimizer = torch.optim.Adam([
            {"params": gen_params, "lr": lr},
            {"params": prior_params, "lr": prior_lr},
        ], betas=betas)
    else:
        gen_optimizer = torch.optim.Adam(gen_params, lr=lr, betas=betas)

    dis_optimizer = torch.optim.Adam(model.dis.parameters(), lr=lr, betas=betas)

    pbar = trange(n_iter)
    d = model.d
    device = next(model.parameters()).device

    for step in pbar:
        # Sample data
        x_0 = data_fn(batch_size)

        # Sample timesteps
        t_src, t_tgt = sample_times(batch_size, nfe=nfe, device=device)

        # Step weight: larger steps get higher weight
        step_weight = (t_src - t_tgt).abs().clamp_min(0.001)

        if step % 2 == 0:
            # Discriminator step
            model.dis.requires_grad_(True)
            model.gen.requires_grad_(False)
            if model.prior is not None:
                model.prior.requires_grad_(False)

            with torch.no_grad():
                # Sample from prior (no grad for D step)
                if model.prior is not None:
                    x_T = model.prior.sample(batch_size, device=device)
                else:
                    x_T = sample_noise(batch_size, d, device=device)

                x_src = interpolate(x_0, x_T, t_src)
                x_tgt_real = interpolate(x_0, x_T, t_tgt)
                x_tgt_fake = model.gen(x_src, t_src, t_tgt)

            dis_loss = model.compute_dis_loss(
                x_tgt_real, x_tgt_fake, t_tgt,
                gp_scale=gp_scale, gp_eps=gp_eps, cp_scale=cp_scale,
                step_weight=step_weight
            )

            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            if (step + 1) % 100 == 0:
                pbar.set_description(f"D: {dis_loss.item():.4f}")
        else:
            # Generator + Prior step
            model.dis.requires_grad_(False)
            model.gen.requires_grad_(True)
            if model.prior is not None:
                model.prior.requires_grad_(True)

            # Sample from prior WITH gradients
            kl_loss_val = 0.0
            if model.prior_type == "vae":
                # VAE: encode data → z_latent, then conditional flow
                z_latent, mu, logvar = model.prior.encode(x_0)
                x_T = model.prior.sample_given_latent(z_latent)
            elif model.prior is not None:
                x_T = model.prior.sample(batch_size, device=device)
            else:
                x_T = sample_noise(batch_size, d, device=device)

            x_src = interpolate(x_0, x_T, t_src)
            x_tgt_real = interpolate(x_0, x_T, t_tgt)
            x_tgt_fake = model.gen(x_src, t_src, t_tgt)

            logits_real = model.dis(x_tgt_real, t_tgt)

            gen_loss = model.compute_gen_loss(
                x_src, x_tgt_fake, t_tgt, logits_real,
                ot_scale=ot_scale, step_weight=step_weight
            )

            # VAE KL loss
            if model.prior_type == "vae":
                kl_loss = model.prior.kl_loss(mu, logvar)
                gen_loss = gen_loss + 0.1 * kl_loss  # weight can be tuned
                kl_loss_val = kl_loss.item()

            # Optional entropy regularization (for flow prior)
            if entropy_scale > 0 and model.prior_type == "flow":
                entropy_loss = -entropy_scale * model.prior.entropy_lower_bound(batch_size)
                gen_loss = gen_loss + entropy_loss

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # Update EMA
            model.update_ema(ema_decay)

            if (step + 1) % 100 == 0:
                # Compute prior statistics
                stats_str = ""
                if model.prior is not None:
                    with torch.no_grad():
                        z_prior = model.prior.sample(256, device=device)
                        mean_dev = z_prior.mean().abs().item()
                        std_val = z_prior.std().item()
                        if model.prior_type == "vae":
                            # Show KL loss and prior stats
                            stats_str = f" | kl:{kl_loss_val:.2f} μ:{mean_dev:.2f} σ:{std_val:.2f}"
                        else:
                            # Show log-det for flow prior
                            z_base = torch.randn(256, d, device=device)
                            _, log_det = model.prior.forward(z_base)
                            stats_str = f" | ld:{log_det.mean().item():+.2f} μ:{mean_dev:.2f} σ:{std_val:.2f}"
                pbar.set_description(f"G: {gen_loss.item():.4f}{stats_str}")

        # Visualize samples periodically
        if (step + 1) % sample_every == 0:
            model.eval()

            # Visualize prior
            if model.prior is not None:
                with torch.no_grad():
                    # Prior samples (what we sample from at inference)
                    prior_samples = model.prior.sample(n_samples, device=device)
                    viz_2d_data(prior_samples, filename="prior_samples.jpg")

                    # For VAE prior, also show VAE latents
                    if model.prior_type == "vae":
                        data_batch = data_fn(n_samples)
                        z_latent, _, _ = model.prior.encode(data_batch)
                        viz_2d_data(z_latent, filename="vae_latents.jpg")

            _, (t_steps, path) = model.sample(n_samples, nfe=1)
            viz_2d_path(t_steps, path, n_lines=0, color="red", filename="adv_flow_1step.jpg")

            if nfe > 1:
                _, (t_steps, path) = model.sample(n_samples, nfe=nfe)
                viz_2d_path(t_steps, path, n_lines=16, color="green", filename=f"adv_flow_{nfe}step.jpg")

            model.train()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Visualize data distribution
    viz_2d_data(gen_data(4096), filename="data.jpg")

    # Initialize model with VAE + conditional flow prior
    model = AdversarialFlow(
        d=2,
        hidden_dim=256,
        prior_type="vae",  # "vae", "flow", or "fixed"
        prior_layers=4,
        prior_hidden=64,
    ).to(DEVICE)

    print(f"Generator params: {sum(p.numel() for p in model.gen.parameters()):,}")
    print(f"Prior params: {sum(p.numel() for p in model.prior.parameters()):,}")
    print(f"  - VAE encoder: {sum(p.numel() for p in model.prior.vae.parameters()):,}")
    print(f"  - Cond. flow: {sum(p.numel() for p in model.prior.flow.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in model.dis.parameters()):,}")

    train(
        model,
        gen_data,
        n_iter=50000,
        batch_size=1024,
        nfe=1,
        ot_scale=0.05,
        prior_lr=1e-4,
    )

    # Generate final samples
    model.eval()

    with torch.no_grad():
        # Prior samples (inference distribution)
        prior_samples = model.prior.sample(4096, device=DEVICE)
        viz_2d_data(prior_samples, filename="prior_samples_final.jpg")

        # VAE latents (what the encoder produces during training)
        data_batch = gen_data(4096)
        z_latent, _, _ = model.prior.encode(data_batch)
        viz_2d_data(z_latent, filename="vae_latents_final.jpg")

    # 1-step generation
    z_1step, (t_steps, path) = model.sample(4096, nfe=1)
    viz_2d_data(z_1step, filename="samples_final.jpg")
    viz_2d_path(t_steps, path, n_lines=32, color="red", filename="path_final.jpg")

