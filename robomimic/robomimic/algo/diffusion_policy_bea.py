"""
BEA

The combined dataset U = S ∪ T is reindexed so that indices 0..m-1 are
source and m..m+n-1 are target.  The weight vector q ∈ R^{m+n} lives on
CPU and is sliced to GPU per mini-batch via the "index" field that
MetaDataset already provides in each sample.

Design decisions:
  - q is stored as a flat CPU tensor of size (m+n,).  It is NOT a learnable
    nn.Parameter — it is updated by an explicit projected subgradient rule,
    not by autograd.
  - The per-sample diffusion loss is MSE between predicted and true noise,
    averaged over the (time, action_dim) axes but NOT over the batch axis.
    This gives a (B,) vector used for both q-update and q-weighted training.
  - The dataset must include an "index" key in each batch (MetaDataset does
    this automatically).  process_batch_for_training passes it through.
  - The bounded simplex projection (for the optional sum constraint) uses
    bisection on the Lagrange multiplier, which is exact up to tolerance.
  - sign(0) is treated as 0 (a valid subgradient of |u| at u=0).
"""

from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


@register_algo_factory_func("diffusion_policy_bea")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return BEADiffusionPolicyUNet, {}
    else:
        raise RuntimeError("BEA currently only supports UNet backbone")


class BEADiffusionPolicyUNet(DiffusionPolicyUNet):
    """
    BEA Diffusion Policy with UNet backbone,
    """

    def _create_networks(self):
        super()._create_networks()
        self.q = None        # (m+n,) CPU float tensor
        self.p0 = None       # (m+n,) reference weights
        self.d = None         # (m+n,) discrepancy penalties
        self.m = 0            # number of source samples
        self.n = 0            # number of target samples
        self.loss_ema = None  # (m+n,) EMA of per-sample losses (None if disabled)
        self._bea_initialized = False

    #### BEA weight initialization --------------------------------------------------------------------------------

    def init_bea_weights(self, m, n):
        """
        Initialize per-sample weights q, reference weights p0, and
        discrepancy penalties d for the combined dataset of size m+n.

        Must be called after the model is created and before training starts.

        Args:
            m (int): number of source samples (indices 0..m-1)
            n (int): number of target samples (indices m..m+n-1)
        """
        cfg = self.algo_config.bea
        self.m = m
        self.n = n
        total = m + n

        # Reference weights p0
        # When p0_target_uniform=True, each target sample gets p0=1.0 (not
        # 1/n).  This makes the L1 regularization lambda_1*||q-p0||_1 a
        # meaningful anchor: it pulls target q back toward 1.0 once the
        # model is trained and loss is small enough for the pull to matter.
        p0 = torch.zeros(total, dtype=torch.float32)
        p0[:m] = cfg.p0_source
        if cfg.p0_target_uniform:
            p0[m:] = 1.0
        else:
            p0[m:] = cfg.p0_target
        self.p0 = p0

        # Discrepancy penalties: d_hat > 0 for source, 0 for target
        d = torch.zeros(total, dtype=torch.float32)
        d[:m] = cfg.discrepancy_value
        self.d = d

        # Initialize q = p0 (algorithm line 3)
        self.q = p0.clone()

        # Loss EMA buffer for variance reduction (optional)
        if cfg.loss_ema_beta > 0:
            self.loss_ema = torch.zeros(total, dtype=torch.float32)
        else:
            self.loss_ema = None

        self._bea_initialized = True
        ema_str = f"beta={cfg.loss_ema_beta:.3f}" if self.loss_ema is not None else "off"
        print(
            f"[BEA] Initialized weights: m={m}, n={n}, q_shape=({total},), "
            f"loss_ema={ema_str}, target_q_min={cfg.target_q_min}"
        )

    #### Batch processing — pass through sample indices ------------------------------------------------------------

    def process_batch_for_training(self, batch):
        """
        Extends parent to also pass through the "index" field, which is
        needed to look up q weights per sample.
        """
        input_batch = super().process_batch_for_training(batch)
        if "index" in batch:
            input_batch["index"] = batch["index"].long().to(self.device)
        return input_batch

    #### Per-sample diffusion loss (no batch reduction) ------------------------------------------------------------

    def _diffusion_forward(self, batch):
        """
        Run the diffusion forward pass and return per-sample MSE losses.

        Returns:
            per_sample_loss: (B,) tensor — MSE averaged over (Tp, ac_dim)
                but not over batch.
            info: dict with intermediate tensors for logging.
        """
        actions = batch["actions"]
        B = actions.shape[0]

        inputs = {"obs": batch["obs"], "goal": batch["goal_obs"]}
        for k in self.obs_shapes:
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )
        obs_cond = obs_features.flatten(start_dim=1)

        noise = torch.randn(actions.shape, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self.nets["policy"]["noise_pred_net"](
            noisy_actions, timesteps, global_cond=obs_cond
        )

        # Per-sample MSE: average over (time, action_dim), keep batch dim
        # noise_pred and noise have shape (B, Tp, ac_dim)
        per_sample_loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2))

        return per_sample_loss, {"noise_pred": noise_pred, "noise": noise}

    #### q-update step  ------------------------------------------------------------------------------------------

    @torch.no_grad()
    def update_q(self, batch_indices, per_sample_losses):
        """
        Projected subgradient step on q for the given batch indices.

        For each i in batch_indices:
            g_i = l_i + lambda_d * d_i + lambda_1 * sign(q_i - p0_i) + 2 * lambda_2 * q_i
            q_i = q_i - eta_q * g_i
            q_i = clip(q_i, 0, q_max)

        If loss_ema_beta > 0, the EMA of losses is updated first and the
        smoothed loss is used in the subgradient instead of the raw loss.
        This reduces variance from random diffusion timestep/noise sampling.

        If target_q_min > 0, target sample weights are clamped from below
        to prevent them from collapsing to zero.

        The optional sum-constraint projection is NOT applied here — it is
        applied once after the full pass over the dataset (see
        project_q_simplex).

        Args:
            batch_indices: (B,) long tensor (CPU or GPU) — global dataset indices
            per_sample_losses: (B,) float tensor (CPU or GPU) — l(h(x_i), y_i)
        """
        cfg = self.algo_config.bea
        idx = batch_indices.cpu()
        losses = per_sample_losses.detach().cpu()

        # Optional: update and use loss EMA for variance reduction
        if self.loss_ema is not None:
            beta = cfg.loss_ema_beta
            self.loss_ema[idx] = beta * self.loss_ema[idx] + (1.0 - beta) * losses
            losses_for_grad = self.loss_ema[idx]
        else:
            losses_for_grad = losses

        q = self.q[idx]
        p0 = self.p0[idx]
        d = self.d[idx]

        # Subgradient of the full objective w.r.t. q_i
        # sign(0) = 0 which is a valid subgradient of |u| at u=0
        grad = (
            losses_for_grad
            + cfg.lambda_d * d
            + cfg.lambda_1 * torch.sign(q - p0)
            + 2.0 * cfg.lambda_2 * q
        )

        q = q - cfg.eta_q * grad
        q = q.clamp(0.0, cfg.q_max)

        # Enforce target weight floor if configured
        target_q_min = cfg.target_q_min
        if target_q_min > 0:
            target_mask = idx >= self.m
            q[target_mask] = q[target_mask].clamp(min=target_q_min)

        self.q[idx] = q

    def project_q_simplex(self):
        """
        Project q onto the constraint set
            {q : sum(q) = target_sum, lo_i <= q_i <= q_max}
        where target_sum = n + alpha * m, and lo_i = target_q_min for
        target samples (i >= m) and 0 for source samples (i < m).

        Uses bisection on the Lagrange multiplier nu:
            q_i^*(nu) = clip(q_i - nu, lo_i, q_max)
        Find nu such that sum_i q_i^*(nu) = target_sum.

        Only called if sum_constraint_enabled is True.
        """
        cfg = self.algo_config.bea
        if not cfg.sum_constraint_enabled:
            # Even without the sum constraint, enforce target_q_min
            if cfg.target_q_min > 0 and self.n > 0:
                self.q[self.m:] = self.q[self.m:].clamp(min=cfg.target_q_min)
            return

        target_sum = float(self.n) + cfg.sum_constraint_alpha * float(self.m)
        q = self.q
        q_max = cfg.q_max

        # Per-sample lower bounds: target_q_min for target, 0 for source
        q_lo = torch.zeros_like(q)
        if cfg.target_q_min > 0 and self.n > 0:
            q_lo[self.m:] = cfg.target_q_min
        q_hi = torch.full_like(q, q_max)

        # Feasibility check
        min_possible = q_lo.sum().item()
        max_possible = q_hi.sum().item()
        if target_sum < min_possible or target_sum > max_possible:
            print(
                f"[BEA] WARNING: target_sum={target_sum:.2f} infeasible "
                f"(range [{min_possible:.2f}, {max_possible:.2f}]). Clamping."
            )
            target_sum = max(min_possible, min(max_possible, target_sum))

        # Bisection: find nu such that sum(clip(q - nu, q_lo, q_hi)) = target_sum
        # sum is non-increasing in nu.
        lo_nu = float(q.min()) - q_max - 1.0
        hi_nu = float(q.max()) + 1.0

        for _ in range(64):
            mid = (lo_nu + hi_nu) / 2.0
            projected = torch.clamp(q - mid, min=q_lo, max=q_hi)
            current_sum = projected.sum().item()
            if current_sum > target_sum:
                lo_nu = mid
            else:
                hi_nu = mid
            if abs(current_sum - target_sum) < 1e-6:
                break

        nu = (lo_nu + hi_nu) / 2.0
        self.q = torch.clamp(q - nu, min=q_lo, max=q_hi)

    #### q-weighted training step  --------------------------------------------------------------------------------

    def train_on_batch(self, batch, epoch, validate=False):
        """
        q-weighted training step.

        Computes per-sample diffusion loss, weights by q[index], averages
        over the batch, and does a gradient step on the model parameters.

        Returns an info dict with losses and q-statistics for logging.
        """
        assert self._bea_initialized, (
            "BEA weights not initialized. Call model.init_bea_weights(m, n) "
            "before training."
        )

        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = OrderedDict()
            assert validate or self.nets.training

            per_sample_loss, _ = self._diffusion_forward(batch)

            # Look up q weights for this batch
            indices = batch["index"].long()
            q_batch = self.q[indices.cpu()].to(self.device).detach()

            # Weighted mean loss: (1/B) * sum_i q_i * l_i
            weighted_loss = (q_batch * per_sample_loss).mean()
            # Also track unweighted loss for comparison
            unweighted_loss = per_sample_loss.mean()

            losses = {
                "l2_loss": unweighted_loss,
                "weighted_loss": weighted_loss,
                "q_mean": q_batch.mean(),
                "q_std": q_batch.std() if B > 1 else torch.tensor(0.0),
            }

            # Per-domain q stats
            src_mask = indices < self.m
            tgt_mask = ~src_mask
            if src_mask.any():
                losses["q_source_mean"] = q_batch[src_mask].mean()
            if tgt_mask.any():
                losses["q_target_mean"] = q_batch[tgt_mask].mean()

            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=weighted_loss,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)

                info["policy_grad_norms"] = policy_grad_norms

        return info

    #### Compute losses for q-update pass (no gradient on model params) ------------------------------------------

    @torch.no_grad()
    def compute_losses_for_q_update(self, batch):
        """
        Forward pass to compute per-sample losses for the q-update phase.
        Called with model in eval mode and torch.no_grad().

        Args:
            batch: processed + postprocessed batch dict with "index" key.

        Returns:
            indices: (B,) long tensor — global dataset indices
            per_sample_losses: (B,) float tensor — per-sample MSE
        """
        per_sample_loss, _ = self._diffusion_forward(batch)
        indices = batch["index"].long()
        return indices, per_sample_loss

    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["l2_loss"].item()
        log["Weighted_Loss"] = info["losses"]["weighted_loss"].item()
        log["q_mean"] = info["losses"]["q_mean"].item()
        log["q_std"] = info["losses"]["q_std"].item()
        if "q_source_mean" in info["losses"]:
            log["q_source_mean"] = info["losses"]["q_source_mean"].item()
        if "q_target_mean" in info["losses"]:
            log["q_target_mean"] = info["losses"]["q_target_mean"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    #### Serialization — save / load q weights alongside model ------------------------------------------------------

    def serialize(self):
        d = super().serialize()
        d["bea"] = {
            "q": self.q,
            "p0": self.p0,
            "d": self.d,
            "m": self.m,
            "n": self.n,
            "loss_ema": self.loss_ema,
        }
        return d

    def deserialize(self, model_dict, load_optimizers=False):
        super().deserialize(model_dict, load_optimizers=load_optimizers)
        if "bea" in model_dict:
            bea = model_dict["bea"]
            self.q = bea["q"]
            self.p0 = bea["p0"]
            self.d = bea["d"]
            self.m = bea["m"]
            self.n = bea["n"]
            self.loss_ema = bea.get("loss_ema", None)
            self._bea_initialized = True
