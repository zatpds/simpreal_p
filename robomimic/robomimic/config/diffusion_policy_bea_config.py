"""
Config for Best-Effort Adaptation (BEA) Diffusion Policy algorithm.

Extends the standard DiffusionPolicyConfig with hyperparameters for
q-weighted ERM with discrepancy penalties and alternating minimization
over predictor parameters and per-sample weights.
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class BEADiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_bea"

    def algo_config(self):
        super(BEADiffusionPolicyConfig, self).algo_config()

        # ── BEA q-weight hyperparameters ──────────────────────────────
        # lambda_d: coefficient on discrepancy penalty  sum_i q_i * d_i
        self.algo.bea.lambda_d = 0.1
        # lambda_1: coefficient on L1 deviation from reference  ||q - p0||_1
        self.algo.bea.lambda_1 = 0.01
        # lambda_2: coefficient on L2 regularization  ||q||_2^2
        self.algo.bea.lambda_2 = 0.001
        # q_max: upper bound on each weight  0 <= q_i <= q_max
        self.algo.bea.q_max = 5.0
        # eta_q: step size for the projected subgradient update on q
        self.algo.bea.eta_q = 0.01

        # ── Discrepancy penalties d_i ─────────────────────────────────
        # d_i = discrepancy_value for source samples, 0 for target samples
        self.algo.bea.discrepancy_value = 1.0

        # ── Reference weights p_0 ────────────────────────────────────
        # p0_source: reference weight for each source sample (typically 0)
        self.algo.bea.p0_source = 0.0
        # p0_target_uniform: if True, p0_i = 1.0 for each target sample
        # if False, p0_i = p0_target for target samples
        self.algo.bea.p0_target_uniform = True
        # p0_target: reference weight for target samples when p0_target_uniform=False
        self.algo.bea.p0_target = 0.0

        # ── Target weight protection ─────────────────────────────────
        # target_q_min: lower bound on q for target samples (0.0 = no floor).
        # Without pre-training the model, diffusion loss starts at ~1.0 which
        # overwhelms all regularization terms and drives q to zero in one
        # step.  A nonzero floor prevents this deadlock so the model can
        # still learn from target data while source weights are adapted.
        self.algo.bea.target_q_min = 1.0

        # ── Sum constraint: sum_i q_i = n + alpha * m ─────────────────
        # This is what makes BEA actually use source data.  Without it
        # (and with p0_source=0), source weights stay at zero because the
        # L1 term only pulls toward p0=0.  The constraint forces total
        # weight mass to be distributed, so the projection pushes some
        # mass onto low-loss source samples.
        self.algo.bea.sum_constraint_enabled = True
        # alpha controls how much source mass is budgeted:
        #   target_sum = n + alpha * m
        # alpha=0 → only target mass, alpha=1 → full source+target mass.
        self.algo.bea.sum_constraint_alpha = 0.5

        # ── q update schedule ─────────────────────────────────────────
        # update q every K epochs (1 = every epoch, faithful to algorithm)
        self.algo.bea.q_update_every = 1

        # ── Loss EMA for variance reduction ───────────────────────────
        # Diffusion loss is high-variance (random noise + timestep each call).
        # If loss_ema_beta > 0, an EMA of per-sample losses is maintained and
        # used for the q subgradient instead of the raw single-sample estimate.
        # Set to 0.0 to disable (use raw losses, faithful to Algorithm 3).
        self.algo.bea.loss_ema_beta = 0.0
