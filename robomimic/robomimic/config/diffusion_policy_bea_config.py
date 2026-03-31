"""
Config for BEA Diffusion Policy

Extends the standard DiffusionPolicyConfig
"""

from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig


class BEADiffusionPolicyConfig(DiffusionPolicyConfig):
    ALGO_NAME = "diffusion_policy_bea"

    def algo_config(self):
        super(BEADiffusionPolicyConfig, self).algo_config()

        #### q - weight hyperparameters
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

        #### discrepancy penalties d_i 
        # d_i = discrepancy_value for source samples, 0 for target samples
        self.algo.bea.discrepancy_value = 1.0
        # dynamic_d: if True, compute per-sample d_i via k-NN distance in
        # obs-encoder feature space (updated every q_update_every epochs).
        # if False, use the flat constant discrepancy_value for all source.
        self.algo.bea.dynamic_d = False
        # k for k-NN distance computation (only used when dynamic_d=True)
        self.algo.bea.dynamic_d_k = 3

        #### reference weights p_0
        # p0_source: reference weight for each source sample (typically 0)
        self.algo.bea.p0_source = 0.0
        # p0_target_uniform: if True, p0_i = 1.0 for each target sample
        # if False, p0_i = p0_target for target samples
        self.algo.bea.p0_target_uniform = True
        # p0_target: reference weight for target samples when p0_target_uniform=False
        self.algo.bea.p0_target = 0.0

        #### target weight protection
        # target_q_min: lower bound on q for target samples (0.0 = no floor).
        # w/o pre-training, diffusion loss starts at ~1.0 which
        # overwhelms all regularization terms and drives q to zero in one
        # step.  A nonzero floor prevents this deadlock so the model can
        # still learn from target data while source weights are adapted.
        self.algo.bea.target_q_min = 1.0

        #### sum constraint: sum_i q_i = n + alpha * m 
        # this is what makes BEA actually use source data.  w/o it
        # (and with p0_source=0), source weights stay at zero because the
        # L1 term only pulls toward p0=0.  The constraint forces total
        # weight mass to be distributed, so the projection pushes some
        # mass onto low-loss source samples.
        self.algo.bea.sum_constraint_enabled = True
        # alpha controls how much source mass is budgeted:
        #   target_sum = n + alpha * m
        # alpha=0 → only target mass, alpha=1 → full source+target mass.
        self.algo.bea.sum_constraint_alpha = 0.5

        #### q update mode
        # "full"     — solve the global convex optimization problem for q
        #              (paper-faithful: argmin over the full dataset)
        # "minibatch" — projected subgradient descent on q per mini-batch
        self.algo.bea.q_update_mode = "full"

        #### q update schedule
        # update q every K epochs (1 = every epoch, faithful to algorithm)
        self.algo.bea.q_update_every = 1

        #### loss EMA for variance reduction
        # if loss_ema_beta > 0, an EMA of per-sample losses is maintained and
        # used for the q subgradient instead of the raw single-sample estimate.
        # Set to 0.0 to disable (use raw losses, faithful to Algorithm 3).
        self.algo.bea.loss_ema_beta = 0.0
