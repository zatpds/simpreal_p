"""
ManiSkill environment wrapper for robomimic.

Wraps ManiSkill3 gymnasium environments so that they can be used seamlessly
with robomimic's training and evaluation pipelines.  Observations produced by
the ManiSkill SAPIEN simulator are automatically converted to the *robosuite
world-frame convention* so that a policy trained on mixed robosuite + ManiSkill
data sees a single consistent observation space.

Frame alignment (SAPIEN → robosuite):
  - Rotation:    −90° around the Z (up) axis
  - Translation: +z_offset (≈ 0.812 m, the robosuite table height)
  - Gripper:     finger-1 sign negated to match robosuite convention
  - Quaternions: wxyz → xyzw, then rotated

See ``ms_collect/frame_alignment.md`` for derivation details.
"""

import json
import cv2
import numpy as np
from copy import deepcopy

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

# ---------------------------------------------------------------------------
#  Frame-transform constants (SAPIEN → robosuite)
# ---------------------------------------------------------------------------

# Rotation matrix: −90° around Z
_R_MS2RS = np.array([
    [ 0.,  1.,  0.],
    [-1.,  0.,  0.],
    [ 0.,  0.,  1.],
], dtype=np.float64)

# Same rotation as a unit quaternion in (x, y, z, w) convention
_Q_MS2RS_XYZW = np.array([0., 0., -np.sin(np.pi / 4), np.cos(np.pi / 4)],
                          dtype=np.float64)

# Default Z offset (robosuite table height + mount)
_DEFAULT_Z_OFFSET = 0.8122


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------

def _wxyz_to_xyzw(quat):
    """Convert quaternion(s) from SAPIEN (w,x,y,z) to robosuite (x,y,z,w)."""
    if quat.ndim == 1:
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    return np.concatenate([quat[:, 1:4], quat[:, 0:1]], axis=1).astype(np.float64)


def _quat_multiply_xyzw(q1, q2):
    """Hamilton product of two (x,y,z,w) quaternions.  q1 is always (4,)."""
    x1, y1, z1, w1 = q1
    if q2.ndim == 1:
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float64)
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=1).astype(np.float64)


def _frame_pos(pos, z_offset):
    """Rotate + translate position from SAPIEN frame to robosuite frame."""
    rotated = (pos @ _R_MS2RS.T).astype(np.float64)
    rotated[..., 2] += z_offset
    return rotated


def _frame_quat(quat_xyzw):
    """Rotate quaternion from SAPIEN frame to robosuite frame."""
    return _quat_multiply_xyzw(_Q_MS2RS_XYZW, quat_xyzw)


def _frame_delta(delta):
    """Rotate a delta vector (no translation)."""
    return (delta @ _R_MS2RS.T).astype(np.float64)


def _construct_object_vector(cubeA_pos, cubeA_quat_xyzw,
                             cubeB_pos, cubeB_quat_xyzw,
                             eef_pos):
    """Build the 23-D ``object`` vector matching robosuite Stack."""
    parts = [
        cubeA_pos, cubeA_quat_xyzw,
        cubeB_pos, cubeB_quat_xyzw,
        cubeA_pos - eef_pos,
        cubeB_pos - eef_pos,
        cubeB_pos - cubeA_pos,
    ]
    if cubeA_pos.ndim == 1:
        return np.concatenate(parts).astype(np.float64)
    return np.concatenate(parts, axis=1).astype(np.float64)


def _to_numpy(x):
    """Convert torch Tensor (or numpy) to numpy.  Squeeze leading batch dim if 1."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x)
    if x.ndim > 1 and x.shape[0] == 1:
        x = x[0]
    return x


def _resize(img, h, w):
    """Resize a single HWC uint8 image."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h))


# ---------------------------------------------------------------------------
#  ManiSkill environment wrapper
# ---------------------------------------------------------------------------

class EnvManiSkill(EB.EnvBase):
    """Wrapper class for ManiSkill3 environments (https://maniskill.ai)."""

    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
        lang=None,
        **kwargs,
    ):
        """
        Args:
            env_name (str): ManiSkill environment id, e.g. ``StackCube-v1``.
            render (bool): on-screen rendering (unused for headless training).
            render_offscreen (bool): off-screen rendering for video recording.
            use_image_obs (bool): include camera RGB observations.
            use_depth_obs (bool): include depth observations (not yet used).
            lang: language instruction (unused).
            **kwargs: forwarded to ``gymnasium.make`` (e.g. control_mode,
                obs_mode, camera_heights, camera_widths, …).
        """
        import gymnasium as gym
        import mani_skill.envs  # register envs

        self._env_name = env_name
        self._use_image_obs = use_image_obs or render_offscreen
        self._use_depth_obs = use_depth_obs

        # Pop robomimic-specific / conversion metadata that ManiSkill doesn't understand
        kwargs = deepcopy(kwargs)
        kwargs.pop("controller_configs", None)
        kwargs.pop("camera_names", None)

        # Desired output image size (robomimic convention)
        self._camera_h = kwargs.pop("camera_heights", 96)
        self._camera_w = kwargs.pop("camera_widths", 96)

        # Frame-transform parameters
        self._z_offset = kwargs.pop("z_offset", _DEFAULT_Z_OFFSET)

        # Build ManiSkill gymnasium env
        ms_kwargs = dict(
            obs_mode=kwargs.pop("obs_mode", "rgbd"),
            control_mode=kwargs.pop("control_mode", "pd_ee_delta_pose"),
            render_mode=kwargs.pop("render_mode", "rgb_array"),
        )
        ms_kwargs.update(kwargs)
        self._init_kwargs = deepcopy(ms_kwargs)
        self._init_kwargs["camera_heights"] = self._camera_h
        self._init_kwargs["camera_widths"] = self._camera_w
        self._init_kwargs["z_offset"] = self._z_offset

        # Override max_episode_steps so gymnasium's TimeLimit wrapper does
        # not truncate the episode.  Robomimic controls episode length via
        # its own rollout horizon, matching how EnvRobosuite works
        # (ignore_done=True).
        self.env = gym.make(env_name, max_episode_steps=10_000, **ms_kwargs)

        self._current_obs_raw = None  # last raw gym obs (nested dict with tensors)
        self._current_reward = None
        self._current_info = {}

    # ------------------------------------------------------------------
    #  Core API
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Step the environment.  The *action* is expected in the **robosuite
        frame** (consistent with the training data).  We convert it back to
        the SAPIEN frame before forwarding to ManiSkill.

        Returns:
            observation (dict), reward (float), done (bool), info (dict)
        """
        # Convert action from robosuite frame → SAPIEN frame
        action = np.asarray(action, dtype=np.float64).copy()
        # Inverse of −90° around Z is +90° around Z → transpose of _R_MS2RS
        action[:3] = action[:3] @ _R_MS2RS  # inverse rotation for delta pos
        action[3:6] = action[3:6] @ _R_MS2RS  # inverse rotation for delta rot
        # gripper (action[6]) is unchanged

        obs_raw, reward, terminated, truncated, info = self.env.step(action.astype(np.float32))
        self._current_obs_raw = obs_raw
        self._current_reward = float(_to_numpy(reward))
        self._current_info = info

        obs = self.get_observation(obs_raw)
        info["is_success"] = self.is_success()
        # Always return done=False; robomimic controls episode length via
        # its rollout horizon, just like EnvRobosuite (ignore_done=True).
        return obs, self._current_reward, self.is_done(), info

    def reset(self):
        """Reset the environment, return initial observation."""
        obs_raw, info = self.env.reset()
        self._current_obs_raw = obs_raw
        self._current_reward = 0.0
        self._current_done = False
        self._current_info = info
        return self.get_observation(obs_raw)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): must contain ``states`` key with a flat numpy array
                previously returned by ``get_state()``.

        Returns:
            observation (dict) or None
        """
        import torch
        if "states" in state:
            state_tensor = torch.tensor(state["states"]).float().unsqueeze(0)
            self.env.unwrapped.set_state(state_tensor)
            obs_raw = self.env.unwrapped.get_obs()
            self._current_obs_raw = obs_raw
            return self.get_observation(obs_raw)
        return None

    # ------------------------------------------------------------------
    #  Observation conversion (SAPIEN → robosuite frame)
    # ------------------------------------------------------------------

    def get_observation(self, obs_raw=None):
        """
        Convert ManiSkill's nested observation dict to robomimic's flat dict
        with keys matching the robosuite Stack task, in the robosuite frame.
        """
        if obs_raw is None:
            obs_raw = self._current_obs_raw
        if obs_raw is None:
            obs_raw = self.env.unwrapped.get_obs()

        obs = {}

        # --- Proprioception ---
        qpos = _to_numpy(obs_raw["agent"]["qpos"])  # (9,)
        qvel = _to_numpy(obs_raw["agent"]["qvel"])  # (9,)

        obs["robot0_joint_pos"] = qpos[:7].astype(np.float64)
        obs["robot0_joint_vel"] = qvel[:7].astype(np.float64)

        # Gripper: negate finger-1 to match robosuite sign convention
        gripper_qpos = qpos[7:9].astype(np.float64).copy()
        gripper_qvel = qvel[7:9].astype(np.float64).copy()
        gripper_qpos[1] = -gripper_qpos[1]
        gripper_qvel[1] = -gripper_qvel[1]
        obs["robot0_gripper_qpos"] = gripper_qpos
        obs["robot0_gripper_qvel"] = gripper_qvel

        # --- EEF pose (tcp_pose is wxyz quaternion in SAPIEN) ---
        tcp = _to_numpy(obs_raw["extra"]["tcp_pose"])  # (7,) = pos(3) + quat_wxyz(4)
        eef_pos = tcp[:3].astype(np.float64)
        eef_quat_xyzw = _wxyz_to_xyzw(tcp[3:7].astype(np.float64))

        eef_pos = _frame_pos(eef_pos, self._z_offset)
        eef_quat_xyzw = _frame_quat(eef_quat_xyzw)

        obs["robot0_eef_pos"] = eef_pos
        obs["robot0_eef_quat"] = eef_quat_xyzw

        # --- Object vector (if cube poses available) ---
        extra = obs_raw.get("extra", {})
        has_cubeA = "cubeA_pose" in extra
        has_cubeB = "cubeB_pose" in extra
        if has_cubeA and has_cubeB:
            cA = _to_numpy(extra["cubeA_pose"])  # (7,) pos + wxyz
            cB = _to_numpy(extra["cubeB_pose"])

            cubeA_pos = _frame_pos(cA[:3].astype(np.float64), self._z_offset)
            cubeA_q   = _frame_quat(_wxyz_to_xyzw(cA[3:7].astype(np.float64)))
            cubeB_pos = _frame_pos(cB[:3].astype(np.float64), self._z_offset)
            cubeB_q   = _frame_quat(_wxyz_to_xyzw(cB[3:7].astype(np.float64)))

            obs["object"] = _construct_object_vector(
                cubeA_pos, cubeA_q, cubeB_pos, cubeB_q, eef_pos)

        # --- Camera images ---
        sensor = obs_raw.get("sensor_data", {})
        if "base_camera" in sensor:
            rgb = _to_numpy(sensor["base_camera"]["rgb"])  # (H, W, 3) uint8
            obs["agentview_image"] = _resize(rgb, self._camera_h, self._camera_w)
        if "hand_camera" in sensor:
            rgb = _to_numpy(sensor["hand_camera"]["rgb"])
            obs["robot0_eye_in_hand_image"] = _resize(rgb, self._camera_h, self._camera_w)

        return obs

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render the environment.

        In ``rgb_array`` mode, returns an (H, W, 3) uint8 numpy array.

        Args:
            camera_name: which camera to use.  Supported values:
                ``"base_camera"`` / ``"agentview"`` – front view (default),
                ``"hand_camera"`` / ``"robot0_eye_in_hand"`` – wrist camera.
                If *None*, defaults to the base (front) camera so that
                evaluation videos match the agent's training viewpoint.
        """
        if mode == "human":
            return self.env.render()
        elif mode == "rgb_array":
            # --- try to use the requested sensor camera from observations ---
            # Map robomimic-style names → ManiSkill sensor names
            _cam_map = {
                "agentview": "base_camera",
                "robot0_eye_in_hand": "hand_camera",
            }
            ms_cam = _cam_map.get(camera_name, camera_name) if camera_name else "base_camera"

            obs_raw = self._current_obs_raw
            sensor = obs_raw.get("sensor_data", {}) if obs_raw is not None else {}
            if ms_cam in sensor and "rgb" in sensor[ms_cam]:
                frame = _to_numpy(sensor[ms_cam]["rgb"])
            else:
                # Fallback: use the default human render camera
                frame = self.env.render()
                frame = _to_numpy(frame)

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            h = height or self._camera_h
            w = width or self._camera_w
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            return frame
        else:
            raise NotImplementedError(f"mode={mode!r} is not implemented")

    # ------------------------------------------------------------------
    #  State / reward / success helpers
    # ------------------------------------------------------------------

    def get_state(self):
        """Return current simulator state as a flat numpy array (in a dict)."""
        state = self.env.unwrapped.get_state()
        return dict(states=_to_numpy(state).astype(np.float64))

    def get_reward(self):
        """Get current reward."""
        if self._current_reward is not None:
            return self._current_reward
        return 0.0

    def get_goal(self):
        """Get goal observation.  Not supported for ManiSkill tasks."""
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """Set goal.  Not supported for ManiSkill tasks."""
        raise NotImplementedError

    def is_done(self):
        """Check if the task is done."""
        # During rollout, robomimic handles the horizon externally —
        # returning False keeps behaviour consistent with EnvRobosuite.
        return False

    def is_success(self):
        """
        Check task success.

        Returns:
            dict with at least ``{"task": bool}``.
        """
        info = self._current_info
        if isinstance(info, dict) and "success" in info:
            success = info["success"]
            success = bool(_to_numpy(np.asarray(success)))
            return {"task": success}
        return {"task": False}

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def action_dimension(self):
        return self.env.action_space.shape[0]

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return EB.EnvType.MANISKILL_TYPE

    @property
    def version(self):
        try:
            import mani_skill
            return mani_skill.__version__
        except Exception:
            return None

    def serialize(self):
        """
        Return metadata dict sufficient to re-create this environment.
        This is the ``env_meta`` stored in HDF5 datasets.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs),
        )

    # ------------------------------------------------------------------
    #  Data processing (optional)
    # ------------------------------------------------------------------

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        has_camera = len(camera_names) > 0

        # Map robomimic camera names to image obs keys
        image_modalities = []
        for cn in camera_names:
            if cn in ("agentview", "base_camera"):
                image_modalities.append("agentview_image")
            elif cn in ("robot0_eye_in_hand", "hand_camera"):
                image_modalities.append("robot0_eye_in_hand_image")
            else:
                image_modalities.append(f"{cn}_image")

        obs_modality_specs = {
            "obs": {
                "low_dim": [],
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(
            env_name=env_name,
            render=(False if render is None else render),
            render_offscreen=(has_camera if render_offscreen is None else render_offscreen),
            use_image_obs=(has_camera if use_image_obs is None else use_image_obs),
            use_depth_obs=(False if use_depth_obs is None else use_depth_obs),
            camera_heights=camera_height,
            camera_widths=camera_width,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """Exceptions to catch during rollouts so training doesn't crash."""
        return (Exception,)

    @property
    def base_env(self):
        return self.env

    def __repr__(self):
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
