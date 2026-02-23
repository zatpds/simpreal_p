from collections import OrderedDict

import numpy as np
import os
import copy

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robomimic.assets.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
import robosuite.utils.transform_utils as T
from robosuite.models.objects import BoxObject, CylinderObject

import robosuite_model_zoo
from robosuite_model_zoo.utils.mjcf_obj import MJCFObject
from robomimic.assets.objects.composite_body import BinObject

from robomimic.envs.robosuite.stack import modify_xml_for_camera_movement
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

import mimicgen
XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")

class Put_in_Bin_RL2(SingleArmEnv_MG):
    """
    This class corresponds to the mug insertion task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        gripper_visualizations=None,  # Roboteleop issue
        bin_size = [0.144, 0.144, 0.143],
        bin_thickness = 0.01,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.912)) # will be override

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.initial_joint_pos = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

        self.bin_size = bin_size
        self.bin_thickness = bin_thickness

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
        # z_val = np.random.uniform(-0.01, 0.01, ())
        self.table_offset = [0, 0, 0.912]
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # load bin
        self.bin_object = BinObject(
            name=f"bin",
            bin_size=self.bin_size,
            wall_thickness=self.bin_thickness,
            transparent_walls=False,
            density=1000.
        )

        self.objects = [self.bin_object]
        # Create placement initializer

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.objs_bound = self._get_initial_placement_bounds()

        bin_ranges = self.objs_bound["bin"]
        x, y, z_rot = bin_ranges["x"], bin_ranges["y"], bin_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"BinSampler",
                mujoco_objects=self.bin_object,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001
            ),
        )

        self.load_other_objects()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def load_other_objects(self):
        pass

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.bin_body_id = self.sim.model.body_name2id(self.bin_object.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.robot_joints = self.robots[0].robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.initial_joint_pos

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name == "bin":  # additional rotation
                    # rot_angle = np.pi/2
                    # obj_quat = T.quat_multiply(np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0]), obj_quat) 
                    self.bin_initial_pos = obj_pos
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # simulate for a few steps to let the peg settle
        for _ in range(100):
            self.sim.step()
            self.sim.forward()

        self.robot_joints = self.robots[0].robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.initial_joint_pos

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def bin_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.bin_body_id])

            @sensor(modality=modality)
            def bin_quat(obs_cache):
                return T.convert_quat(np.array(self.sim.data.body_xquat[self.bin_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_bin(obs_cache):
                return (
                    obs_cache["bin_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "bin_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [bin_pos, bin_quat, gripper_to_bin]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def reward(self, action=None):
        return 0

    # def visualize(self, vis_settings):
    #     """
    #     In addition to super call, visualize gripper site proportional to the distance to the cube.

    #     Args:
    #         vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
    #             component should be visualized. Should have "grippers" keyword as well as any other relevant
    #             options specified.
    #     """
    #     # Run superclass method first
    #     super().visualize(vis_settings=vis_settings)

    #     # Color the gripper visualization site according to its distance to the cube
    #     if vis_settings["grippers"]:
    #         self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.mug_object)

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        obj_bounds = {}
        obj_bounds["bin"] = dict(
                x=(-0.01, 0.01),
                y=(-0.09, -0.11),
                z_rot=(-np.pi/6, +np.pi/6),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912)),
            )
        return obj_bounds

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def in_bin(self, obj_pos):

        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        bin_rot_mat = T.quat2mat(T.convert_quat(np.array(self.sim.data.body_xquat[self.bin_body_id]), to="xyzw"))
        res = False

        error_in_bin_frame = bin_rot_mat.T @ (obj_pos - bin_pos)

        # print(error_in_bin_frame)

        if (
            abs(error_in_bin_frame[0]) < self.bin_size[0]/2 - 0.01
            and abs(error_in_bin_frame[1]) < self.bin_size[1]/2 - 0.01
            and obj_pos[2] < self.table_offset[2] + self.bin_size[2] - 0.01
        ):
            res = True

        return res


class Put_Can_in_Bin_RL2(Put_in_Bin_RL2):
    def __init__(self, robots, env_configuration="default", controller_configs=None, gripper_types="default", initialization_noise="default", table_full_size=(0.8, 0.8, 0.05), table_friction=(1, 0.005, 0.0001), use_camera_obs=True, use_object_obs=True, reward_scale=1, reward_shaping=False, placement_initializer=None, has_renderer=False, has_offscreen_renderer=True, render_camera="frontview", render_collision_mesh=False, render_visual_mesh=True, render_gpu_device_id=-1, control_freq=20, horizon=1000, ignore_done=False, hard_reset=True, camera_names="agentview", camera_heights=256, camera_widths=256, camera_depths=False, camera_segmentations=None, renderer="mujoco", renderer_config=None, gripper_visualizations=None, bin_size=[0.144, 0.144, 0.143], bin_thickness=0.01):
        super().__init__(robots, env_configuration, controller_configs, gripper_types, initialization_noise, table_full_size, table_friction, use_camera_obs, use_object_obs, reward_scale, reward_shaping, placement_initializer, has_renderer, has_offscreen_renderer, render_camera, render_collision_mesh, render_visual_mesh, render_gpu_device_id, control_freq, horizon, ignore_done, hard_reset, camera_names, camera_heights, camera_widths, camera_depths, camera_segmentations, renderer, renderer_config, gripper_visualizations, bin_size=bin_size, bin_thickness=bin_thickness)

    def load_other_objects(self):
        # self.size_can = np.array([0.048, 0.0255, 0.041])
        # delta = np.array([0.003, 0.003, 0.003])
        self.size_can_max = np.array([0.123, 0.055, 0.111])
        self.size_can_min = np.array([0.103, 0.035, 0.091])

        self.size_can_max = np.array([0.113+0.001, 0.045+0.001, 0.101+0.001])
        self.size_can_min = np.array([0.113-0.001, 0.045-0.001, 0.101-0.001])

        self.size_can = (self.size_can_max + self.size_can_min) / 2 / 2
        delta = (self.size_can_max - self.size_can_min) / 2 / 2

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.can_object = BoxObject(
            name="Can",
            size_min=self.size_can - delta,
            size_max=self.size_can + delta,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.objects.append(self.can_object)

        can_ranges = self.objs_bound["can"]
        x, y, z_rot = can_ranges["x"], can_ranges["y"], can_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"CanSampler",
                mujoco_objects=self.can_object,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001
            ),
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        obj_bounds = super()._get_initial_placement_bounds()
        obj_bounds["can"] = dict(
                x=(-0.02, 0.02),
                y=(+0.1, +0.14),
                z_rot=(-np.pi/6, np.pi/6),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912)),
            )
        return obj_bounds

    def _setup_observables(self):
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def Can_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.can_body_id])

            @sensor(modality=modality)
            def Can_quat(obs_cache):
                return T.convert_quat(np.array(self.sim.data.body_xquat[self.can_body_id]), to="xyzw")


            sensors = [Can_pos, Can_quat]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.can_body_id = self.sim.model.body_name2id(self.can_object.root_body)

    def _check_success(self):
        """
        Returns:
            bool: True if all nuts are placed correctly
        """
        can_pos = self.sim.data.body_xpos[self.can_body_id]
        can_vel = self.sim.data.get_body_xvelp(self.can_object.root_body)
        bin_pos = self.sim.data.body_xpos[self.bin_body_id]
        grasp_mug = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.can_object)
        # bin_not_moved = np.linalg.norm(self.bin_initial_pos - bin_pos) < 0.005
        bin_not_moved = True
        can_still = np.linalg.norm(can_vel) < 0.001
        res = int(bin_not_moved and self.in_bin(can_pos) and not grasp_mug and can_still)

        return res
    
class Put_Can_in_Short_Bin_RL2(Put_Can_in_Bin_RL2):
    def __init__(self, robots, env_configuration="default", controller_configs=None, gripper_types="default", initialization_noise="default", table_full_size=(0.8, 0.8, 0.05), table_friction=(1, 0.005, 0.0001), use_camera_obs=True, use_object_obs=True, reward_scale=1, reward_shaping=False, placement_initializer=None, has_renderer=False, has_offscreen_renderer=True, render_camera="frontview", render_collision_mesh=False, render_visual_mesh=True, render_gpu_device_id=-1, control_freq=20, horizon=1000, ignore_done=False, hard_reset=True, camera_names="agentview", camera_heights=256, camera_widths=256, camera_depths=False, camera_segmentations=None, renderer="mujoco", renderer_config=None, gripper_visualizations=None):
        bin_size = [0.224, 0.154, 0.039]
        bin_thickness = 0.0095
        super().__init__(robots, env_configuration, controller_configs, gripper_types, initialization_noise, table_full_size, table_friction, use_camera_obs, use_object_obs, reward_scale, reward_shaping, placement_initializer, has_renderer, has_offscreen_renderer, render_camera, render_collision_mesh, render_visual_mesh, render_gpu_device_id, control_freq, horizon, ignore_done, hard_reset, camera_names, camera_heights, camera_widths, camera_depths, camera_segmentations, renderer, renderer_config, gripper_visualizations, bin_size=bin_size, bin_thickness=bin_thickness)

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        obj_bounds = {}
        obj_bounds["bin"] = dict(
                x=(-0.04, -0.02),
                y=(-0.09, -0.11),
                z_rot=(-np.pi/24, +np.pi/24),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912)),
            )
        obj_bounds["can"] = dict(
                x=(-0.05, -0.01),
                y=(+0.1, +0.14),
                z_rot=(-np.pi/12, np.pi/12),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912)),
            )
        return obj_bounds

    def in_bin(self, obj_pos):

        bin_pos = np.array(self.sim.data.body_xpos[self.bin_body_id])
        bin_rot_mat = T.quat2mat(T.convert_quat(np.array(self.sim.data.body_xquat[self.bin_body_id]), to="xyzw"))
        res = False

        error_in_bin_frame = bin_rot_mat.T @ (obj_pos - bin_pos)

        # print(error_in_bin_frame)

        if (
            abs(error_in_bin_frame[0]) < self.bin_size[0]/2 - 0.01
            and abs(error_in_bin_frame[1]) < self.bin_size[1]/2 - 0.01
            and obj_pos[2] < self.table_offset[2] + self.bin_thickness + self.size_can[2] + 0.01
        ):
            res = True

        return res

class Put_Can_in_Short_Bin_RL2_range(Put_Can_in_Short_Bin_RL2):
    def __init__(self, **kwargs):
        self.reset_range = copy.deepcopy(kwargs.pop("reset_range", None)) # make a copy to avoid adding ndarray to the original dict
        assert self.reset_range is not None, "reset_range must be specified"
        self.move_cam_name = "agentview"
        self.cam_pose = kwargs.pop("cam_pose", 
            [[ 0.07496072,  0.81140203, -0.57966165,  0.92794331],
            [ 0.996927  , -0.04771872,  0.06212468, -0.04752],
            [ 0.02274738, -0.58253726, -0.81248563,  0.42513748],
            [ 0.0        ,  0.0        ,  0.0       ,  1.0        ]]
        )

        self.cam_pose = np.array(self.cam_pose)
        self.cam_pose[:3, :3] = self.cam_pose[:3, :3]@ np.array([[1, 0, 0],
                                                                [0, -1, 0],
                                                                [0, 0, -1]])
        self.cam_pose[:3, 3] += np.array([-0.56, 0.0, 0.912])
        self.cam_K = kwargs.pop("cam_K",
            np.array([[608.74664307, 0, 314.9819],
                [0, 608.8836059, 249.2951],
                [0, 0, 1]])
        )
        self.cam_K = np.array(self.cam_K)
        
        self.domain_shift = kwargs.pop("domain_shift", None)
        super().__init__(**kwargs)

    def _get_initial_placement_bounds(self):
           # small range
        self.reset_range
        for key in self.reset_range.keys():
            self.reset_range[key]["reference"] = np.array(self.table_offset)
        return self.reset_range

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation
        Args:
            xml_string (str): If specified, creates MjSim object from this filepath
        """
        xml = xml_string if xml_string else self.model.get_xml()

        try:
            xml = modify_xml_for_camera_movement(xml, camera_name=self.move_cam_name)
        except:
            pass

        # process the xml before initializing sim
        if self._xml_processor is not None:
            xml = self._xml_processor(xml)

        # Create the simulation instance
        self.sim = MjSim.from_xml_string(xml)

        try:
            cam_body_id = self.sim.model.body_name2id("cameramover")
        except:
            xml = self.sim.model.get_xml()
            xml = modify_xml_for_camera_movement(xml, camera_name=self.move_cam_name)
            self.sim = MjSim.from_xml_string(xml)
            cam_body_id = self.sim.model.body_name2id("cameramover")

        move_cam_body_id = cam_body_id

        camera_pos = self.cam_pose[:3, 3]
        camera_rot = self.cam_pose[:3, :3]

        self.sim.model.body_pos[move_cam_body_id] = camera_pos
        self.sim.model.body_quat[move_cam_body_id] = T.convert_quat(T.mat2quat(camera_rot), to='wxyz')

        move_cam_id = self.sim.model.camera_name2id(self.move_cam_name)

        intrisics = [self.cam_K[0, 0], self.cam_K[1, 1], self.cam_K[0, 2], self.cam_K[1, 2]]
        self.sim.model.cam_intrinsic[move_cam_id] = np.array(intrisics)

        # sim_state = self.sim.get_state().flatten()
        # self.sim.set_state_from_flattened(sim_state)

        # run a single step to make sure changes have propagated through sim state
        self.sim.forward()

        # Setup sim time based on control frequency
        self.initialize_time(self.control_freq)

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)

        if self.domain_shift == "RGB->GBR":
            # convert RGB to GBR
            obs["agentview_image"] = obs["agentview_image"][:, :, [1, 2, 0]]

        return obs

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
        # z_val = np.random.uniform(-0.01, 0.01, ())
        self.table_offset = [0, 0, 0.912]
        # load model for table top workspace
        if self.domain_shift == "table-wood":
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
                texture="wood"
            )
        else:
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

class Put_Salad_in_Bin_RL2(Put_in_Bin_RL2):
    def __init__(self, robots, env_configuration="default", controller_configs=None, gripper_types="default", initialization_noise="default", table_full_size=(0.8, 0.8, 0.05), table_friction=(1, 0.005, 0.0001), use_camera_obs=True, use_object_obs=True, reward_scale=1, reward_shaping=False, placement_initializer=None, has_renderer=False, has_offscreen_renderer=True, render_camera="frontview", render_collision_mesh=False, render_visual_mesh=True, render_gpu_device_id=-1, control_freq=20, horizon=1000, ignore_done=False, hard_reset=True, camera_names="agentview", camera_heights=256, camera_widths=256, camera_depths=False, camera_segmentations=None, renderer="mujoco", renderer_config=None, gripper_visualizations=None):
        super().__init__(robots, env_configuration, controller_configs, gripper_types, initialization_noise, table_full_size, table_friction, use_camera_obs, use_object_obs, reward_scale, reward_shaping, placement_initializer, has_renderer, has_offscreen_renderer, render_camera, render_collision_mesh, render_visual_mesh, render_gpu_device_id, control_freq, horizon, ignore_done, hard_reset, camera_names, camera_heights, camera_widths, camera_depths, camera_segmentations, renderer, renderer_config, gripper_visualizations)

    def load_other_objects(self):
        shape_scale = 1.22
        mjcf_path = os.path.join(XML_ASSETS_BASE_PATH, "objects/salad_dressing/salad_dressing.xml")
        self.salad_object = MJCFObject(
            name=f"salad",
            mjcf_path=mjcf_path,
            scale=shape_scale,
            # solimp=(0.998, 0.998, 0.001),
            # solref=(0.001, 1),
            density=100,
            # friction=(0.95, 0.3, 0.1),
            friction=(1.0, 1.0, 1e-4),
            # margin=0.001,
        )

        self.objects.append(self.salad_object)

        ranges = self.objs_bound["salad"]
        x, y, z_rot = ranges["x"], ranges["y"], ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"SaladSampler",
                mujoco_objects=self.salad_object,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001
            ),
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        obj_bounds = super()._get_initial_placement_bounds()
        obj_bounds["salad"] = dict(
                x=(-0.02, 0.02),
                y=(+0.1, +0.14),
                # z_rot=(0, 0),
                z_rot=(-np.pi/6, np.pi/6),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912)),
            )
        return obj_bounds

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.salad_body_id = self.sim.model.body_name2id(self.salad_object.root_body)

    def _check_success(self):
        """
        Returns:
            bool: True if all nuts are placed correctly
        """
        salad_pos = self.sim.data.body_xpos[self.salad_body_id]
        salad_vel = self.sim.data.get_body_xvelp(self.salad_object.root_body)
        bin_pos = self.sim.data.body_xpos[self.bin_body_id]
        grasp_mug = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.salad_object)
        bin_not_moved = np.linalg.norm(self.bin_initial_pos - bin_pos) < 0.005
        salad_still = np.linalg.norm(salad_vel) < 0.001
        res = int(bin_not_moved and self.in_bin(salad_pos) and not grasp_mug and salad_still)

        return res