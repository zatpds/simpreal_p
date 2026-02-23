# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from collections import OrderedDict
import numpy as np
import copy

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, string_to_array

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robomimic.assets.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.manipulation.stack import Stack
import robosuite.utils.transform_utils as T
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG

import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import find_elements, find_parent

class Stack_D0(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        return { 
            k : dict(
                x=(-0.08, 0.08),
                y=(-0.08, 0.08),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB"]
        }

class Stack_SRL(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
        z_val = np.random.uniform(-0.01, 0.01, ())
        self.table_offset = [0, 0, 0.912+z_val]
        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # real_quat = [0.6361348, 0.3157979, 0.2793737, 0.6461846]
        # real_quat = [0.64618, 0.3157979, 0.2793737, 0.6361348, ]
        # real_quat = [0.64618, 0.2793737, 0.3157979, 0.6361348,]

        # real_rmat = np.array([[-0.60023955, 0.31215838, -0.73638959],
        #                       [0.79957225, 0.25711919, -0.54274665],
        #                       [0.01991698, -0.91457469, -0.40392629]])
        #
        # real_quat = T.mat2quat(real_rmat)
        # # real_quat = T.convert_quat(real_quat, "wxyz")
        #
        # print("ori quat: ", real_quat)
        # print("quat[wxyz]: ", real_quat)
        # # xyzw: [-0.36965108 -0.7518784   0.4845601   0.2514723 ]
        # # wxyz: [ 0.2514723  -0.36965108 -0.7518784   0.4845601 ]
        #
        # mujoco_arena.set_camera(
        #     camera_name="sideview",
        #     pos=[-0.56 + 0.862, 0.460, 0.912 + 0.314],
        #     quat=real_quat,
        #     # camera_attribs={"fovy": 75}
        # )

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = 0.025 + np.random.uniform(-0.003, 0.003, ())
        size_b = 0.025 + np.random.uniform(-0.003, 0.003, ())

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[size_b]*3,
            size_max=[size_b]*3,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            raise
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        obj_bounds = {
            k : dict(
                x=(-0.1, 0.1),
                y=(-0.1, 0.1),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912+0.01)),
            )
            for k in ["cubeA", "cubeB"]
        }
        return obj_bounds

class Stack_SRL_Fixed(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"
        self.scene_config = kwargs["scene_config"]
        del kwargs["scene_config"]
        Stack.__init__(self, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = 0.025
        size_b = 0.025

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[size_b]*3,
            size_max=[size_b]*3,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        cube_a_x = self.scene_config["cubeA"]["x"]
        cube_a_y = self.scene_config["cubeA"]["y"]
        cube_a_z_rot = self.scene_config["cubeA"]["z"]

        cube_b_x = self.scene_config["cubeB"]["x"]
        cube_b_y = self.scene_config["cubeB"]["y"]
        cube_b_z_rot = self.scene_config["cubeB"]["z"]

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"ObjectSampler1",
                mujoco_objects=self.cubeA,
                x_range=cube_a_x,
                y_range=cube_a_y,
                rotation=cube_a_z_rot,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"ObjectSampler2",
                mujoco_objects=self.cubeB,
                x_range=cube_b_x,
                y_range=cube_b_y,
                rotation=cube_b_z_rot,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )
        self.placement_initializer = placement_initializer


        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        obj_bounds = {
            k : dict(
                x=(-0.15, 0.15),
                y=(-0.15, 0.15),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912+0.01)),
            )
            for k in ["cubeA", "cubeB"]
        }
        return obj_bounds

class Stack_RL2(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()
        # self.conf_id = kwargs["conf_id"]
        # del kwargs["conf_id"]

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
        # z_val = np.random.uniform(-0.01, 0.01, ())
        self.table_offset = [0, 0, 0.912+0.018]
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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = 0.025
        size_b = 0.025

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[size_b]*3,
            size_max=[size_b]*3,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer

        cuba_a_pos = [[0, 0.15],
                      [0.15, 0.15],
                      [0.15, 0],
                      [0.15, -0.15],
                      [0, -0.15],
                      [-0.15, -0.15],
                      [-0.15, 0],
                      [-0.15, 0.15]]
        cube_a_x, cube_a_y = cuba_a_pos[7]

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"ObjectSampler1",
                mujoco_objects=self.cubeA,
                x_range=[cube_a_x, cube_a_x],
                y_range=[cube_a_y, cube_a_y],
                rotation=(-0.1 * np.pi, 0.1 * np.pi),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"ObjectSampler2",
                mujoco_objects=self.cubeB,
                x_range=[0, 0],
                y_range=[0, 0],
                rotation=(-0.1 * np.pi, 0.1 * np.pi),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )
        self.placement_initializer = placement_initializer


        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        obj_bounds = {
            k : dict(
                x=(-0.1, 0.1),
                y=(-0.1, 0.1),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912+0.018)),
            )
            for k in ["cubeA", "cubeB"]
        }
        return obj_bounds

class Stack_RL2_random(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()
        # self.conf_id = kwargs["conf_id"]
        # del kwargs["conf_id"]

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = 0.025
        size_b = 0.025

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[size_b]*3,
            size_max=[size_b]*3,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            raise
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        obj_bounds = {
            k : dict(
                x=(-0.1, 0.1),
                y=(-0.1, 0.1),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912+0.01)),
            )
            for k in ["cubeA", "cubeB"]
        }
        return obj_bounds

class Stack_RL2_random_large_cube(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()
        # self.conf_id = kwargs["conf_id"]
        # del kwargs["conf_id"]

        # ensure cube symmetry
        # assert len(bounds) == 2
        # for k in ["x", "y", "z_rot", "reference"]:
        #     assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = 0.025
        size_b = [0.025, 0.05, 0.025]

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b,
            size_max=size_b,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            raise
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        # obj_bounds = {
        #     k : dict(
        #         x=(-0.1, 0.1),
        #         y=(-0.1, 0.1),
        #         z_rot=(-0.1 * np.pi, 0.1 * np.pi),
        #         # NOTE: hardcoded @self.table_offset since this might be called in init function
        #         reference=np.array((0, 0, 0.912+0.01)),
        #     )
        #     for k in ["cubeA", "cubeB"]
        # }
        obj_bounds = {
            "cubeA": dict(
                x=(-0.1, 0.1),
                y=(-0.2, -0.05),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            ),
            "cubeB": dict(
                x=(-0.1, 0.1),
                y=(0.05, 0.2),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            )
        }
        return obj_bounds


class Stack_RL2_random_large_cube_2_sides(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        # bounds = self._get_initial_placement_bounds()

        # self.conf_id = kwargs["conf_id"]
        # del kwargs["conf_id"]

        # ensure cube symmetry
        # assert len(bounds) == 2
        # for k in ["x", "y", "z_rot", "reference"]:
        #     assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        # placement_initializer = UniformRandomSampler(
        #     name="ObjectSampler",
        #     x_range=bounds["cubeA"]["x"],
        #     y_range=bounds["cubeA"]["y"],
        #     rotation=bounds["cubeA"]["z_rot"],
        #     rotation_axis='z',
        #     ensure_object_boundary_in_range=False,
        #     ensure_valid_placement=True,
        #     reference_pos=bounds["cubeA"]["reference"],
        #     z_offset=0.01,
        # )



        Stack.__init__(self, placement_initializer=None, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = 0.025
        size_b = [0.025, 0.05, 0.025]

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b,
            size_max=size_b,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        cube_names = ["cubeA", "cubeB"]
        # Create placement initializer

        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for ii, (cube_name, cube) in enumerate(zip(cube_names, cubes)):
            bound_i = bounds[cube_name]
            x = bound_i["x"]
            y = bound_i["y"]
            z = bound_i["z_rot"]
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"ObjectSampler-obj{ii}",
                    mujoco_objects=cube,
                    x_range=[x[0], x[1]],
                    y_range=[y[0], y[1]],
                    rotation=(z[0], z[1]),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                ),
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        # obj_bounds = {
        #     k : dict(
        #         x=(-0.1, 0.1),
        #         y=(-0.1, 0.1),
        #         z_rot=(-0.1 * np.pi, 0.1 * np.pi),
        #         # NOTE: hardcoded @self.table_offset since this might be called in init function
        #         reference=np.array((0, 0, 0.912+0.01)),
        #     )
        #     for k in ["cubeA", "cubeB"]
        # }
        obj_bounds = {
            "cubeA": dict(
                x=(-0.1, 0.1),
                y=(-0.15, -0.05),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            ),
            "cubeB": dict(
                x=(-0.1, 0.1),
                y=(0.05, 0.15),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            )
        }
        return obj_bounds


class Stack_RL2_random_cube_2_sides(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"
        if "gripper_visualizations" in kwargs.keys():
            del kwargs["gripper_visualizations"]
        Stack.__init__(self, placement_initializer=None, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        cubeA_stable = np.linalg.norm(self.sim.data.get_body_xvelp("cubeA"))<0.001
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB and cubeA_stable

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = np.array([0.025, 0.025, 0.025])
        size_b = np.array([0.025, 0.05, 0.025])
        delta = np.array([0.001, 0.001, 0.001])

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=size_a - delta,
            size_max=size_a + delta,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b - delta,
            size_max=size_b + delta,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        cube_names = ["cubeA", "cubeB"]
        # Create placement initializer

        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for ii, (cube_name, cube) in enumerate(zip(cube_names, cubes)):
            bound_i = bounds[cube_name]
            x = bound_i["x"]
            y = bound_i["y"]
            z = bound_i["z_rot"]
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"ObjectSampler-obj{ii}",
                    mujoco_objects=cube,
                    x_range=[x[0], x[1]],
                    y_range=[y[0], y[1]],
                    rotation=(z[0], z[1]),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                ),
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        # obj_bounds = {
        #     k : dict(
        #         x=(-0.1, 0.1),
        #         y=(-0.1, 0.1),
        #         z_rot=(-0.1 * np.pi, 0.1 * np.pi),
        #         # NOTE: hardcoded @self.table_offset since this might be called in init function
        #         reference=np.array((0, 0, 0.912+0.01)),
        #     )
        #     for k in ["cubeA", "cubeB"]
        # }
        obj_bounds = {
            "cubeA": dict(
                x=(-0.1, 0.1),
                y=(-0.15, -0.05),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            ),
            "cubeB": dict(
                x=(-0.1, 0.1),
                y=(0.05, 0.15),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            )
        }
        return obj_bounds

class Stack_RL2_random_cube_multi_modal(Stack_RL2_random_cube_2_sides):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_initial_placement_bounds(self):
        mode_1 = dict(
            cubeA=dict(
                    x=(-0.1, 0.1),
                    y=(-0.15, -0.05),
                    z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                    # NOTE: hardcoded @self.table_offset since this might be called in init function
                    reference=np.array((0, 0, 0.912 + 0.01)),
                ),
            cubeB=dict(
                    x=(-0.1, 0.1),
                    y=(0.05, 0.15),
                    z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                    # NOTE: hardcoded @self.table_offset since this might be called in init function
                    reference=np.array((0, 0, 0.912 + 0.01)),
                )
        )

        mode_2 = dict(
            cubeA=dict(
                    x=(-0.1, 0.1),
                    y=(0.05, 0.15),
                    z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                    # NOTE: hardcoded @self.table_offset since this might be called in init function
                    reference=np.array((0, 0, 0.912 + 0.01)),
                ),
            cubeB=dict(
                    x=(-0.1, 0.1),
                    y=(-0.15, -0.05),
                    z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                    # NOTE: hardcoded @self.table_offset since this might be called in init function
                    reference=np.array((0, 0, 0.912 + 0.01)),
                )
        )

        return [mode_1, mode_2]
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super(Stack, self)._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            mode = np.random.choice(range(len(self.placement_initializers)))
            object_placements = self.placement_initializers[mode].sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = np.array([0.025, 0.025, 0.025])
        size_b = np.array([0.025, 0.05, 0.025])
        delta = np.array([0.005, 0.005, 0.005])

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=size_a - delta,
            size_max=size_a + delta,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b - delta,
            size_max=size_b + delta,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        cube_names = ["cubeA", "cubeB"]
        # Create placement initializer

        bounds_list = self._get_initial_placement_bounds()
        self.placement_initializers = []

        for bound in bounds_list:
            placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for ii, (cube_name, cube) in enumerate(zip(cube_names, cubes)):
                bound_i = bound[cube_name]
                x = bound_i["x"]
                y = bound_i["y"]
                z = bound_i["z_rot"]
                placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"ObjectSampler-obj{ii}",
                        mujoco_objects=cube,
                        x_range=[x[0], x[1]],
                        y_range=[y[0], y[1]],
                        rotation=(z[0], z[1]),
                        rotation_axis='z',
                        ensure_object_boundary_in_range=True,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.01,
                    ),
                )
            self.placement_initializers.append(placement_initializer)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

def modify_xml_for_camera_movement(xml, camera_name):
    """
    Cameras in mujoco are 'fixed', so they can't be moved by default.
    Although it's possible to hack position movement, rotation movement
    does not work. An alternative is to attach a camera to a body element,
    and move the body.

    This function modifies the camera with name @camera_name in the xml
    by attaching it to a body element that we can then manipulate. In this
    way, we can move the camera by moving the body.

    See http://www.mujoco.org/forum/index.php?threads/move-camera.2201/ for
    further details.

    xml (str): Mujoco sim XML file as a string
    camera_name (str): Name of camera to tune
    """
    tree = ET.fromstring(xml)

    # find the correct camera
    camera_elem = None
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        if camera.get("name") == camera_name:
            camera_elem = camera
            break
    assert camera_elem is not None, "No valid camera name found, options are: {}"\
        .format([camera.get("name") for camera in cameras])

    # Find parent element of the camera element
    parent = find_parent(root=tree, child=camera_elem)
    assert parent is not None

    # add camera body
    cam_body = ET.SubElement(parent, "body")
    cam_body.set("name", "cameramover")
    cam_body.set("pos", camera_elem.get("pos"))
    cam_body.set("quat", camera_elem.get("quat"))
    new_camera = ET.SubElement(cam_body, "camera")
    new_camera.set("mode", "fixed")
    new_camera.set("name", camera_elem.get("name"))
    new_camera.set("pos", "0 0 0")
    # Also need to define inertia
    inertial = ET.SubElement(cam_body, "inertial")
    inertial.set("diaginertia", "1e-08 1e-08 1e-08")
    inertial.set("mass", "1e-08")
    inertial.set("pos", "0 0 0")

    # remove old camera element
    parent.remove(camera_elem)
    # camera_elem.set("name", "fixed_agentview")

    return ET.tostring(tree, encoding="utf8").decode("utf8")

class Stack_RL2_range(Stack_RL2_random_cube_2_sides):
    def __init__(self, **kwargs):
        self.reset_range = copy.deepcopy(kwargs.pop("reset_range", None)) # make a copy to avoid adding ndarray to the original dict
        assert self.reset_range is not None, "reset_range must be specified"

        self.initial_joint_pos = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

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

    def _reset_internal(self):
        super()._reset_internal()
        # simulate for a few steps to let the cubes settle
        for _ in range(100):
            self.sim.step()
            self.sim.forward()

        self.robot_joints = self.robots[0].robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.initial_joint_pos
        
    def _get_initial_placement_bounds(self):
           # small range
        for key in self.reset_range.keys():
            self.reset_range[key]["reference"] = np.array(self.table_offset)
        return self.reset_range

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

class Stack_RL2_random_cube_random_layout(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        Stack.__init__(self, placement_initializer=None, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = np.array([0.025, 0.025, 0.025])
        size_b = np.array([0.025, 0.05, 0.025])
        delta = np.array([0.003, 0.003, 0.003])

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=size_a - delta,
            size_max=size_a + delta,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b - delta,
            size_max=size_b + delta,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        cube_names = ["cubeA", "cubeB"]
        # Create placement initializer

        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for ii, (cube_name, cube) in enumerate(zip(cube_names, cubes)):
            bound_i = bounds[cube_name]
            x = bound_i["x"]
            y = bound_i["y"]
            z = bound_i["z_rot"]
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"ObjectSampler-obj{ii}",
                    mujoco_objects=cube,
                    x_range=[x[0], x[1]],
                    y_range=[y[0], y[1]],
                    rotation=(z[0], z[1]),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                ),
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        # obj_bounds = {
        #     k : dict(
        #         x=(-0.1, 0.1),
        #         y=(-0.1, 0.1),
        #         z_rot=(-0.1 * np.pi, 0.1 * np.pi),
        #         # NOTE: hardcoded @self.table_offset since this might be called in init function
        #         reference=np.array((0, 0, 0.912+0.01)),
        #     )
        #     for k in ["cubeA", "cubeB"]
        # }
        obj_bounds = {
            "cubeA": dict(
                x=(-0.2, 0.1),
                y=(-0.15, 0.15),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            ),
            "cubeB": dict(
                x=(-0.2, 0.1),
                y=(-0.15, 0.15),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            )
        }
        return obj_bounds


class Stack_RL2_random_large_cube_2_sides_fixed(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"
        self.scene_config = kwargs["scene_config"]
        del kwargs["scene_config"]
        Stack.__init__(self, placement_initializer=None, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        size_a = 0.025
        size_b = [0.025, 0.05, 0.025]

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[size_a]*3,
            size_max=[size_a]*3,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b,
            size_max=size_b,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        cube_names = ["cubeA", "cubeB"]
        # Create placement initializer

        cube_a_params = self.scene_config["cubeA"]
        cube_a_x = cube_a_params["x"]
        cube_a_y = cube_a_params["y"]
        cube_a_z_rot = cube_a_params["z"]

        cube_b_params = self.scene_config["cubeB"]
        cube_b_x = cube_b_params["x"]
        cube_b_y = cube_b_params["y"]
        cube_b_z_rot = cube_b_params["z"]

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"ObjectSampler1",
                mujoco_objects=self.cubeA,
                x_range=cube_a_x,
                y_range=cube_a_y,
                rotation=cube_a_z_rot,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"ObjectSampler2",
                mujoco_objects=self.cubeB,
                x_range=cube_b_x,
                y_range=cube_b_y,
                rotation=cube_b_z_rot,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        )

        self.placement_initializer = placement_initializer

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        # obj_bounds = {
        #     k : dict(
        #         x=(-0.1, 0.1),
        #         y=(-0.1, 0.1),
        #         z_rot=(-0.1 * np.pi, 0.1 * np.pi),
        #         # NOTE: hardcoded @self.table_offset since this might be called in init function
        #         reference=np.array((0, 0, 0.912+0.01)),
        #     )
        #     for k in ["cubeA", "cubeB"]
        # }
        obj_bounds = {
            "cubeA": dict(
                x=(-0.1, 0.1),
                y=(-0.15, -0.05),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            ),
            "cubeB": dict(
                x=(-0.1, 0.1),
                y=(0.05, 0.15),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            )
        }
        return obj_bounds


class Stack_RL2_random_cube_random_layout_only_A(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        Stack.__init__(self, placement_initializer=None, **kwargs)


    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        # self.cubeA = BoxObject(
        #     name="cubeA",
        #     size_min=[0.025, 0.025, 0.025],
        #     size_max=[0.025, 0.025, 0.025],
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        # )
        size_a = np.array([0.025, 0.025, 0.025])
        size_b = np.array([0.025, 0.05, 0.025])
        delta = np.array([0.003, 0.003, 0.003])

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=size_a - delta,
            size_max=size_a + delta,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=size_b - delta,
            size_max=size_b + delta,
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        cubes = [self.cubeA, self.cubeB]
        cube_names = ["cubeA", "cubeB"]
        # Create placement initializer

        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for ii, (cube_name, cube) in enumerate(zip(cube_names, cubes)):
            bound_i = bounds[cube_name]
            x = bound_i["x"]
            y = bound_i["y"]
            z = bound_i["z_rot"]
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"ObjectSampler-obj{ii}",
                    mujoco_objects=cube,
                    x_range=[x[0], x[1]],
                    y_range=[y[0], y[1]],
                    rotation=(z[0], z[1]),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.01,
                ),
            )


        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
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
        # obj_bounds = {
        #     k : dict(
        #         x=(-0.1, 0.1),
        #         y=(-0.1, 0.1),
        #         z_rot=(-0.1 * np.pi, 0.1 * np.pi),
        #         # NOTE: hardcoded @self.table_offset since this might be called in init function
        #         reference=np.array((0, 0, 0.912+0.01)),
        #     )
        #     for k in ["cubeA", "cubeB"]
        # }
        obj_bounds = {
            "cubeA": dict(
                x=(-0.2, 0.1),
                y=(-0.15, 0.),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            ),
            "cubeB": dict(
                x=(-0.2, 0.1),
                y=(0.05, 0.15),
                z_rot=(-0.1 * np.pi, 0.1 * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.912 + 0.01)),
            )
        }
        return obj_bounds




class Stack_D1(Stack_D0):
    """
    Much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB"]
        }


class StackThree(Stack_D0):
    """
    Stack three cubes instead of two.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 3
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))
            assert np.array_equal(np.array(bounds["cubeB"][k]), np.array(bounds["cubeC"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def reward(self, action=None):
        """
        We only return sparse rewards here.
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _check_cubeC_lifted(self):
        # cube C needs to be higher than A
        return self._check_lifted(self.cubeC_body_id, margin=0.08)

    def _check_cubeC_stacked(self):
        grasping_cubeC = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        cubeC_lifted = self._check_cubeC_lifted()
        cubeC_touching_cubeA = self.check_contact(self.cubeC, self.cubeA)
        return (not grasping_cubeC) and cubeC_lifted and cubeC_touching_cubeA

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.
        Returns:
            3-tuple:
                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Stacking successful when A is on top of B and C is on top of A.
        # This means both A and C are lifted, not grasped by robot, and we have contact
        # between (A, B) and (A, C).

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_reach = 0.
        r_lift = 0.
        r_stack = 0.
        if self._check_cubeA_stacked() and self._check_cubeC_stacked():
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

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
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube C
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

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
            def cubeC_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeC_body_id])

            @sensor(modality=modality)
            def cubeC_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeC_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache[f"{pf}eef_pos"] if \
                    "cubeC_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeA_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache["cubeA_pos"] if \
                    "cubeA_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeB_to_cubeC(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache["cubeC_pos"] if \
                    "cubeB_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            sensors = [cubeC_pos, cubeC_quat, gripper_to_cubeC, cubeA_to_cubeC, cubeB_to_cubeC]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

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
        return { 
            k : dict(
                x=(-0.10, 0.10),
                y=(-0.10, 0.10),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }


class StackThree_D0(StackThree):
    """Rename base class for convenience."""
    pass


class StackThree_D1(StackThree_D0):
    """
    Less z-rotation (for easier datagen) and much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }
