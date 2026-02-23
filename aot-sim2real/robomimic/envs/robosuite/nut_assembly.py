# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
import os
import copy
from six import with_metaclass

import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.environments.manipulation.nut_assembly import NutAssembly, NutAssemblySquare
from robosuite.models.arenas import PegsArena
from robosuite.models.objects import SquareNutObject, RoundNutObject
from robosuite.models.tasks import ManipulationTask
from robomimic.assets.arenas import TableArena
import robosuite.utils
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string, string_to_array, find_elements
from robosuite.utils import RandomizationError
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite_model_zoo.utils.mjcf_obj import MJCFObject
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, string_to_array
from robomimic.assets.objects import PegWithBaseObject, PegWithBaseTallTiltObject, SquareNutThickObject
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from robosuite.utils.mjmod import DynamicsModder
import mimicgen
import robosuite.utils.transform_utils

from robomimic.envs.robosuite.stack import modify_xml_for_camera_movement
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
import robosuite.utils.transform_utils as T

XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")

class NutAssembly_D0(NutAssembly, SingleArmEnv_MG):
    """
    Augment robosuite nut assembly task for mimicgen.
    """
    def __init__(self, **kwargs):
        NutAssembly.__init__(self, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

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
        return dict(
            square_nut=dict(
                x=(-0.115, -0.11),
                y=(0.11, 0.225),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            round_nut=dict(
                x=(-0.115, -0.11),
                y=(-0.225, -0.11),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )


class Square_D0(NutAssemblySquare, SingleArmEnv_MG):
    """
    Augment robosuite nut assembly square task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        # make placement initializer here
        nut_names = ("SquareNut", "RoundNut")

        # note: makes round nut init somewhere far off the table
        round_nut_far_init = (-1.1, -1.0)

        bounds = self._get_initial_placement_bounds()
        nut_x_ranges = (bounds["nut"]["x"], bounds["nut"]["x"])
        nut_y_ranges = (bounds["nut"]["y"], round_nut_far_init)
        nut_z_ranges = (bounds["nut"]["z_rot"], bounds["nut"]["z_rot"])
        nut_references = (bounds["nut"]["reference"], bounds["nut"]["reference"])

        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for nut_name, x_range, y_range, z_range, ref in zip(nut_names, nut_x_ranges, nut_y_ranges, nut_z_ranges, nut_references):
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{nut_name}Sampler",
                    x_range=x_range,
                    y_range=y_range,
                    rotation=z_range,
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=ref,
                    z_offset=0.02,
                )
            )

        NutAssemblySquare.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

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
        return dict(
            nut=dict(
                x=(-0.115, -0.11),
                y=(0.11, 0.225),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )


class Square_D1(Square_D0):
    """
    Specifies a different placement initializer for the pegs where it is initialized
    with a broader x-range and broader y-range.
    """
    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0.115, 0.115),
                y=(-0.255, 0.255),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg=dict(
                x=(-0.1, 0.3),
                y=(-0.2, 0.2),
                z_rot=(0., 0.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation 
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # define nuts
        self.nuts = []
        nut_names = ("SquareNut", "RoundNut")

        # super class should already give us placement initializer in init
        assert self.placement_initializer is not None

        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (nut_cls, nut_name) in enumerate(zip(
                (SquareNutObject, RoundNutObject),
                nut_names,
        )):
            nut = nut_cls(name=nut_name)
            self.nuts.append(nut)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{nut_name}Sampler", mujoco_objects=nut)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(nut)

        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg_bounds = self._get_initial_placement_bounds()["peg"]

        sample_x = np.random.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
        sample_y = np.random.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
        sample_z_rot = np.random.uniform(low=peg_bounds["z_rot"][0], high=peg_bounds["z_rot"][1])
        peg1_xml_pos[0] = peg_bounds["reference"][0] + sample_x
        peg1_xml_pos[1] = peg_bounds["reference"][1] + sample_y
        peg1_xml_quat = np.array([np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)])

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg1_xml.set("quat", array_to_string(peg1_xml_quat))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.nuts,
        )

    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            peg1_id = self.sim.model.body_name2id("peg1")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[peg1_id])

            name = "peg1_pos"
            observables[name] = Observable(
                name=name,
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )

        return observables


class Square_D1_RL2(Square_D0):
    """
    Specifies a different placement initializer for the pegs where it is initialized
    with a broader x-range and broader y-range.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        # make placement initializer here
        NutAssemblySquare.__init__(self, **kwargs)
        self.table_offset = [0, 0, 0.912]

    def _get_initial_placement_bounds(self):
        return dict(
            nut=dict(
                x=(-0., 0.1),
                y=(-0.15, -0.1),
                z_rot=(-np.pi/12+np.pi, np.pi/12+np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
            peg=dict(
                x=(-0., 0.1),
                y=(0.1, 0.15),
                z_rot=(-np.pi/12, np.pi/12),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
        )

    def on_peg(self, obj_pos, peg_id):

        if peg_id == 0:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg1_body_id])
        else:
            peg_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.table_offset[2] + 0.02
        ):
            res = True

        return res

    def _check_success(self):
        """
        Check if all nuts have been successfully placed around their corresponding pegs.

        Returns:
            bool: True if all nuts are placed correctly
        """
        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        for i, nut in enumerate(self.nuts):
            obj_str = nut.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            grasp_nut = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=nut)
            self.objects_on_pegs[i] = int(self.on_peg(obj_pos, i) and not grasp_nut)

        if self.single_object_mode > 0:
            return np.sum(self.objects_on_pegs) > 0  # need one object on peg

        # returns True if all objects are on correct pegs
        return np.sum(self.objects_on_pegs) == len(self.nuts)


    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            success = False
            for _ in range(5000): # 5000 retries

                # Sample from the placement initializer for all objects
                object_placements = self.placement_initializer.sample()

                # ADDED: check collision with pegs and maybe re-sample
                location_valid = True
                for obj_pos, obj_quat, obj in object_placements.values():
                    horizontal_radius = obj.horizontal_radius

                    peg1_id = self.sim.model.body_name2id("peg1")
                    peg1_pos = np.array(self.sim.data.body_xpos[peg1_id])
                    peg1_horizontal_radius = self.peg1_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg1_pos[0], obj_pos[1] - peg1_pos[1]))
                        <= peg1_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                    peg2_id = self.sim.model.body_name2id("peg2")
                    peg2_pos = np.array(self.sim.data.body_xpos[peg2_id])
                    peg2_horizontal_radius = self.peg2_horizontal_radius
                    if (
                        np.linalg.norm((obj_pos[0] - peg2_pos[0], obj_pos[1] - peg2_pos[1]))
                        <= peg2_horizontal_radius + horizontal_radius
                    ):
                        location_valid = False
                        break

                if location_valid:
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
        self.table_offset = [0, 0, 0.912]
        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation
        SingleArmEnv._load_model(self)
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()
        bounds = self._get_initial_placement_bounds()

        nut_ranges = bounds["nut"]
        x, y, z_rot = nut_ranges["x"], nut_ranges["y"], nut_ranges["z_rot"]

        nut = SquareNutObject(name="SquareNut")

        # define nuts
        self.nuts = [nut]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"SquareNutSampler",
                mujoco_objects=nut,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ),
        )

        # Reset sampler before adding any new samplers / objects
        # self.placement_initializer.reset()
        #
        # if isinstance(self.placement_initializer, SequentialCompositeSampler):
        #     # assumes we have two samplers so we add nuts to them
        #     self.placement_initializer.add_objects_to_sampler(sampler_name=f"SquareNutSampler", mujoco_objects=nut)
        # else:
        #     # This is assumed to be a flat sampler, so we just add all nuts to this sampler
        #     self.placement_initializer.add_objects(nut)

        peg_ranges = bounds["peg"]

        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))

        sample_x = np.random.uniform(low=peg_ranges["x"][0], high=peg_ranges["x"][1])
        sample_y = np.random.uniform(low=peg_ranges["y"][0], high=peg_ranges["y"][1])
        sample_z_rot = np.random.uniform(low=peg_ranges["z_rot"][0], high=peg_ranges["z_rot"][1])
        peg1_xml_pos[0] = self.table_offset[0] + sample_x
        peg1_xml_pos[1] = self.table_offset[1] + sample_y
        peg1_xml_pos[2] = self.table_offset[2] - 0.04

        peg1_xml_quat = np.array([np.cos(sample_z_rot / 2), 0, 0, np.sin(sample_z_rot / 2)])

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg1_xml.set("quat", array_to_string(peg1_xml_quat))


        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )

    def _setup_observables(self):
        """
        Add in peg-related observables, since the peg moves now.
        For now, just try adding peg position.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"
            peg1_id = self.sim.model.body_name2id("peg1")

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[peg1_id])

            name = "peg1_pos"
            observables[name] = Observable(
                name=name,
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )

        return observables

class Square_Peg_RL2(Square_D0):
    """
    Specifies a different placement initializer for the pegs where it is initialized
    with a broader x-range and broader y-range.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        NutAssemblySquare.__init__(self, **kwargs)
        self.table_offset = [0, 0, 0.912]


    def _get_initial_placement_bounds(self):
        # large range
        return dict(
            nut=dict(
                x=(-0.05, 0.05),
                y=(-0.02, -0.15),
                # z_rot=(-np.pi/12 - 0.65*np.pi, np.pi/12 - 0.65*np.pi),
                z_rot=(-np.pi/12 + np.pi, np.pi/12 + np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
            peg=dict(
                x=(-0.02, 0.05),
                y=(0.1, 0.15),
                z_rot=(-np.pi/12, np.pi/12),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
        )
    
    def _get_observations(self, force_update=False):
        observations = super()._get_observations(force_update)
        obj_id = self.sim.model.body_name2id("Peg_main")
        obj_pos = np.array(self.sim.data.body_xpos[obj_id])
        obj_quat = np.array(self.sim.data.body_xquat[obj_id])
        observations["Peg_pos"] = obj_pos
        observations["Peg_quat"] = obj_quat
        return observations

    def on_peg(self, obj_pos, peg_pos):
        res = False
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.table_offset[2] + 0.04
        ):
            res = True
        return res

    def _check_success(self):
        """
        Check if all nuts have been successfully placed around their corresponding pegs.

        Returns:
            bool: True if all nuts are placed correctly
        """
        # remember objects that are on the correct pegs
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        nut = self.nuts[0]
        peg = self.objects[1]
        nut_str = nut.name
        peg_str = peg.name

        nut_pos = self.sim.data.body_xpos[self.obj_body_id[nut_str]]
        dist = np.linalg.norm(gripper_site_pos - nut_pos)
        r_reach = 1 - np.tanh(10.0 * dist)
        grasp_nut = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=nut)

        peg_id = self.sim.model.body_name2id(peg.root_body)
        peg_pos = self.sim.data.body_xpos[peg_id]

        # peg_not_moved = np.linalg.norm(self.peg_initial_pos - peg_pos) < 0.03
        peg_not_moved = True

        self.objects_on_pegs[0] = int((self.on_peg(obj_pos=nut_pos, peg_pos=peg_pos) and not grasp_nut) and peg_not_moved)

        if self.single_object_mode > 0:
            return np.sum(self.objects_on_pegs) > 0  # need one object on peg

        # returns True if all objects are on correct pegs and peg is not moved
        return (np.sum(self.objects_on_pegs) == len(self.nuts)) and peg_not_moved

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                # record peg initial pose
                if obj == self.objects[1]:
                    self.peg_initial_pos = obj_pos

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

        # simulate for a few steps to let the peg settle
        for _ in range(100):
            self.sim.step()
            self.sim.forward()

        self.robot_joints = self.robots[0].robot_model.joints
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.initial_joint_pos

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """
        self.table_offset = [0, 0, 0.912]
        # load model for table top workspace
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        return mujoco_arena

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation
        SingleArmEnv._load_model(self)
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()
        bounds = self._get_initial_placement_bounds()


        nut = SquareNutObject(name="SquareNut")
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

        peg_min = np.array([0.029, 0.029, 0.059])/2
        peg_max = np.array([0.031, 0.031, 0.061])/2

        # peg = BoxObject(
        #     name="Peg",
        #     size_min=peg_min,
        #     size_max=peg_max,
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        #     density=100000
        # )

        peg = PegWithBaseObject(name="Peg")

        # define nuts
        self.nuts = [nut]
        self.objects = [nut, peg]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        nut_ranges = bounds["nut"]
        x, y, z_rot = nut_ranges["x"], nut_ranges["y"], nut_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"SquareNutSampler",
                mujoco_objects=nut,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            ),
        )

        peg_ranges = bounds["peg"]
        x, y, z_rot = peg_ranges["x"], peg_ranges["y"], peg_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"PegSampler",
                mujoco_objects=peg,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            ),
        )


        # # get xml element corresponding to both pegs
        # peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        # peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # # apply randomization
        # peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        # peg1_xml_pos[0] = -12
        # peg1_xml_pos[1] = 10

        # # move peg2 completely out of scene
        # peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        # peg2_xml_pos[0] = -10.
        # peg2_xml_pos[1] = 0.

        # # set modified entry in xml
        # peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        # peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # # get collision checking entries
        # peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        # peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        # self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        # self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """

        SingleArmEnv._setup_references(self)

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        for nut in self.nuts:
            self.obj_body_id[nut.name] = self.sim.model.body_name2id(nut.root_body)
            self.obj_geom_id[nut.name] = [self.sim.model.geom_name2id(g) for g in nut.contact_geoms]

        # information of objects
        self.object_site_ids = [self.sim.model.site_name2id(nut.important_sites["handle"]) for nut in self.nuts]

        # keep track of which objects are on their corresponding pegs
        self.objects_on_pegs = np.zeros(len(self.nuts))

class Square_Peg_RL2_small_reset_range(Square_Peg_RL2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_initial_placement_bounds(self):
           # small range
        return dict(
            nut=dict(
                x=(0.01, 0.05),
                y=(-0.1, -0.05),
                z_rot=(-np.pi/18+np.pi, np.pi/18+np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
            peg=dict(
                x=(0.01, 0.05),
                y=(0.1, 0.15),
                z_rot=(-np.pi/18, np.pi/18),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
        )

    # def _get_initial_placement_bounds(self):
    #     return dict(
    #         nut=dict(
    #             x=(0.0, 0.0),
    #             y=(-0.1, -0.1),
    #             z_rot=(np.pi, np.pi),
    #             # NOTE: hardcoded @self.table_offset since this might be called in init function
    #             reference=np.array(self.table_offset),
    #         ),
    #         peg=dict(
    #             x=(0.0, 0.0),
    #             y=(0.1, 0.1),
    #             z_rot=(0, 0),
    #             # NOTE: hardcoded @self.table_offset since this might be called in init function
    #             reference=np.array(self.table_offset),
    #         ),
    #     )

class Square_Peg_RL2_range(Square_Peg_RL2):
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

    def _get_initial_placement_bounds(self):
        self.reset_range
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


class Square_Peg_RL2_Multi_Modal(Square_Peg_RL2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_initial_placement_bounds(self):
        mode_1 = dict(
            nut=dict(
                x=(-0.05, 0.05),
                y=(-0.02, -0.15),
                z_rot=(-np.pi/12 + np.pi, np.pi/12 + np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
            peg=dict(
                x=(-0.02, 0.05),
                y=(0.1, 0.15),
                z_rot=(-np.pi/12, np.pi/12),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
        )

        mode_2 = dict(
            nut=dict(
                x=(-0.05, 0.05),
                y=(0.02, 0.15),
                z_rot=(-np.pi/12 + np.pi, np.pi/12 + np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
            peg=dict(
                x=(-0.02, 0.05),
                y=(-0.15, -0.1),
                z_rot=(-np.pi/12, np.pi/12),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array(self.table_offset),
            ),
        )

        return [mode_1, mode_2]

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            mode = np.random.choice(range(len(self.placement_initializers)))
            object_placements = self.placement_initializers[mode].sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                # record peg initial pose
                if obj == self.objects[1]:
                    self.peg_initial_pos = obj_pos

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation
        SingleArmEnv._load_model(self)
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()
        bounds_list = self._get_initial_placement_bounds()

        nut = SquareNutObject(name="SquareNut")
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

        peg_min = np.array([0.029, 0.029, 0.059])/2
        peg_max = np.array([0.031, 0.031, 0.061])/2

        peg = PegWithBaseObject(name="Peg")

        # define nuts
        self.nuts = [nut]
        self.objects = [nut, peg]

        self.placement_initializers = []

        for bounds in bounds_list:
            placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

            nut_ranges = bounds["nut"]
            x, y, z_rot = nut_ranges["x"], nut_ranges["y"], nut_ranges["z_rot"]

            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"SquareNutSampler",
                    mujoco_objects=nut,
                    x_range=[x[0], x[1]],
                    y_range=[y[0], y[1]],
                    rotation=(z_rot[0], z_rot[1]),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.001,
                ),
            )
            self.placement_initializers.append(placement_initializer)

            peg_ranges = bounds["peg"]
            x, y, z_rot = peg_ranges["x"], peg_ranges["y"], peg_ranges["z_rot"]

            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"PegSampler",
                    mujoco_objects=peg,
                    x_range=[x[0], x[1]],
                    y_range=[y[0], y[1]],
                    rotation=(z_rot[0], z_rot[1]),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.001,
                ),
            )


        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg1_xml_pos[0] = -12
        peg1_xml_pos[1] = 10

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )    

class Square_Peg_RL2_Dynamics_Randomization(Square_Peg_RL2):
    def __init__(self, **kwargs):
        kwargs["hard_reset"] = False  # need this to avoid "AttributeError: 'MjSim' object has no attribute 'model'"
        super().__init__(**kwargs)
        self.dynamics_randomization_args = {
            # Opt parameters
            "randomize_density": False,
            "randomize_viscosity": False,
            # Body parameters
            "body_names": None,  # all bodies randomized
            "randomize_position": False,
            "randomize_quaternion": False,
            "randomize_inertia": True,
            "randomize_mass": True,
            "position_perturbation_size": 0.0015,
            "quaternion_perturbation_size": 0.003,
            "inertia_perturbation_ratio": 0.1,
            "mass_perturbation_ratio": 0.1,
            # Geom parameters
            "geom_names": None,  # all geoms randomized
            "randomize_friction": False,
            "randomize_solref": False,
            "randomize_solimp": False,
            # Joint parameters
            "joint_names": None,  # all joints randomized
            "randomize_stiffness": False,
            "randomize_frictionloss": True,
            "randomize_damping": True,
            "randomize_armature": True,
            # "stiffness_perturbation_ratio": 0.1,
            "frictionloss_perturbation_size": 0.05,
            "damping_perturbation_size": 0.01,
            "armature_perturbation_size": 0.01,
        }
        self.kp_perturbation_size = 30

        self.dynamics_modder = DynamicsModder(
            sim=self.sim,
            **self.dynamics_randomization_args,
        )

        self.dynamics_modder.save_defaults()

    def reset(self):
        self.dynamics_modder.restore_defaults()
        observations = super().reset()
        self.dynamics_modder.save_defaults()
        for name in self.dynamics_modder.joint_defaults.keys():
            jnt_id = self.sim.model.joint_name2id(name)
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            print(name + " damping: ", np.array(self.sim.model.dof_damping[dof_idx]))

        print("controller_kp:", self.robots[0].controller.kp)
        print("controller_kd:", self.robots[0].controller.kd)

        # domain randomize + regenerate observation
        self.dynamics_modder.randomize()
        self.robots[0].controller.kp += (np.random.rand()*2-1)*self.kp_perturbation_size
        self.robots[0].controller.kd = 2*np.sqrt(self.robots[0].controller.kp)
        print("controller_kp:", self.robots[0].controller.kp)
        print("controller_kd:", self.robots[0].controller.kd)

        print("after randomization")
        for name in self.dynamics_modder.joint_defaults.keys():
            jnt_id = self.sim.model.joint_name2id(name)
            dof_idx = [i for i, v in enumerate(self.sim.model.dof_jntid) if v == jnt_id]
            print(name + " damping: ", np.array(self.sim.model.dof_damping[dof_idx]))
        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        self.dynamics_modder.update_sim(self.sim)
        return observations

class Square_Peg_RL2_Thick_Square(Square_Peg_RL2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation
        SingleArmEnv._load_model(self)
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()
        bounds = self._get_initial_placement_bounds()


        nut = SquareNutThickObject(name="SquareNut")
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

        peg_min = np.array([0.029, 0.029, 0.059])/2
        peg_max = np.array([0.031, 0.031, 0.061])/2

        # peg = BoxObject(
        #     name="Peg",
        #     size_min=peg_min,
        #     size_max=peg_max,
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        #     density=100000
        # )

        peg = PegWithBaseObject(name="Peg")

        # define nuts
        self.nuts = [nut]
        self.objects = [nut, peg]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        nut_ranges = bounds["nut"]
        x, y, z_rot = nut_ranges["x"], nut_ranges["y"], nut_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"SquareNutSampler",
                mujoco_objects=nut,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            ),
        )

        peg_ranges = bounds["peg"]
        x, y, z_rot = peg_ranges["x"], peg_ranges["y"], peg_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"PegSampler",
                mujoco_objects=peg,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            ),
        )


        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg1_xml_pos[0] = -12
        peg1_xml_pos[1] = 10

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

class Square_Peg_RL2_Tilt(Square_Peg_RL2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_model(self):
        """
        Override to modify xml of pegs. This is necessary because the pegs don't have free
        joints, so we must modify the xml directly before loading the model.
        """

        # skip superclass implementation
        SingleArmEnv._load_model(self)
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()
        bounds = self._get_initial_placement_bounds()


        nut = SquareNutThickObject(name="SquareNut")
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

        peg_min = np.array([0.029, 0.029, 0.059])/2
        peg_max = np.array([0.031, 0.031, 0.061])/2

        # peg = BoxObject(
        #     name="Peg",
        #     size_min=peg_min,
        #     size_max=peg_max,
        #     rgba=[1, 0, 0, 1],
        #     material=redwood,
        #     density=100000
        # )

        peg = PegWithBaseTallTiltObject(name="Peg")

        # define nuts
        self.nuts = [nut]
        self.objects = [nut, peg]
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        nut_ranges = bounds["nut"]
        x, y, z_rot = nut_ranges["x"], nut_ranges["y"], nut_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"SquareNutSampler",
                mujoco_objects=nut,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            ),
        )

        peg_ranges = bounds["peg"]
        x, y, z_rot = peg_ranges["x"], peg_ranges["y"], peg_ranges["z_rot"]

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"PegSampler",
                mujoco_objects=peg,
                x_range=[x[0], x[1]],
                y_range=[y[0], y[1]],
                rotation=(z_rot[0], z_rot[1]),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            ),
        )


        # get xml element corresponding to both pegs
        peg1_xml = mujoco_arena.worldbody.find("./body[@name='peg1']")
        peg2_xml = mujoco_arena.worldbody.find("./body[@name='peg2']")

        # apply randomization
        peg1_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg1_xml_pos[0] = -12
        peg1_xml_pos[1] = 10

        # move peg2 completely out of scene
        peg2_xml_pos = string_to_array(peg1_xml.get("pos"))
        peg2_xml_pos[0] = -10.
        peg2_xml_pos[1] = 0.

        # set modified entry in xml
        peg1_xml.set("pos", array_to_string(peg1_xml_pos))
        peg2_xml.set("pos", array_to_string(peg2_xml_pos))

        # get collision checking entries
        peg1_size = string_to_array(peg1_xml.find("./geom").get("size"))
        peg2_size = string_to_array(peg2_xml.find("./geom").get("size"))
        self.peg1_horizontal_radius = np.linalg.norm(peg1_size[0:2], 2)
        self.peg2_horizontal_radius = peg2_size[0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _reset_internal(self):
        """
        Modify from superclass to keep sampling nut locations until there's no collision with either peg.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                # record peg initial pose
                if obj == self.objects[1]:
                    euler = robosuite.utils.transform_utils.mat2euler(robosuite.utils.transform_utils.quat2mat(obj_quat), "szyx")
                    offset = np.array([np.cos(euler[-1] - np.pi)*0.02, np.sin(euler[-1])*0.02, -0.0075])
                    self.peg_initial_pos = obj_pos + offset

        # Move objects out of the scene depending on the mode
        nut_names = {nut.name for nut in self.nuts}
        if self.single_object_mode == 1:
            self.obj_to_use = random.choice(list(nut_names))
            for nut_type, i in self.nut_to_id.items():
                if nut_type.lower() in self.obj_to_use.lower():
                    self.nut_id = i
                    break
        elif self.single_object_mode == 2:
            self.obj_to_use = self.nuts[self.nut_id].name
        if self.single_object_mode in {1, 2}:
            nut_names.remove(self.obj_to_use)
            self.clear_objects(list(nut_names))

        # Make sure to update sensors' active and enabled states
        if self.single_object_mode != 0:
            for i, sensor_names in self.nut_id_to_sensors.items():
                for name in sensor_names:
                    # Set all of these sensors to be enabled and active if this is the active nut, else False
                    self._observables[name].set_enabled(i == self.nut_id)
                    self._observables[name].set_active(i == self.nut_id)

    def on_peg(self, obj_pos, peg_pos):
        res = False
        # print(obj_pos[2] - self.table_offset[2])
        # print(obj_pos[0] - peg_pos[0])
        # print(obj_pos[1] - peg_pos[1])
        # print("="*20)
        if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < self.table_offset[2] + 0.055
        ):
            res = True
        return res

class Square_D2(Square_D1):
    """
    Even broader range for everything, and z-rotation randomization for peg.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

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
        return dict(
            nut=dict(
                x=(-0.25, 0.25),
                y=(-0.25, 0.25),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
            peg=dict(
                x=(-0.25, 0.25),
                y=(-0.25, 0.25),
                z_rot=(0., np.pi / 2.),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.82)),
            ),
        )
