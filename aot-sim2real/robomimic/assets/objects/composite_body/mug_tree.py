import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.utils.transform_utils as T

class MugTreeObject(CompositeObject):
    """
    Generates a four-walled bin container with an open top.
    Args:
        name (str): Name of this Bin object
        bin_size (3-array): (x,y,z) full size of bin
        wall_thickness (float): How thick to make walls of bin
        transparent_walls (bool): If True, walls will be semi-translucent
        friction (3-array or None): If specified, sets friction values for this bin. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        base_size=(0.16, 0.16, 0.03),
        branch_height=0.12,
        branch_size = (0.08, 0.005, 0.015),
        tree_size=(0.03, 0.03, 0.16),
        friction=None,
        density=5000.,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.base_size = np.array(base_size)
        self.branch_height = branch_height
        self.branch_size = np.array(branch_size)
        self.tree_size = np.array(tree_size)
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.bin_mat_name = "light_wood_mat"

        # Element references
        self._base_geom = "base"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        bin_mat = CustomMaterial(
            texture="WoodLight",
            tex_name="light_wood",
            mat_name=self.bin_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(bin_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": (self.base_size + np.array([0, 0, self.tree_size[2]])) / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Base
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, self.base_size[2] / 2 - (self.tree_size[2]+self.base_size[2])/2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.base_size/2,
            geom_names=self._base_geom,
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )
        # Tree
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, self.base_size[2] + self.tree_size[2] / 2 - (self.tree_size[2]+self.base_size[2])/2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.tree_size / 2,
            geom_names="tree",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )
        # Branches
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(self.tree_size[0]/2 + self.branch_size[0]/2, 0, self.base_size[2] + self.branch_height - (self.tree_size[2]+self.base_size[2])/2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.branch_size / 2,
            geom_names="branches",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )
        # Add back in base args and site args
        obj_args.update(base_args)
        # obj_args.update({"joints": None})
        # Return this dict
        return obj_args

    @property
    def base_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to bin base
        """
        return [self.correct_naming(self._base_geom)]