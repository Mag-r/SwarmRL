from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp
import open3d as o3d

from swarmrl.observables.observable import Observable
from swarmrl.components.colloid import Colloid
import copy

class TopDownImage(Observable, ABC):
    """
    A class representing a top-down (camera-style) image observable in a simulation.

    Attributes:
        box_length (np.ndarray): The length of the simulation box.
        particle_type (int): The type of particles to include in the image.
        custom_mesh (optional): A custom mesh to use for the particles.
        is_2D (bool): Flag indicating whether the simulation is 2D or 3D.
        vis (o3d.visualization.Visualizer): The Open3D visualizer object.

    Methods:
        compute_observable: Computes the top-down image based on the given colloids.

    """

    def __init__(self,  box_length: np.ndarray, particle_type: int = 0, custom_mesh=None, is_2D=False):
        """
        Initializes a TopDownImage object.

        Args:
            box_length (np.ndarray): The length of the simulation box.
            particle_type (int, optional): The type of particles to include in the image. Defaults to 0.
            custom_mesh (optional): A custom mesh to use for the particles.
            is_2D (bool, optional): Flag indicating whether the simulation is 2D or 3D. Defaults to False.
        """
        super().__init__(particle_type=particle_type)
        self.box_length = box_length
        self.particle_type = particle_type
        self.custom_mesh = custom_mesh
        self.is_2D = is_2D
        # o3d.visualization.rendering.OffscreenRenderer(640,640)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)
        opt = self.vis.get_render_option()
        opt.light_on = True
        
        
    def create_bounding_box(self):
        """
        Creates a bounding box mesh for the simulation.

        """
        top = o3d.geometry.TriangleMesh.create_box(width=self.box_length[0], height=1000, depth=0.1)
        self.vis.add_geometry(top)

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


    def compute_observable(self, colloids: List[Colloid]):
        """
        Computes the top-down image based on the given colloids.

        Args:
            colloids (List[Colloid]): A list of Colloid objects representing the particles in the simulation.

        Returns:
            np.ndarray: The computed top-down image as a numpy array.
        """
        self.create_bounding_box()
        for colloid in colloids:
            if colloid.type == self.particle_type:
                if self.custom_mesh is not None:
                    particle = copy.deepcopy(self.custom_mesh)
                else:
                    particle = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                particle.translate(colloid.pos)
                if self.is_2D:
                    rot_matrix = particle.get_rotation_matrix_from_xyz([0,0,np.arctan2(colloid.director[1],colloid.director[0])])
                else:
                    rot_matrix = particle.get_rotation_matrix_from_xyz(colloid.director)
                particle.rotate(rot_matrix)
                particle.compute_vertex_normals()
                particle.paint_uniform_color([0.5,0.5,0.5])
                self.vis.add_geometry(particle)

        self.vis.poll_events()
        self.vis.update_renderer()
        image = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        self.vis.clear_geometries()
        return self.rgb2gray(image)




