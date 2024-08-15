import open3d as o3d
import numpy as np
import logging
from abc import ABC
from typing import List
import copy
from swarmrl.observables.observable import Observable
from swarmrl.components.colloid import Colloid

class TopDownImage(Observable, ABC):
    """
    The image produced by a top-down view of the scene. Produces an grayscale image similar to a camera image.
    """
    def __init__(self, box_length: np.ndarray, particle_type: int = 0, custom_mesh=None, is_2D=False):
        """
        Initializes the TopDownImage object.

        Args:
            box_length (np.ndarray): The length of the box.
            particle_type (int, optional): The type of particle. Defaults to 0.
            custom_mesh (None, optional): Custom mesh. Defaults to None.
            is_2D (bool, optional): Flag indicating if the image is 2D. Defaults to False.
        """
        self.logger = logging.getLogger(__name__)
        super().__init__(particle_type=particle_type)
        self.box_length = box_length
        self.particle_type = particle_type
        self.custom_mesh = custom_mesh
        self.is_2D = is_2D
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 1280)
        self.renderer.scene.set_background([0, 0, 0, 1])  # Set background to black
        self.logger.debug("TopDownImage initialized")

    def create_bounding_box(self) -> None:
        """
        Creates a bounding box in the scene.

        Returns:
            None
        """
        bounding_box = o3d.geometry.TriangleMesh.create_box(width=self.box_length[0], height=self.box_length[1], depth=self.box_length[2])
        bounding_box.paint_uniform_color([1, 1, 1])  # Set bounding box color to white
        self.renderer.scene.add_geometry("bounding_box", bounding_box, o3d.visualization.rendering.MaterialRecord())
        self.logger.debug("Bounding box created")

    def rgb2gray(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image to grayscale.

        Parameters:
            rgb (np.ndarray): The input RGB image.

        Returns:
            np.ndarray: The grayscale image.

        """
        return np.dot(np.array(rgb[..., :3]), np.array([0.2989, 0.5870, 0.1140]))


    def positionCameraAboveCenter(self) -> None:
        """
        Positions the camera above the center of the scene. Works only for Gaurav's microbots.

        This method calculates the center of the scene's bounding box and positions the camera
        above it. The camera is set up with a 60-degree field of view, and the up direction is
        set to [0, 1, 0].

        Parameters:
            None

        Returns:
            None
        """
        bbox = self.renderer.scene.bounding_box
        center = bbox.get_center()
        eye = center + np.array([0, 0, 10000])  # Position the camera above the center
        up = np.array([0, 1, 0])  # Up direction
        self.renderer.setup_camera(60, center, eye, up)
        self.logger.debug(f"Camera setup with center: {center}, eye: {eye}, up: {up}")

    def add_colloids_to_image(self, colloids: List[Colloid]) -> None:
        """
        Adds colloids to the image.

        Args:
            colloids (List[Colloid]): A list of colloids to be added to the image.

        Returns:
            None
        """
        for colloid in colloids:
            if colloid.type == self.particle_type:
                if self.custom_mesh is not None:
                    particle = copy.deepcopy(self.custom_mesh)
                else:
                    particle = o3d.geometry.TriangleMesh.create_sphere(radius=1000)  # Adjust radius if needed
                particle.translate(colloid.pos)
                if self.is_2D:
                    rot_matrix = particle.get_rotation_matrix_from_xyz([0, 0, np.arctan2(colloid.director[1], colloid.director[0])])
                else:
                    rot_matrix = particle.get_rotation_matrix_from_xyz(colloid.director)
                particle.rotate(rot_matrix)
                particle.compute_vertex_normals()
                particle.paint_uniform_color([0.5, 0.5, 0.5])
                self.renderer.scene.add_geometry(f"particle_{colloid.id}", particle, o3d.visualization.rendering.MaterialRecord())
        self.logger.debug("Colloids added to the image")


    def compute_observable(self, colloids: List[Colloid]):
        """
        Computes the top-down image observable based on the given colloids.

        Args:
            colloids (List[Colloid]): A list of colloids representing the objects in the scene.

        Returns:
            np.ndarray: The computed top-down image as a numpy array.
        """
        self.create_bounding_box()
        self.add_colloids_to_image(colloids)
        self.positionCameraAboveCenter()
        image = self.renderer.render_to_image()
        self.logger.debug("Screen captured")
        self.renderer.scene.clear_geometry()
        self.logger.info("Top-down image computed")
        image=self.rgb2gray(np.asarray(image))
        return np.asarray(image)

