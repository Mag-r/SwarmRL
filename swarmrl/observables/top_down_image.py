import numpy as np
import logging
from abc import ABC
from typing import List
import copy
import matplotlib.pyplot as plt
from swarmrl.observables.observable import Observable
from swarmrl.components.colloid import Colloid
import os
import shutil
import threading
import queue
os.environ['EGL_PLATFORM'] = 'surfaceless' 
import open3d as o3d

class TopDownImage(Observable, ABC):
    """
    The image produced by a top-down view of the scene. Produces an grayscale image similar to a camera image.
    """

    def __init__(
        self,
        box_length: np.ndarray,
        image_resolution: np.ndarray = np.array([1280, 1280]),
        particle_type: int = 0,
        custom_mesh=None,
        is_2D=True,
        save_images=False,
    ):
        """
        Initializes the TopDownImage object. (Works only for MPI Rafts)

        Args:
            box_length (np.ndarray): The length of the box.
            image_resolution (np.ndarray, optional): The resolution of the image. Defaults to np.array([1280,1280]).
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
        self.image_resolution = image_resolution

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            image_resolution[0], image_resolution[1]
        )
        self.renderer.scene.set_background([0, 0, 0, 1])  # Set background to black

        sun_dir = np.array([0, 1, -1], dtype=np.float32)
        self.renderer.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            sun_dir,
        )

        self.save_images = save_images
        if self.save_images:
            # Clear the directory 'images' if it exists
            if os.path.exists("images"):
                shutil.rmtree("images")
            os.makedirs("images")
            self.image_count = 0
            self.image_queue = queue.Queue()
            self.image_saving_thread = threading.Thread(
                target=self.save_images_async, daemon=True
            )
            self.image_saving_thread.start()
        self.create_bounding_box()
        self.logger.debug("TopDownImage initialized")

    def save_images_async(self):
        while True:
            if not self.image_queue.empty():
                image = self.image_queue.get()
                plt.imsave("images/latest_image.png", image, cmap="gray")
            else:
                threading.Event().wait(0.1)

    def create_bounding_box(self) -> None:
        """
        Creates a bounding box in the scene.

        Returns:
            None
        """
        bounding_box = o3d.geometry.TriangleMesh.create_box(
            width=self.box_length[0],
            height=self.box_length[1],
            depth=self.box_length[2],
        )
        bounding_box.paint_uniform_color([0, 0, 0])  # Set bounding box color to black
        self.renderer.scene.add_geometry(
            "bounding_box", bounding_box, o3d.visualization.rendering.MaterialRecord()
        )
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
        """np.asarray(image.reshape(1,self.image_resolution[0],self.image_resolution[1]))
        Parameters:
            None

        Returns:
            None
        """
        center = self.box_length / 2  # Center of the box
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
        for id, colloid in enumerate(colloids):
            if self.custom_mesh is not None:
                particle = copy.deepcopy(self.custom_mesh)
            else:
                particle = o3d.geometry.TriangleMesh.create_sphere(
                    radius=1000
                )  # Adjust radius if needed
            if np.shape(colloid.pos)[0] == 2:
                colloid.pos = np.append(colloid.pos, 0)
            particle.translate(colloid.pos)
            if self.is_2D:
                rot_matrix = particle.get_rotation_matrix_from_xyz(
                    [0, 0, colloid.alpha]
                )
            else:
                rot_matrix = particle.get_rotation_matrix_from_xyz(colloid.director)
            particle.rotate(rot_matrix)
            particle.compute_vertex_normals()
            self.renderer.scene.add_geometry(
                f"particle_{id}", particle, o3d.visualization.rendering.MaterialRecord()
            )
        self.logger.debug("Colloids added to the image")

    def compute_observable(self, colloids: List[Colloid]):
        """f
        Computes the top-down image observable based on the given colloids.

        Args:
            colloids (List[Colloid]): A list of colloids representing the objects in the scene.

        Returns:
            np.ndarray: The computed top-down image as a numpy array.
        """
        # self.create_bounding_box()
        self.add_colloids_to_image(colloids)
        self.positionCameraAboveCenter()
        image = self.renderer.render_to_image()
        self.logger.debug("Screen captured")
        self.renderer.scene.clear_geometry()
        self.logger.debug("Top-down image computed")
        image = self.rgb2gray(np.asarray(image))
        image = np.asarray(image)
        if self.save_images:
            if self.image_queue.qsize() > 3:
                self.logger.warning(
                    f"Image queue is starting to fill. Current size {self.image_queue.qsize()}"
                )
            self.image_queue.put(image)
        return np.asarray(
            image.reshape(1, self.image_resolution[0], self.image_resolution[1], 1)
        )
