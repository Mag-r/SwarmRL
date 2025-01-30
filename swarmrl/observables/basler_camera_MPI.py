"""
Position observable computer.
"""

from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp
import cv2
from pypylon import pylon
import logging
from matplotlib import pyplot as plt

from swarmrl.observables.observable import Observable

logger = logging.getLogger(__name__)
class BaslerCameraObservable(Observable, ABC):
    """
    Class to receive images from Gauravs Camera.
    some abuse of the observable class to return the image directly.
    """
    camParam = {
        "camName": "40011027",
        "topicName": "/image",
        "fps": 10,
        "exposureTime": 5000,
        "width": 2048,
        "height": 2048,
        "xOffset": 128,
        "yOffset": 0,
        "xReverse": False,
        "yReverse": False,
        "scale": 0.5,
        "lightSource": "Daylight5000K",
        "balanceRatioSelector_r": "Red",
        "balanceRatio_r": 1.3,
        "balanceRatioSelector_g": "Green",
        "balanceRatio_g": 1.0,
        "balanceRatioSelector_b": "Blue",
        "balanceRatio_b": 1.6,
        "colored": True,
    }

    def __init__(self, resolution: List[int]):
        """
        Constructor for the observable.
        """
        super().__init__(particle_type=0) #better way to do this?
        self.resolution = resolution
        tlf = pylon.TlFactory.GetInstance()
        detected_cameras = tlf.EnumerateDevices()
        self.camera = None
        self.image_count = 0
        for cam in detected_cameras:
            serial_number = cam.GetFriendlyName().split()[-1][1:-1]
            if serial_number == self.camParam["camName"]:
                self.camera = pylon.InstantCamera(tlf.CreateDevice(cam))
                self.init_camera()
                logger.info(f"Camera {self.camParam['camName']} found and init.")
                break
        if self.camera is None or not self.camera.IsPylonDeviceAttached():
            logger.error(f"Camera {self.camParam['camName']} not found.")
            raise Exception(f"Camera {self.camParam['camName']} not found.")
    def init_camera(self):
        self.camera.Open()
        self.camera.AcquisitionMode.SetValue("Continuous")
        self.camera.AcquisitionFrameRate.SetValue(self.camParam['fps'])
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.ExposureAuto.SetValue('Off')
        self.camera.ExposureTime.SetValue(self.camParam['exposureTime'])
        self.camera.DeviceLinkThroughputLimitMode.SetValue('Off')
        self.camera.ReverseX.SetValue(self.camParam['xReverse'])
        self.camera.ReverseY.SetValue(self.camParam['yReverse'])
        self.camera.Width.SetValue(self.camParam['width'])
        self.camera.Height.SetValue(self.camParam['height'])
        self.camera.OffsetX.SetValue(self.camParam['xOffset'])
        self.camera.OffsetY.SetValue(self.camParam['yOffset'])
        if not self.camera.PixelFormat().count('Mono'):
            self.camera.LightSourcePreset.SetValue(self.camParam['lightSource'])
            self.camera.BalanceWhiteAuto.SetValue('Off')
            self.camera.BalanceRatioSelector.SetValue(self.camParam['balanceRatioSelector_r'])
            self.camera.BalanceRatio.SetValue(self.camParam['balanceRatio_r'])
            self.camera.BalanceRatioSelector.SetValue(self.camParam['balanceRatioSelector_g'])
            self.camera.BalanceRatio.SetValue(self.camParam['balanceRatio_g'])
            self.camera.BalanceRatioSelector.SetValue(self.camParam['balanceRatioSelector_b'])
            self.camera.BalanceRatio.SetValue(self.camParam['balanceRatio_b'])
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        

    def compute_observable(self, colloids: np.ndarray) -> np.ndarray:
        """
        Abuse of the compute_observable method to return the image directly obtained from the experiment.
        Parameters
        ----------
        colloids : should be None
        """
        if colloids is not None:
            raise ValueError("Colloids should be None for this observable.")
        if self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = grabResult.GetArray()
                image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)
                
                image = cv2.resize(image, (self.resolution[0], self.resolution[1]))
                grabResult.Release()
            else:
                logger.error("Grab failed.")
                image = np.zeros(self.resolution)
        plt.imshow(image,cmap='gray')
        plt.axis('off')
        plt.savefig(f'images/top_down_image_{self.image_count:03d}.png')
        self.image_count = self.image_count + 1
        image = np.array(image[np.newaxis, :, :, np.newaxis])
        return image
