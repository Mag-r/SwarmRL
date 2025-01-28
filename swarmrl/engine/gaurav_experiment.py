import rospy
import numpy as np


from processing_ROS.msg import Coilfreq
from sensor_msgs.msg import Image

from swarmrl.engine.engine import Engine
from swarmrl.engine.gaurav_sim import GauravSim
from swarmrl.force_functions.global_force_fn import GlobalForceFunction
from swarmrl.actions.mpi_action import MPIAction


class GauravExperiment(Engine):
    def __init__(self, simulation: GauravSim):
        super().__init__()
        self.simulation = simulation
        rospy.init_node("srl_controller")
        self.image_subscriber = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.action_publisher = rospy.Publisher("/control", Coilfreq, queue_size=10)
        self.image = None

    def image_callback(self, msg):
        self.image = np.frombuffer(msg.data).reshape(msg.height, msg.width)
        self.image = self.image[::4, ::4, np.newaxis]


    def create_action_message(self, action: MPIAction, shutdown: bool = False):
        #maybe change later
        message = Coilfreq()
        action = self.clip_actions(action)
        message.double_signal = True
        message.enable_coils = not shutdown # change later
        message.general_stop = shutdown
        message.B1 = action.amplitudes[0]
        message.f1 = action.frequencies[0]
        
        message.Bx = 1
        message.By = action.amplitudes[1]/message.B1
        message.fx = 1
        message.fy = action.frequencies[1]/message.f1
        message.Bx1 = action.offsets[0]/message.B1
        message.By1 = action.offsets[1]/message.B1
        return message
        
    def clip_actions(self, action: MPIAction, max_amplitude: float = 0.01):
        action.magnetic_field = np.clip(action.magnetic_field, 0, max_amplitude)
        return action
        
        

    def integrate(
        self,
        n_slices: int,
        force_model: GlobalForceFunction,
    ) -> None:
        """
        Perform the real-experiment equivalent of an integration step.

        Parameters
        ----------
        n_slices : int
            Number of slices to integrate.
        force_model : ForceFunction
            The force model to use for integration.
        """
        try:
            for _ in range(n_slices):
                if self.image is not None:
                    action = force_model.calc_action(self.image)
                    action = self.simulation.convert_actions_to_sim_units(action)
                    message = self.create_action_message(action) 
                    self.action_publisher.publish(message)   
                else:
                    rospy.logwarn("No image received")
                rospy.sleep(0.1)
        finally:
            message = self.create_action_message(MPIAction(np.zeros(2), 0), shutdown=True)
            self.action_publisher.publish(message)
            
