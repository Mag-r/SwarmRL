import numpy as np
import pypylon
from jax import numpy as jnp
import logging
import threading
import socket
import time

from swarmrl.engine.engine import Engine
from swarmrl.engine.gaurav_sim import GauravSim
from swarmrl.force_functions.global_force_fn import GlobalForceFunction
from swarmrl.actions.mpi_action import MPIAction

logger = logging.getLogger(__name__)


class GauravExperiment(Engine):

    labview_port = 6344
    labview_ip = "134.105.56.173"
    closing_message = "S_Goodbye".encode("utf-8")
    TDMS_file_name = "H_".encode("utf-8")  

    def __init__(self, simulation: GauravSim, update_rate: float = 20.0):
        super().__init__()
        self.simulation = simulation
        self.update_rate = update_rate
        self.labview_listener = None
        self.labview_publisher = None
        self.server_connection = None
        self.keep_publishing = threading.Event()
        self.publishing_thread = None
        self.message_to_publish = " ".encode("utf-8")
        self.colloids = None
        self.establish_connection()
        self.lock = threading.Lock()
        self.keeping_time = 5

    def establish_connection(self):
        """Establish TCP connections with LabVIEW"""
        self.labview_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.labview_publisher = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info("Attempting to establish connection with LabVIEW at %s:%d", self.labview_ip, self.labview_port)
        self.labview_listener.settimeout(100)

        self.labview_listener.connect((self.labview_ip, self.labview_port))
        starting_message = self.receive_message()

        if starting_message and "Start" in starting_message:
            parts = starting_message.split(",")
            time_port = int(parts[2])
            # Close listener connection
            self.labview_listener.sendall(self.closing_message)
            # Start server for publishing
            self.labview_publisher.settimeout(10)
            self.labview_publisher.bind(("", time_port))
            self.labview_publisher.listen(1)

            self.server_connection, _ = self.labview_publisher.accept()
            self.server_connection.settimeout(0.001)

            logger.info("Connected!")

            # Send TDMS filename if required
            self.send_message(self.TDMS_file_name)

            # Start background thread for sending messages
            self.keep_publishing.set()
            self.publishing_thread = threading.Thread(
                target=self._publish_loop, daemon=True
            )
            self.publishing_thread.start()

    def receive_message(self):
        """Receive messages from LabVIEW listener."""
        try:
            message_length = int(self.labview_listener.recv(8).decode())
            return self.labview_listener.recv(message_length).decode()
        except Exception:
            return None

    def send_message(self, message: str):
        """Send a message to the LabVIEW publisher."""
        if not isinstance(message, bytes):
            message = message.encode("utf-8")
        if self.server_connection:
            try:
                self.server_connection.sendall(
                    len(message).to_bytes(4, byteorder="big")
                )
                self.server_connection.sendall(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                if message != self.closing_message:
                    self.stop_publishing()

    def update_message(self, new_message: str):
        """Update the message to be published."""
        with self.lock:
            self.message_to_publish = new_message.encode("utf-8")

    def _publish_loop(self):
        """Continuously send updated messages at the given update rate."""
        while self.keep_publishing.is_set():
            start_time = time.time()
            self.send_message(self.message_to_publish)
            elapsed_time = time.time() - start_time
            self.message_to_publish = "R_{:.6f}".format(time.time()).encode("utf-8")
            time.sleep(max(0, (1 / self.update_rate) - elapsed_time))

    def finalize(self):
        self.stop_publishing()
        return super().finalize()

    def stop_publishing(self):
        """Stop the publishing thread and close connections."""
        self.keep_publishing.clear()
        if self.publishing_thread:
            self.publishing_thread.join()

        if self.labview_listener:
            self.labview_listener.close()

        if self.server_connection:
            self.send_message(self.closing_message)
            self.server_connection.close()

        logger.info("Publishing stopped and connections closed.")

    def send_action(self, action: MPIAction):
        """Send an action to LabVIEW."""
        # Convert action to string message and send

        action = self.clip_actions(action)
        action_message = f"M_0.0_{action.magnitude[0]:.03f}_{action.magnitude[1]:.03f}_{action.frequency[0]:.03f}_{action.frequency[1]:.03f}_{action.keep_magnetic_field:.03f}_{action.gradient[0]:.03f}_{action.gradient[1]:.03f}"
        
        self.update_message(action_message)

    def clip_actions(
        self, action: MPIAction, max_amplitude: float = 90, max_frequency: float = 40
    ):
        """Clip the action values."""
        action.magnitude = np.clip(action.magnitude, -max_amplitude, max_amplitude)
        action.frequency = np.clip(action.frequency, 1, max_frequency + 1) - 1  # if freq is below 1 it is set to zero
        return action

    def seperate_rafts(self):
        """Seperate the rafts by sending a strong magnetic field."""
        
        seperation_action = MPIAction(
            magnitude=[100, 100], frequency=[30, 30], keep_magnetic_field=10
        )
        self.send_action(seperation_action)
        time.sleep(10)
    
    
    def integrate(self, n_slices: int, force_model: GlobalForceFunction):
        """Perform a real-experiment equivalent of an integration step."""

        force_model.set_training_mode(True)
        start = time.time()
        for _ in range(n_slices):
            action = force_model.calc_action(None)
            action = MPIAction(
                magnitude=[30, 30], frequency=[11, 6], keep_magnetic_field=1, gradient=action
            )
            self.send_action(action)
            time.sleep(1)
            force_model.set_training_mode(False)
            while time.time() - start < self.keeping_time:
                force_model.calc_action(None)
                time.sleep(1)

            start = time.time()
            force_model.set_training_mode(True)
            force_model.calc_reward(self.colloids)
