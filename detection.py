import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import json
import uuid
import time
import datetime
import threading
import logging
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# AWS IoT Core Configuration
AWS_IOT_ENDPOINT = ""
CLIENT_ID = ""
CERT_PATH = ""
PRIVATE_KEY_PATH = ""
ROOT_CA_PATH = ""
TOPIC = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.mqtt_client = None
        self.detection_history = {}  # To prevent duplicate messages for the same person
        self.detection_cooldown = 5  # Seconds to wait before sending another detection for the same ID
        self.mqtt_lock = threading.Lock()
        self.device_id = f"raspberry-pi-5-{uuid.uuid4().hex[:6]}"
        self.setup_aws_iot()

    def new_function(self):  # New function example
        return "The meaning of life is: "
    
    def setup_aws_iot(self):
        """Set up and connect to AWS IoT Core"""
        try:
            # Create an MQTT client
            self.mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
            self.mqtt_client.configureEndpoint(AWS_IOT_ENDPOINT, 8883)
            self.mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
            
            # Configure connection parameters
            self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
            self.mqtt_client.configureOfflinePublishQueueing(-1)  # Infinite offline queueing
            self.mqtt_client.configureDrainingFrequency(2)  # Draining: 2 Hz
            self.mqtt_client.configureConnectDisconnectTimeout(10)  # 10 sec
            self.mqtt_client.configureMQTTOperationTimeout(5)  # 5 sec
            
            # Connect to AWS IoT Core
            logger.info("Connecting to AWS IoT Core...")
            connected = self.mqtt_client.connect()
            if connected:
                logger.info("Connected to AWS IoT Core")
            else:
                logger.error("Failed to connect to AWS IoT Core")
        except Exception as e:
            logger.error(f"Error setting up AWS IoT: {e}")
            self.mqtt_client = None
    
    def send_detection_to_aws(self, detection_count, detections_data):
        """Send person detection data to AWS IoT Core"""
        if self.mqtt_client is None:
            logger.warning("MQTT client not initialized. Cannot send detection.")
            return
        
        with self.mqtt_lock:
            try:
                # Prepare the message payload
                timestamp = datetime.datetime.now().isoformat()
                
                message = {
                    "device_id": self.device_id,
                    "timestamp": timestamp,
                    "detection_count": detection_count,
                    "detections": detections_data
                }
                
                # Convert the message to JSON
                message_json = json.dumps(message)
                
                # Publish the message to the AWS IoT topic
                result = self.mqtt_client.publish(TOPIC, message_json, 1)
                
                if result:
                    logger.info(f"Successfully published detection: {detection_count} people detected")
                else:
                    logger.error("Failed to publish detection to AWS IoT Core")
                    
            except Exception as e:
                logger.error(f"Error sending detection to AWS IoT: {e}")

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    detections_data = []
    current_time = time.time()
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
                
            string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
            
            # Add detection data to the list
            detection_info = {
                "track_id": int(track_id),
                "confidence": float(confidence),
                "bbox": {
                    "x": float(bbox.x),
                    "y": float(bbox.y),
                    "width": float(bbox.width),
                    "height": float(bbox.height)
                }
            }
            detections_data.append(detection_info)
            
            # Check if we should send an update for this track_id
            should_send = False
            if track_id not in user_data.detection_history:
                should_send = True
            elif (current_time - user_data.detection_history.get(track_id, 0)) > user_data.detection_cooldown:
                should_send = True
                
            if should_send:
                user_data.detection_history[track_id] = current_time
    
    # If people are detected, send the data to AWS IoT Core
    if detection_count > 0 and detections_data:
        # Send detection using a separate thread to avoid blocking the pipeline
        threading.Thread(
            target=user_data.send_detection_to_aws,
            args=(detection_count, detections_data),
            daemon=True
        ).start()
    
    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Add AWS IoT status
        connection_status = "Connected" if user_data.mqtt_client else "Disconnected"
        cv2.putText(frame, f"AWS IoT: {connection_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()