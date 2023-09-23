
import os
import cv2
import paho.mqtt.client as mqtt
import constants
from console.logger import ConsolePrinter
from datetime import datetime


class FurbinatorActivator:

    """
    Class to activate the Furbinator 3000
    """
    def __init__(self,mqtt_client:mqtt.Client):
        self.mqtt_client = mqtt_client # This needs to be injected through DI - this chaining works for now
        self.snapshot_image_path=os.path.join(constants.IMAGE_SNAPSHOT_FOLDER_LOCATION)

        self.start_pause_time = datetime.strptime(constants.FURBINATOR_START_PAUSE_TIME, "%H:%M").time()
        self.end_pause_time = datetime.strptime(constants.FURBINATOR_END_PAUSE_TIME, "%H:%M").time()
        pass

    def _is_pause_time(self):
        current_time = datetime.now().time()
        return self.start_pause_time <= current_time <= self.end_pause_time


    def activate_furbinator(self,camera_name):
        if self._is_pause_time() != True:
            if "garden" in camera_name.lower() and camera_name.lower() != "front garden":
                self.mqtt_client.publish("fox", " **strobe")

    def snapshot_target(self,frame,object_name,time_found: datetime):
        if self._is_pause_time() != True:
            image_filename = self.snapshot_image_path + "/" + object_name + "_" + time_found.strftime("%H-%M-%S") + ".jpg"
            ConsolePrinter.print("Saving image " + image_filename)
            cv2.imwrite(image_filename, frame)