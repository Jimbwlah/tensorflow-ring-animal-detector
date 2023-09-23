# Import packages
import keyboard
import paho.mqtt.client as mqtt
from datetime import datetime
from video_streaming.threaded_video_dispatcher import ThreadedVideoDispatcher
import constants
from console.logger import ConsolePrinter

class MQTTHandler:
    
    """Class to handle all MQTT functionality"""
    def __init__(self):

        # Instatiate the mqtt client
        self.client = mqtt.Client("VideoStreamTFAnalysisServer")

        # MQTT filter topic list
        self.topic_filter_list = ['motion/state']
        
        # Define cameras
        self.cameras = constants.MQTT_RING_CAMERA_MAPPING

    def start_mqtt(self):

        # Connect to MQTT
        self.client.on_message=self.on_message
        self.client.on_publish=self.on_publish
        self.client.connect(constants.MQTT_HOSTNAME, constants.MQTT_PORT)
        self.client.subscribe("#") # Subscribe to all topics
        ConsolePrinter.print("Connected to MQTT broker")

        self.client.loop_start()

        while True:
            if keyboard.read_key() == "esc":
                self.client.loop_stop()
                self.client.disconnect()
                ConsolePrinter.print("MQTT terminated due to key press")
                break
        
    def on_disconnect(self, client, userdata,rc=0):
        ConsolePrinter.print("Disconnected result code "+str(rc))
        self.client.loop_stop()

    def print_topic_message(self,topic=""):
        time_now = datetime.now().strftime("%H-%M-%S")
        print("")
        ConsolePrinter.print(f"****** Message received @ {time_now} ******")
        ConsolePrinter.print(f"Topic = {topic}")

    def handle_message(self, search_text, message_payload, camera_name, camera_id, message_topic):

        # Ignore Ring-MQTT snapshots
        if "snapshot" not in message_topic:
            if search_text in message_topic:
                if len(message_topic) > 0:

                    self.print_topic_message(message_topic)

                    # Handle motion detection mqtt topics and start video analysis
                    if "motion/state" in message_topic:
                        message_payload_decoded = str(message_payload.decode("utf-8"))

                        ThreadedVideoDispatcher(self.client).process_video(
                            message_payload_decoded=message_payload_decoded,
                            camera_name=camera_name,
                            camera_id=camera_id)

    def on_message(self, client, userdata, message):
        
        # Filter MQTT topics based on filter list
        search_hit = bool([ele for ele in self.topic_filter_list if(ele in message.topic)])
        ring_pretext = f"ring/{constants.MQTT_RING_LOCATION_ID}/camera/"

        try:
            if search_hit:
                for camera_id,camera_name in self.cameras.items():
                    self.handle_message(f"{ring_pretext}{camera_id}/", message.payload, camera_name, camera_id, message.topic)

        except Exception as inst:
            ConsolePrinter.print("Issue processing/receiving")
            ConsolePrinter.print(inst.args) 
            ConsolePrinter.print(inst)

    def on_publish(self,client,userdata,result):  
        ConsolePrinter.print("MQTT Publish detected, details are - " + str(userdata))