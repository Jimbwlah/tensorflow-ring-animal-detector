import threading
import paho.mqtt.client as mqtt
from video_streaming.tf_video_analysis import TFVideoAnalysis
from console.logger import ConsolePrinter
import constants_sensitive
import constants

class ThreadedVideoDispatcher:

    """Threaded video dispatcher class"""
    def __init__(self,mqtt_client: mqtt.Client):
        
        # Process worker tracker
        # TODO - Turn this into a Pool
        self.processes = {}

        self.mqtt_client = mqtt_client

    def _threaded_video_dispatcher(self, video_stream, camera_name):
        TFVideoAnalysis(
            video_url_or_filename=video_stream, 
            camera_name=camera_name,
            mqtt_client=self.mqtt_client).start()

    def process_video(self, message_payload_decoded, camera_name, camera_id):
        
        # Message payload is either ON or OFF for motion/state
        if "ON" in message_payload_decoded:
            
            ConsolePrinter.print(f"{camera_name} ON, running detection")

            # Threaded video analysis
            thread = threading.Thread(
                target=self._threaded_video_dispatcher, 
                args=[f"rtsp://localhost:8554/{camera_id}_live", camera_name]
            )
            thread.daemon = True
            thread.start()
            self.processes[camera_name] = thread

            # Trigger analysis of an additional camera for the garden
            if "garden" in camera_name.lower() and camera_name.lower() != "front garden":
                thread_ipcam = threading.Thread(
                    target=self._threaded_video_dispatcher, 
                    args=[f"rtsp://{constants_sensitive.GARDEN_RTSP_UNAME}:{constants_sensitive.GARDEN_RTSP_PASSWORD}@{constants.SECONDARY_RTSP_CAMERA_ADDRESS}", f"{constants.SECONDARY_RTSP_CAMERA_NAME}"]
                )
                thread_ipcam.daemon = True
                thread_ipcam.start()
                self.processes["secondary_cam"] = thread_ipcam
            
        if "OFF" in message_payload_decoded:
            # Tidy up worker threads
            if camera_name in self.processes:
                self.processes[camera_name].join()

                if "secondary_cam" in self.processes:
                    self.processes["secondary_cam"].join()
            ConsolePrinter.print(f"{camera_name} OFF")