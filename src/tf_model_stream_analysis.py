# Python tool to run a TFLite model against a RTSP video stream.
# Original Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py

# Import packages
from mqtt.mqtt_handler import MQTTHandler

class TFModelStream:

    def __init__(self):
        pass
        
    @classmethod
    def start_video_analysis_monitor(self):
        MQTTHandler().start_mqtt()

if __name__ == '__main__':
    TFModelStream.start_video_analysis_monitor()