import os
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from video_streaming.tf_video_analysis import TFVideoAnalysis
import constants_sensitive
import constants


TFVideoAnalysis(
            video_url_or_filename=f"rtsp://{constants_sensitive.GARDEN_RTSP_UNAME}:{constants_sensitive.GARDEN_RTSP_PASSWORD}@{constants.SECONDARY_RTSP_CAMERA_ADDRESS}", 
            camera_name="Test IP Cam",
            mqtt_client=None,
            debug_motion_tracking=True).start()
