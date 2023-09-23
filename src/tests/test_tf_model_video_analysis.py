import os
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from video_streaming.tf_video_analysis import TFVideoAnalysis
from video_streaming.tf_class_tracker import ClassTracker
import constants

video_fname = constants.VIDEO_ANALYSIS_TEST_VIDEO_FILENAME

TFVideoAnalysis(
    video_url_or_filename=os.path.join(os.getcwd(),f"training-content/test-video/{video_fname}"), 
    camera_name="Local Test",
    video_file_analysis=True,
    mqtt_client=None,
    disable_class_tracker=True).start()
