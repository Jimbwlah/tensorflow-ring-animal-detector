"""This module defines the project-level constants."""

##### DETECTION OPTIONS #####

MODEL_LOCATION = "src\models\custom_trained\ssdmobilenet_v2_640_V2"
DETECT_TF_FILENAME = "detect.tflite"
LABEL_MAP_FILENAME = "training-content\labelmap.txt"
DETECTION_ACCURACY_THRESHOLD = 0.6 # 60%

VIDEO_ANALYSIS_RESOLUTION = "854x480"
VIDEO_ANALYSIS_MAX_TIME = 60
VIDEO_ANALYSIS_FPS_MAX = 10 # Streamed RTSP FPS
VIDEO_ANALYSIS_FILE_FPX_MAX = 24 # Local video file FPS
VIDEO_ANALYSIS_ANIMAL_FILTER = ['fox', 'badger']

VIDEO_ANALYSIS_TEST_VIDEO_FILENAME = "<test_video_filename>.mp4"

SECONDARY_RTSP_CAMERA_ADDRESS = "<rtsp_address>"
SECONDARY_RTSP_CAMERA_NAME = "<rtsp camera name>"

IMAGE_SNAPSHOT_FOLDER_LOCATION = "<snapshot_location>" # Location to store snapshot images
IMAGE_SNAPSHOT_TIME_BETWEEN_SNAPSHOTS = 3 # Time to wait between taking snapshots of detected animals

FURBINATOR_START_PAUSE_TIME = "07:00"
FURBINATOR_END_PAUSE_TIME = "16:00"

##### MQTT #####

MQTT_HOSTNAME = "localhost"
MQTT_PORT = 1883

MQTT_RING_LOCATION_ID = "<ring_mqtt_location_id>"
MQTT_RING_CAMERA_MAPPING = {
        "<ring_mqtt_camera_1_id>":"<ring_mqtt_camera_1_name>",
        "<ring_mqtt_camera_2_id>":"<ring_mqtt_camera_2_name>"
    }

##### Training #####

TRAINING_NUM_STEPS = 40000
TRAINING_BATCH_SIZE = 16 # For SSD MobileNet V2
# TRAINING_BATCH_SIZE = 2 # For EfficientDet
TRAINING_MODEL_FOLDER_NAME = "ssd-mobilenet-v2-fpnlite-320"
# TRAINING_MODEL_FOLDER_NAME = "efficientdet_d0_coco17_tpu-32"

##### mAP Evaluation #####

MAP_MODEL_TO_TEST = "ssdmobilenet_v2_640_v2"
MAP_CONFIDENCE_THRESHOLD = 0.5
MAP_NUM_IMAGES_TO_TEST = 25
