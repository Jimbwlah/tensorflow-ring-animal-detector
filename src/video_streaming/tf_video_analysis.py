import os
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from video_streaming.video_stream import VideoStream
import constants
from console.logger import ConsolePrinter
from tensorflow.lite.python.interpreter import Interpreter
from video_streaming.tf_class_tracker import ClassTracker

class TFVideoAnalysis:

    """Tensorflow Analysis of a VideoStream or Video"""
    def __init__(
            self,
            video_url_or_filename,
            mqtt_client: mqtt.Client,
            camera_name="",
            video_file_analysis=False,
            disable_class_tracker=False,
            debug_motion_tracking=False):
        self.cwd = os.getcwd()
        self.model_name = constants.MODEL_LOCATION
        self.video_url_or_filename = video_url_or_filename
        self.graph_name = constants.DETECT_TF_FILENAME
        self.label_location = constants.LABEL_MAP_FILENAME
        self.threshold = float(constants.DETECTION_ACCURACY_THRESHOLD)
        self.resolution = constants.VIDEO_ANALYSIS_RESOLUTION
        resW, resH = self.resolution.split('x')
        self.resolution_w = int(resW)
        self.resolution_h = int(resH)
        self.edge_tpu = ""
        self.framerate = constants.VIDEO_ANALYSIS_FPS_MAX
        self.video_uptime = constants.VIDEO_ANALYSIS_MAX_TIME
        self.camera_name = camera_name
        self.video_file_analysis=video_file_analysis
        self.disable_class_tracker = disable_class_tracker

        if video_file_analysis:
            self.framerate = constants.VIDEO_ANALYSIS_FILE_FPX_MAX

        self.mqtt_client = mqtt_client
        self.class_tracker = ClassTracker(mqtt_client)

        self.debug_motion_tracking = debug_motion_tracking
        

    def start(self):

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(self.cwd,self.model_name,self.graph_name)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(self.label_location)

        ConsolePrinter.print("Initialising TFLite ...")

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Load the Tensorflow Lite model.
        interpreter = Interpreter(model_path=PATH_TO_CKPT)
        ConsolePrinter.print(f"Allocating tensors for {PATH_TO_CKPT} ...")
        interpreter.allocate_tensors()

        # Get and store/calculate TFLite model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5
        boxes_idx, classes_idx, scores_idx = 1, 3, 0 # This is a TF2 model

        # Initialize frame rate calculation
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()

        ConsolePrinter.print("Interpreter ready. Starting video stream ...")

        # Create window name
        window_name = f"Object Detector - {self.camera_name}"

        # Initialize video stream
        videostream = VideoStream(
            resolution=(self.resolution_w,self.resolution_h),
            video_url_or_filename=self.video_url_or_filename,
            framerate=self.framerate,
            video_file_analysis=self.video_file_analysis).start()

        ConsolePrinter.print("Streaming video at " 
              + str(self.resolution_w) 
              + "x" 
              + str(self.resolution_h))

        # Max streaming time
        time_to_stop_video = datetime.now() + timedelta(seconds=float(self.video_uptime))

        # FPS control
        target_fps = self.framerate
        milisecond_delay = 1000/target_fps
        fps_delay = datetime.now() + timedelta(milliseconds=milisecond_delay)

        # Motion detection variables
        first_frame = None
        frame_diff = None
        motion_contours = None

        while True:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Implement FPS delay
            if datetime.now() >= fps_delay:

                try:
                    # Grab frame from video stream
                    frame1 = videostream.read()
                    if type(frame1) is not None:

                        frame1 = cv2.resize(frame1, (self.resolution_w, self.resolution_h))

                        # Acquire frame and resize to expected shape [1xHxWx3]
                        frame = frame1.copy()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (width, height))
                        input_data = np.expand_dims(frame_resized, axis=0)

                        #### Motion Detection ####
                        # Thanks to https://towardsdatascience.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42
                        # Prepare image; grayscale and blur
                        prepared_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)

                        # Calculate difference and update previous frame
                        if (first_frame is None):
                            first_frame = prepared_frame
                        else:
                            frame_diff = cv2.absdiff(src1=first_frame, src2=prepared_frame) # calculate the dif
                            first_frame = prepared_frame # replace the first frame
                        
                            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
                            kernel = np.ones((5, 5))
                            frame_diff = cv2.dilate(frame_diff, kernel, 1)

                            # Only take different areas that are different enough (>10 / 255)
                            thresh_frame = cv2.threshold(src=frame_diff, thresh=25, maxval=255, type=cv2.THRESH_BINARY)[1]
                            if (self.debug_motion_tracking):
                                cv2.imshow("Motion Detection Analysis", thresh_frame)

                            # Use motion contours when an object is detected and correlate
                            motion_contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

                        #### Object Detection ####
                        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                        if floating_model:
                            input_data = (np.float32(input_data) - input_mean) / input_std

                        # Perform the actual detection by running the model with the image as input
                        interpreter.set_tensor(input_details[0]['index'],input_data)
                        interpreter.invoke()

                        # Retrieve detection results
                        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
                        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class indeqx of detected objects
                        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

                        # Loop over all detections and draw detection box if confidence is above minimum threshold
                        for i in range(len(scores)):
                            if ((scores[i] > self.threshold) and (scores[i] <= 1.0)):

                                # Get bounding box coordinates and draw box
                                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                                ymin = int(max(1,(boxes[i][0] * self.resolution_h)))
                                xmin = int(max(1,(boxes[i][1] * self.resolution_w)))
                                ymax = int(min(self.resolution_h,(boxes[i][2] * self.resolution_h)))
                                xmax = int(min(self.resolution_w,(boxes[i][3] * self.resolution_w)))
                                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                                # Draw label
                                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                                # Motion detection inside object detection - determine if it's near the object that's been detected.
                                # This improves object detection as it can mistakenly recognise stationary grass/leaves/patterns as animals
                                motion_near_object = False
                                search_hit = bool([ele for ele in constants.VIDEO_ANALYSIS_ANIMAL_FILTER if(ele in object_name)])
                                if search_hit:
                                    if motion_contours != None:
                                        for contour in motion_contours:
                                            (x, y, w, h) = cv2.boundingRect(contour)

                                            # Calculate difference between coords of detected object and detected motion
                                            ymin_calc = abs(ymin-y)
                                            xmin_calc = abs(xmin-x)
                                            ymax_calc = abs(ymax-(y+h))
                                            xmax_calc = abs(xmax-(x+w))

                                            # Motion detected if difference is no lower than threshold (20% of resolution width)
                                            threshold = self.resolution_w/20
                                            if ymin_calc <= threshold and xmin_calc <= threshold and ymax_calc <= threshold and xmax_calc <= threshold:
                                                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2) # blue
                                                motion_near_object = True

                                    if motion_near_object and self.disable_class_tracker is False:
                                        # Class tracker for writing images and MQTT comms if something is detected
                                        self.class_tracker.class_tracker(object_name,frame.copy(),scores[i],self.camera_name)


                        # Draw framerate in corner of frame
                        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                        # All the results have been drawn on the frame, so it's time to display it.
                        cv2.imshow(window_name, frame)

                        # Calculate framerate
                        t2 = cv2.getTickCount()
                        time1 = (t2-t1)/freq
                        frame_rate_calc= 1/time1

                        # Reset FPS delay
                        fps_delay = datetime.now() + timedelta(milliseconds=milisecond_delay)

                        # Press 'q' to quit
                        if cv2.waitKey(1) == ord('q'):
                            ConsolePrinter.print("Stream manually terminated")
                            break

                        if time_to_stop_video <= datetime.now():
                            ConsolePrinter.print("Stream terminated due to max time")
                            break

                except Exception as inst:
                    ConsolePrinter.print("Something went wrong... ")
                    ConsolePrinter.print(type(inst))    # the exception type
                    ConsolePrinter.print(inst.args)     # arguments stored in .args
                    ConsolePrinter.print(inst)

                    # Clean up and stop
                    cv2.destroyWindow(window_name)
                    videostream.stop()

        # Clean up
        cv2.destroyWindow(window_name)
        videostream.stop()