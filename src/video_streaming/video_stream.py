# Modified based on VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

# Import packages
import os
import cv2
import threading
from datetime import datetime, timedelta
from console.logger import ConsolePrinter

class VideoStream:

    """Camera object that controls video streaming"""
    def __init__(self,resolution=(640,480),video_url_or_filename="",framerate=30,video_file_analysis=False):

        self.framerate = framerate
        self.video_file_analysis = video_file_analysis

        ConsolePrinter.print(f"Video streaming activated at {framerate} FPS")

	    # Variable to control when the camera is stopped
        self.stopped = False

        ConsolePrinter.print(f"Gathering video from {video_url_or_filename}")

        if video_file_analysis != True:
            # Initialize the RTSP image stream
            ConsolePrinter.print("Initialising video stream ...")
            self.stream = cv2.VideoCapture(video_url_or_filename, cv2.CAP_FFMPEG)
        
        elif video_file_analysis == True:
            # Initialize the local video
            ConsolePrinter.print("Initialising local file analysis")
            self.stream = cv2.VideoCapture(video_url_or_filename)
        
        self.stream.set(3,resolution[0])
        self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()


    def start(self):
	# Start the thread that reads frames from the video streamx
        thread = threading.Thread(target=self.update,args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):

        target_fps = self.framerate
        milisecond_delay = 1000/target_fps
        fps_delay = datetime.now() + timedelta(milliseconds=milisecond_delay)

        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return
            else:
                
                # Limit FPS
                if datetime.now() >= fps_delay:

                    # Grab the next frame from the stream
                    (self.grabbed, self.frame) = self.stream.read()

                    # Reset FPS delay
                    fps_delay = datetime.now() + timedelta(milliseconds=milisecond_delay)
                
                else:
                    # Ensure unused frames from the queue are cleared out
                    if self.video_file_analysis != True:
                        self.stream.read()
    
    def read(self):
        return self.frame

    def stop(self):
	    # Indicate that the camera and thread should be stopped
        self.stopped = True
