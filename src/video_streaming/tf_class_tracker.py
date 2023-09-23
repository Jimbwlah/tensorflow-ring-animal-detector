import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from console.logger import ConsolePrinter
from furbinator.furbinator_activator import FurbinatorActivator
import constants

class ClassTracker:

    """
    Class score tracker for TensorFlow Class Detection
    """
    def __init__(self,mqtt_client: mqtt.Client):

        # Images and dictionary to hold detected classes and times
        self.snapshot_delay_seconds = constants.IMAGE_SNAPSHOT_TIME_BETWEEN_SNAPSHOTS

        self.fox_hit_count = 0
        self.fox_hit_accuracy = []
        self.badger_hit_count = 0
        self.badger_hit_accuracy = []

        self.classes = {}
        self.furbinator = FurbinatorActivator(mqtt_client)

    def class_tracker(self,object_name,frame,score,camera_name):
        """
        Class tracker for jpg dump / logging of detections
        """
        class_exists = self.classes.get(object_name)

        score = score * 100
        score = round(score,2)

        now = datetime.now()

        search_hit = bool([ele for ele in constants.VIDEO_ANALYSIS_ANIMAL_FILTER if(ele in object_name)])
        if search_hit:

            ConsolePrinter.print(f"HIT - {object_name} at {score}%")
            if object_name == "fox":
                self.fox_hit_count = self.fox_hit_count + 1
                self.fox_hit_accuracy.append(score)
                self.get_final_class_scores(object_name)
            if object_name == "badger":
                self.badger_hit_count = self.badger_hit_count + 1
                self.badger_hit_accuracy.append(score)
                self.get_final_class_scores(object_name)
            
            # Snapshot delay - if accurate then this will be hit again so we set a cooldown (snapshot_delay_seconds)
            if class_exists is None:

                # First hit, record in class tracker
                time_found = now.strftime("%H:%M:%S")
                self.classes[object_name] = now

            if class_exists is not None:
                # Second hit, has enough time passed?
                time_found = self.classes[object_name]
                time_difference = time_found + timedelta(seconds=self.snapshot_delay_seconds)
                if time_difference < now:
                    
                    # Enough time has passed since last detection, object has qualified. SCARE IT AND PHOTO!
                    ConsolePrinter.print(f"{object_name} qualified at {time_found}, photoing and scaring!")
                    self.furbinator.activate_furbinator(camera_name)
                    self.furbinator.snapshot_target(frame,object_name,now)
                    self.classes.pop(object_name) # Remove key to start this again
                    

            
    def get_final_class_scores(self,object_name):
        """
        Returns the running average on accuracy scores for given class objects
        """
        badger_accuracy_calc = 0
        if(object_name == "badger"):
            if len(self.badger_hit_accuracy):
                for badger_score in self.badger_hit_accuracy:
                    badger_accuracy_calc += badger_score

                final_badger_score = badger_accuracy_calc/self.badger_hit_count
                ConsolePrinter.print(f"Badger accuracy = {final_badger_score}")

        fox_accuracy_calc = 0
        if(object_name == "fox"):
            if len(self.fox_hit_accuracy):
                for fox_score in self.fox_hit_accuracy:
                    fox_accuracy_calc += fox_score

                final_fox_score = fox_accuracy_calc/self.fox_hit_count
                ConsolePrinter.print(f"Fox accuracy = {final_fox_score}")