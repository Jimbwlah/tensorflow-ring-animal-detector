# Python script to train a new model

# Import packages
import os
import re
import sys
sys.path.append(os.path.join(os.getcwd(),'src'))
import constants

# SSD MobileNet V2 FPNLite has been chosen as a reliable and fast model
# Change the chosen_model variable to deploy different models available in the TF2 object detection zoo
# See TensorFlow 2 Object Detection Model Zoo - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

# Set training parameters for the model
num_steps = constants.TRAINING_NUM_STEPS
batch_size = constants.TRAINING_BATCH_SIZE

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

def main():

    train_record_fname = os.path.join(os.getcwd(),'training-content/train.tfrecord').replace("\\", "/")
    val_record_fname = os.path.join(os.getcwd(),'training-content/val.tfrecord').replace("\\", "/")
    label_map_pbtxt_fname = os.path.join(os.getcwd(),'training-content/labelmap.pbtxt').replace("\\", "/")

    # Set file locations and get number of classes for config file
    model_dir = os.path.join(os.getcwd(),'training-content/training/').replace("\\", "/")
    base_pipeline_fname = os.path.join(os.getcwd(),f'src/models/model_zoo/{constants.TRAINING_MODEL_FOLDER_NAME}/pipeline.config').replace("\\", "/")
    fine_tune_checkpoint = os.path.join(os.getcwd(),f'src/models/model_zoo/{constants.TRAINING_MODEL_FOLDER_NAME}/checkpoint/ckpt-0').replace("\\", "/")
    pipeline_fname = os.path.join(os.getcwd(),'training-content/pipeline.config').replace("\\", "/")

    num_classes = get_num_classes(label_map_pbtxt_fname)
    print('Getting classes - total classes:', num_classes)

    print('Writing custom configuration file ...')

    print(base_pipeline_fname)
    print(train_record_fname)
    
    with open(base_pipeline_fname) as f:
        s = f.read()
    with open(pipeline_fname, 'w') as f:

        # Set fine_tune_checkpoint path
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # Set tfrecord files for train and test datasets
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(val_record_fname), s)

        # Set label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set batch_size
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)

        # Set number of classes num_classes
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), s)

        # Change fine-tune checkpoint type from "classification" to "detection"
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

        # If using ssd-mobilenet-v2, reduce learning rate (because it's too high in the default config file)
        if "ssd-mobilenet" in constants.TRAINING_MODEL_FOLDER_NAME:
            s = re.sub('learning_rate_base: .8',
                        'learning_rate_base: .08', s)

            s = re.sub('warmup_learning_rate: 0.13333',
                        'warmup_learning_rate: .026666', s)
        
        if "efficientdet" in constants.TRAINING_MODEL_FOLDER_NAME:
            s = re.sub('keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
            s = re.sub('pad_to_max_dimension: true', '', s)
            s = re.sub('min_dimension', 'height', s)
            s = re.sub('max_dimension', 'width', s)

        f.write(s)
        print(" done")
        print("")

if __name__ == '__main__':
  main()

