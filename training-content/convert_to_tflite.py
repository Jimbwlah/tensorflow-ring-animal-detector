import tensorflow as tf
import os
import constants

converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(os.getcwd(),f'{constants.MODEL_LOCATION}/saved_model'))
tflite_model = converter.convert()

with open(os.path.join(os.getcwd(),f'{constants.MODEL_LOCATION}/{constants.DETECT_TF_FILENAME}'), 'wb') as f:
  f.write(tflite_model)