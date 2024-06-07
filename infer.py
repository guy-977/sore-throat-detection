import tensorflow as tf
import numpy as np
import argparse

def get_prediction(img, model):
  class_names = [
    'no pharyngitis', 'pharyngitis'
  ]
  img = tf.keras.utils.load_img(img, target_size=(512, 512))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  prediction = model.predict(img_array)
  score = tf.nn.softmax(prediction)

  return [class_names[np.argmax(score)], 100 * np.max(score)]

model = tf.keras.models.load_model('models/model.h5')

## arguments parser to get input for the user's arguments before excuting the program
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', action='store', help='The image path', type=str)
args = parser.parse_args()

img_path = args.image

try:
    if img_path is not None:
        pass
    else:
        input('You must insert image path: ')
    result = get_prediction(img_path, model)

    print("Detection result is:\n", result)
except Exception as e:
    print('Something went wrong!')
    raise(e)
finally:
    print('Inference done successfully!')