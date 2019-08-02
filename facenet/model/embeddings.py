'''Get high-dimensional embeddings for a glob of face image inputs'''

import tensorflow as tf
import numpy as np
import requests
import tarfile
import random
import shutil
import copy
import os

# internal
from facenet.model import facenet_model
from facenet.utils import utils


# default args
defaults = {
  'input_glob': None,
  'output_dir': 'facenet_vectors',
  'img_size': 160,
}


def validate_args(args_d):
  '''
  Validate that the user has passed in required arguments
  '''
  if not args_d or not args_d.get('input_glob'):
    raise Exception('''Please provide an input_glob argument to get_embeddings
      e.g. get_embeddings({input_glob: "cropped/data/*.jpg"})''')


def get_embeddings(arg_d):
  '''
  Get a vector representation of each image in args_d['input_glob']

  Usage: get_embeddings({'input_glob': 'cropped/data/*.jpg'})
  '''
  validate_args(arg_d)
  args = copy.deepcopy(defaults)
  args.update(arg_d)
  if not os.path.exists(args.get('output_dir')):
    os.makedirs(args.get('output_dir'))
  with tf.Graph().as_default():
    with tf.Session() as sess:
      # Load the model if the user had one in mind, else load a default model
      model_path = get_model_path()
      print(' * Loading pretrained model')
      facenet_model.load_model(model_path)
      # Get input and output tensors
      images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
      embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
      phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
      # Get the input images
      input_image_classes = utils.get_dataset(args.get('input_glob'))
      for idx, i in enumerate(input_image_classes):
        print(' * Processing image class', idx+1, 'of', len(input_image_classes), '-', i)
        for jdx, j in enumerate(i.image_paths):
          print(' * Processing image', jdx+1, 'of', len(i.image_paths), '-', j)
          batch_size = 1 # sgd!
          img_data = facenet_model.load_data([j], False, False, args.get('img_size'))
          d = { images_placeholder: img_data, phase_train_placeholder: False }
          arr = sess.run(embeddings, feed_dict=d)
          # save the numpy array
          out_path = os.path.join(args.get('output_dir'), os.path.basename(j))
          np.save(out_path, arr)


def get_model_path():
  '''
  Download a pretrained facenet model to the users cwd and return the path to the model
  '''
  #current_dir = os.path.dirname(os.path.realpath(__file__))
  current_dir = os.getcwd()
  tmp_dir = os.path.join(current_dir, 'tmp')
  download_path = os.path.join(tmp_dir, 'pretrained-facenet.tar.gz')
  model_dir = os.path.join(tmp_dir, 'model')
  model_id = '20180402-114759'
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    os.makedirs(model_dir)
    # specify the model to download
    model_name = 'facenet-pretrained-{0}.tar.gz'.format(model_id)
    # download the model
    url = 'https://lab-apps.s3-us-west-2.amazonaws.com/facenet/' + model_name
    print(' * downloading pretrained model from', url)
    data = requests.get(url, allow_redirects=True).content
    open(download_path, 'wb').write(data)
    # unpack the model
    print(' * extracting model')
    tar = tarfile.open(download_path, 'r:gz')
    tar.extractall(path=model_dir)
    tar.close()
  return os.path.join(model_dir, model_id)