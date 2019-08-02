'''Performs face alignment and stores face thumbnails in the output directory.'''
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import random
import copy
import sys
import os

# internal
from facenet.utils import utils
from facenet.align import detect_face


# default args for cli
defaults = {
  'input_glob': None,
  'inputs_nested': False,
  'output_dir': 'cropped',
  'image_size': 182,
  'margin': 44,
  'gpu_memory_fraction': 1.0,
  'detect_multiple_faces': True,
  'random_order': True,
}


def validate_args(arg_d):
  '''
  Validate that the user passed input args properly
  '''
  if not arg_d.get('input_glob'):
    raise Exception('Please provide an input_glob named argument to crop_faces')


def crop_faces(arg_d):
  '''
  Given a dict of args, crop all user inputs
  '''
  args = copy.deepcopy(defaults)
  args.update(arg_d)
  validate_args(args)
  output_dir = os.path.expanduser(args.get('output_dir'))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  dataset = utils.get_dataset(args.get('input_glob'))
  print('Creating networks and loading parameters')
  with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.get('gpu_memory_fraction'))
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
  factor = 0.709 # scale factor

  # Add a random key to the filename to allow alignment using multiple processes
  random_key = np.random.randint(0, 2**32)
  bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

  with open(bounding_boxes_filename, 'w') as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if args.get('random_order'):
      random.shuffle(dataset)
    for cls in dataset:
      output_class_dir = os.path.join(output_dir, cls.name)
      if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
        if args.get('random_order'):
          random.shuffle(cls.image_paths)
      for image_path in cls.image_paths:
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0]
        output_filename = os.path.join(output_class_dir, filename + '.png')
        print(image_path)
        if not os.path.exists(output_filename):
          try:
            img = misc.imread(image_path)
          except (IOError, ValueError, IndexError) as e:
            print('{}: {}'.format(image_path, e))
          else:
            if img.ndim<2:
              print('Unable to align {0}'.format(image_path))
              text_file.write('%s\n' % (output_filename))
              continue
            if img.ndim == 2:
              img = utils.to_rgb(img)
            img = img[:,:,0:3]

            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces>0:
              det = bounding_boxes[:,0:4]
              det_arr = []
              img_size = np.asarray(img.shape)[0:2]
              if nrof_faces>1:
                if args.get('detect_multiple_faces'):
                  for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
                else:
                  bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                  img_center = img_size / 2
                  offsets = np.vstack([
                    (det[:,0]+det[:,2])/2-img_center[1],
                    (det[:,1]+det[:,3])/2-img_center[0],
                  ])
                  offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                  index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                  det_arr.append(det[index,:])
              else:
                det_arr.append(np.squeeze(det))

              for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-args.get('margin')/2, 0)
                bb[1] = np.maximum(det[1]-args.get('margin')/2, 0)
                bb[2] = np.minimum(det[2]+args.get('margin')/2, img_size[1])
                bb[3] = np.minimum(det[3]+args.get('margin')/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                size = (args.get('image_size'), args.get('image_size'))
                scaled = misc.imresize(cropped, size, interp='bilinear')
                nrof_successfully_aligned += 1
                filename_base, file_extension = os.path.splitext(output_filename)
                if args.get('detect_multiple_faces'):
                  output_filename_n = '{}_{}{}'.format(filename_base, i, file_extension)
                else:
                  output_filename_n = '{}{}'.format(filename_base, file_extension)
                misc.imsave(output_filename_n, scaled)
                text_file.write('{0} {1} {2} {3} {4}\n'.format(output_filename_n, bb[0], bb[1], bb[2], bb[3]))
            else:
              print('Unable to align {0}'.format(image_path))
              text_file.write('{0}\n'.format(output_filename))

  print('Total number of images: {0}'.format(nrof_images_total))
  print('Number of successfully aligned images: {0}'.format(nrof_successfully_aligned))


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('input_glob', type=str,
    help='A glob to a collection of input files. E.g. "cats/*.jpg"')
  parser.add_argument('--inputs_nested', type=bool, default=defaults.get('inputs_nested'),
    help='Boolean indicating whether the files in `input_glob` are nested')
  parser.add_argument('--output_dir', type=str, default=defaults.get('cropped'),
    help='Directory with aligned face thumbnails.')
  parser.add_argument('--image_size', type=int, default=defaults.get('image_size'),
    help='Image size (height, width) in pixels.')
  parser.add_argument('--margin', type=int, default=defaults.get('margin'),
    help='Margin for the crop around the bounding box (height, width) in pixels.')
  parser.add_argument('--gpu_memory_fraction', type=float, default=defaults.get('gpu_memory_fraction'),
    help='Max GPU memory used by the process.')
  parser.add_argument('--detect_multiple_faces', type=bool, default=defaults.get('detect_multiple_faces'),
    help='Detect and align multiple faces per image.')
  parser.add_argument('--random_order', type=bool, default=defaults.get('random_order'),
    help='Shuffles the order of images to enable alignment using multiple processes.')
  return vars(parser.parse_args(argv))

if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:]) # args is a dict
  crop_faces(args)
