from __future__ import division, print_function
from collections import defaultdict
from glob2 import glob
import os


class ImageClass():
  '''
  Stores the paths to images for a given class
  '''
  def __init__(self, name, image_paths):
    self.name = name
    self.image_paths = image_paths

  def __str__(self):
    return self.name + ', ' + str(len(self.image_paths)) + ' images'

  def __len__(self):
    return len(self.image_paths)


def get_dataset(input_glob, nested=True):
  '''
  Return an ImageClass with all images in the input glob. If `nested` is True,
  return an ImageClass for each directory in the penultimate tree level position
  within that glob (e.g. a/b/1.jpg a/b/2.jpg would both have b class).
  '''
  l = []
  d = defaultdict(list) # d[directory] = list of files in that directory
  for i in glob(input_glob):
    path, filename = os.path.split(i)
    d[os.path.split(path)[-1] if nested else 'input_image'].append(i)
  for class_label in d:
    l.append(ImageClass(class_label, d[class_label]))
  return l


def to_rgb(im):
  '''
  Given an image in grayscale, return that image in rgb space
  '''
  w, h = img.shape
  pix = np.empty((w, h, 3), dtype=np.uint8)
  pix[:, :, 0] = pix[:, :, 1] = pix[:, :, 2] = img
  return pix