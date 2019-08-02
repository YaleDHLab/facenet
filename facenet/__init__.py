# filter future warnings from legacy tensorflow
import warnings
warnings.filterwarnings(action='ignore')

import facenet.utils
import facenet.crop
import facenet.model