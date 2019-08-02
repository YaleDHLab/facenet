from setuptools import setup

setup (
  name='yale-dhlab-facenet',
  version='0.0.1',
  packages=['facenet', 'facenet.utils', 'facenet.align'],
  package_data={
    'facenet': [
      'align/model/*',
    ]
  },
  keywords = ['machine-learning', 'tensorflow', 'facenet', 'computer-vision'],
  description='Python implementation of the facenet model',
  url='https://github.com/yaledhlab/yale-facenet',
  author='David Sandberg, packaged by Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'glob2>=0.6',
    'numpy>=1.16.4',
    'scipy==1.1.0',
    'six>=1.11.0',
    'tensorflow==1.7',
  ],
)