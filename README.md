# Facenet

A packaged version of David Sandberg's [facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), based on the paper [FaceNet](https://arxiv.org/abs/1503.03832) (2015).

## Installation

```bash
pip install yale-dhlab-facenet
```

## Usage

To crop all faces in `data/*.jpg`, one can run:

```python
from facenet.align import crop_faces

crop_faces({'input_glob': 'data/*.jpg'})
```

Extracted faces will be written to `./cropped`