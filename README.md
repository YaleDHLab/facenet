# Facenet

![visualization of faces](./images/faces.png)

Crop and vectorize faces from input images. Packaged from David Sandberg's [facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), based on the paper [FaceNet](https://arxiv.org/abs/1503.03832) (2015).

## Installation

```bash
pip install yale-dhlab-facenet
```

## Crop Faces

To crop all faces in `data/*.jpg`, one can run:

```python
from facenet.crop import crop_faces

crop_faces({'input_glob': 'data/*.png'})
```

Extracted faces will be written to `./cropped`

## Extract Face Embeddings

To obtain FaceNet embeddings for each face image in a directory, one can run:

```python
from facenet.model import get_embeddings

get_embeddings({'input_glob': 'cropped/data/*.png'})
```

This process will write create a directory `facenet_vectors`, and will write one numpy array to that directory for each image in `input_glob`.