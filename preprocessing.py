from PIL import Image
import numpy as np
from skimage import transform


def load(filename):
   # np_image = Image.open(filename)
   np_image = np.array(filename).astype('float32')/224
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image