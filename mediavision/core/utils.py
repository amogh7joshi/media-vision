#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np

from mediavision.core.types import ImageOrPath, AnyPath

def load_image(path: ImageOrPath, grayscale: bool = False) -> np.ndarray:
   """Loads an input image and performs preliminary preprocessing."""
   # Load in the image.
   if isinstance(path, np.ndarray):
      # The image has already been loaded in.
      input_image = path
   else:
      # Otherwise, it is a path, so validate it then load in the image.
      if not os.path.exists(path):
         raise FileNotFoundError(f"The provided image path {path} does not exist.")
      input_image = cv2.imread(path)

   # Recolor the image if necessary.
   if grayscale:
      try:
         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
      except cv2.error: # The image already has only one channel.
         pass

   # Check whether the number of output dimensions is correct.
   if input_image.ndim == 2:
      input_image = np.tile(input_image[:, :, None], 3)

   # Return the loaded image.
   return input_image

def is_video_path(path: AnyPath) -> bool:
   """Determines if an input path is to a video or not."""
   if isinstance(path, (str, bytes, os.PathLike)):
      if path.endswith('.mov') or path.endswith('.mp4'):
         return True
   return False
