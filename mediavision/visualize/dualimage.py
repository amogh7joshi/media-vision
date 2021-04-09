#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

from mediavision.core.types import ImageOrPath

def show_input_output(input_image: ImageOrPath, output_image: ImageOrPath,
                      plot_titles: bool = True,
                      plot_size: Union[list, tuple] = (8, 4),
                      **kwargs) -> plt.Figure:
   """Displays an input image alongside its output image in a matplotlib plot."""
   # The images might be either actual images or paths, so first
   # determine the situation of each and load them in as necessary.
   if isinstance(input_image, (str, bytes, os.PathLike)):
      # Validate the path and load in the image.
      if not os.path.exists(input_image):
         raise FileNotFoundError(f"The provided path {input_image} does not exist.")
      input_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2GRAY)
   else:
      input_image = input_image
   if isinstance(output_image, (str, bytes, os.PathLike)):
      # Validate the path and load in the image.
      if not os.path.exists(output_image):
         raise FileNotFoundError(f"The provided path {output_image} does not exist.")
      output_image = cv2.cvtColor(cv2.imread(output_image), cv2.COLOR_BGR2GRAY)
   else:
      output_image = output_image

   # Construct the visualization.
   fig, axes = plt.subplots(1, 2, figsize = plot_size)

   # Plot the images onto the axes.
   images = [input_image, output_image]
   for indx, ax in enumerate(axes.flat):
      # Display the image.
      ax.imshow(images[indx], cmap = 'gray')
      # Format the plot.
      if plot_titles:
         if indx == 0:
            if "fontdict" in kwargs:
               ax.set_title("Input", fontdict = kwargs["fontdict"])
            else:
               ax.set_title("Input", fontsize = 12)
         else:
            if "fontdict" in kwargs:
               ax.set_title("Output", fontdict = kwargs["fontdict"])
            else:
               ax.set_title("Output", fontsize = 12)
      ax.axis('off')

   # Display the visualization.
   plt.show()

   # Return the figure (for Jupyter notebooks, etc).
   return plt.gcf()


