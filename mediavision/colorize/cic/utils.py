#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import warnings
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.color import rgb2lab, lab2rgb

from mediavision.core.types import AnyPath
from mediavision.core.processing_registry import register_processor
from mediavision.core.utils import load_image

def preprocess_image(original_image: np.ndarray,
                     size: tuple = (256, 256)) -> Tuple[torch.Tensor, torch.Tensor]:
   """Preprocesses an image for recoloring."""
   # Resize the image.
   resized_image = cv2.resize(original_image, dsize = size)

   # Convert the images to LAB-space.
   original_image = rgb2lab(original_image)
   resized_image = rgb2lab(resized_image)

   # Get the L-channel from the images.
   original_image = original_image[:, :, 0]
   resized_image = resized_image[:, :, 0]

   # Convert the images to tensors and return them.
   original_tensor = torch.Tensor(original_image)[None, None, :, :]
   resized_tensor = torch.Tensor(resized_image)[None, None, :, :]
   return original_tensor, resized_tensor

def postprocess_output(original_tensor: torch.Tensor,
                       output_tensor: torch.Tensor, mode: str = 'bilinear') -> np.ndarray:
   """Postprocesses an image after it passes through the model."""
   # Get the original and output shapes.
   original_shape = original_tensor.shape[2:]
   output_shape = output_tensor[2:]

   # Resize the output image as necessary.
   if original_shape != output_shape:
      output_tensor = F.interpolate(output_tensor, size = original_shape, mode = mode)

   # Concatenate the L and AB channels into a single image.
   output = torch.cat((original_tensor, output_tensor), dim = 1)

   # Convert the image from LAB-space to RGB and return it.
   output = lab2rgb(output.data.cpu().numpy()[0, ...].transpose((1, 2, 0))) * 255
   return output.astype(np.uint8)

@register_processor(name = 'cic_colorizer_image')
def cic_colorizer_image_processor(input_image: np.ndarray, model: nn.Module,
                                  output_path: AnyPath = None) -> np.ndarray:
   """Process and colorize an image through a cic model."""
   # Load in the image.
   image = load_image(input_image, grayscale = True)

   # Preprocess the image.
   original_image, resized_image = preprocess_image(image)

   # Get the model output.
   output = postprocess_output(original_image, model(resized_image).cpu())

   # Save the image to the output path if requested to.
   if output_path is not None:
      if not os.path.exists(output_path):
         raise FileNotFoundError(f"The provided save path {output_path} does not exist.")
      plt.imsave(output_path, output)

   # Return the output.
   return output

@register_processor(name = 'cic_colorizer_video')
def cic_colorizer_video_processor(input_path: AnyPath, model: nn.Module,
                                  output_path: AnyPath = None) -> None:
   """Processor, colorize, and save a video through a cic model."""
   # Capture all warnings (runaway PyTorch warnings, etc).
   warnings.filterwarnings('ignore')

   # Open the video file.
   cap = cv2.VideoCapture(input_path)

   # Construct the video writer.
   if output_path is not None:
      if os.path.exists(output_path):
         os.remove(output_path) # To prevent breakage.
      else:
         # Ensure that the output directory exists.
         if not os.path.exists(os.path.dirname(output_path)):
            raise NotADirectoryError(f"The directory of the output path "
                                     f"{output_path} does not exist.")
   else:
      # Otherwise, no output path has been provided, so we will
      # write the video to the same directory as the input video.
      output_path = os.path.join(os.path.splitext(input_path)[0] + "_colorized"
                                 + os.path.splitext(input_path)[1])

   fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
   videowriter = cv2.VideoWriter(
      output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True
   )

   # Define colors fro printing out status messages.
   LOG_HEAD = "[COLORIZATION]: "
   BLUE_C = "\033[94m"
   GREEN_C = "\033[92m"
   END_C = "\033[0m"

   # Get the number of frames and construct the progress bar.
   n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   print(BLUE_C + LOG_HEAD + "{} - {} frames to colorize.".format(
      os.path.basename(input_path), n_frames) + END_C)
   print(BLUE_C + LOG_HEAD + "Writing colorized video to {}".format(
      output_path), END_C)
   p_bar = tqdm(total = n_frames, desc = f"{LOG_HEAD}Colorizing Video",
                bar_format = "%s{l_bar}{bar}{r_bar}%s" % (BLUE_C, END_C))

   # Iterate over each frame in the video.
   while cap.isOpened():
      # Read a single frame.
      ret, frame = cap.read()

      # Ensure the video is not complete.
      if not ret:
         break

      # Load the frame.
      frame = load_image(frame, grayscale = True)

      # Preprocess the image.
      original_image, resized_image = preprocess_image(frame)

      # Get the model output.
      output = postprocess_output(original_image, model(resized_image).cpu())

      # Convert the output dtype and recolor it.
      output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)

      # Write the frame to the video writer.
      videowriter.write(output)

      # Update the progress bar.
      p_bar.update(1)

   # Close the video and video writer.
   cap.release()
   videowriter.release()
   p_bar.close()

   # Print a completion status message.
   print(GREEN_C + LOG_HEAD + "Video colorization complete." + END_C)

   # Re-enable warnings.
   warnings.resetwarnings()



