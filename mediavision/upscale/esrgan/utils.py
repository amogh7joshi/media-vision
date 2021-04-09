#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import cv2
import numpy as np

import torch
import torch.nn as nn

from mediavision.core.processing_registry import register_processor
from mediavision.core.types import ImageOrPath
from mediavision.core.utils import load_image

@register_processor(name = 'esrgan_upscaler')
def esrgan_upscaler_processor(input_image: np.ndarray, model: nn.Module) -> np.ndarray:
   """Process and upscale an image through the esrgan upscaler."""
   # Read, normalize, and transpose the input image.
   image = load_image(input_image)
   image * 1.0 / 255.
   # image = np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))

   # Convert the image to a tensor.
   image = torch.from_numpy(image).float()
   image = image.unsqueeze(0)
   image.to(torch.device('cpu'))

   # Pass the image through the model.
   with torch.no_grad():
      output = model(image).data.squeeze().float().cpu().clamp_(0, 1).numpy()

   # Get the output image.
   # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

   # Return the output.
   return output


