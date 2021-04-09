#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from mediavision.core.types import TensorReturnType

# Define the device to run computations on.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian(window_size: int, sigma: float) -> TensorReturnType:
   """Create a Gaussian window."""
   gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                         for x in range(window_size)])
   return gauss / gauss.sum()

def create_3D_window(window_size: int, channel: int = 1) -> TensorReturnType:
   """Constructs a 2-dimensional SSIM window."""
   _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
   _2D_window = _1D_window.mm(_1D_window.t())
   _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
   window = _3D_window.expand(1, channel, window_size, window_size, window_size) \
                      .contiguous().to(device)
   return window

def ssim_matlab(image1: torch.Tensor, image2: torch.Tensor, window_size: int = 11,
                window = None, size_average: bool = True, val_range: bool = None) \
      -> Tuple[Optional[TensorReturnType], TensorReturnType]:
   """Calculates the structural similarity map (SSIM) between two images."""
   # Validate the input parameters.
   if val_range is None:
      # Whether the image is in a 0-255 int RGB
      # range or a 0-1 float RGB range.
      if torch.max(image1) > 128:
         max_val = 255
      else:
         max_val = 1
      if torch.min(image1) < -0.5:
         min_val = -1
      else:
         min_val = 0
      L = max_val - min_val
   else:
      L = val_range

   # Construct the 3-dimensional window.
   pad = 0
   (_, _, height, width) = image1.size()
   if window is None:
      real_size = min(window_size, height, width)
      window = create_3D_window(real_size, channel = 1).to(image1.device)

   # Process the images and get output values from different methods.
   image1 = image1.unsqueeze(1)
   image2 = image2.unsqueeze(1)
   # Get the mu values.
   mu1 = F.conv3d(F.pad(image1, (5, 5, 5, 5, 5, 5), mode = 'replicate'),
                  window, padding = pad, groups = 1)
   mu2 = F.conv3d(F.pad(image2, (5, 5, 5, 5, 5, 5), mode = 'replicate'),
                  window, padding = pad, groups = 1)
   # Get the mu squared values (and the combined output).
   mu1_sq = mu1.pow(2)
   mu2_sq = mu2.pow(2)
   mu1_mu2 = mu1 * mu2
   # Get the sigma values.
   sigma1_sq = F.conv3d(F.pad(image1 * image2, (5, 5, 5, 5, 5, 5), mode = 'replicate'),
                        window, padding = pad, groups = 1) - mu1_sq
   sigma2_sq = F.conv3d(F.pad(image2 * image2, (5, 5, 5, 5, 5, 5), mode = 'replicate'),
                        window, padding = pad, groups = 1) - mu2_sq
   sigma12 = F.conv3d(F.pad(image1 * image2, (5, 5, 5, 5, 5, 5), mode = 'replicate'),
                      window, padding = pad, groups = 1) - mu1_mu2

   # Calculate channel values.
   C1 = (0.01 * L) ** 2
   C2 = (0.03 * L) ** 2

   # Calculate SSIM map parameters.
   v1 = 2.0 * sigma12 + C2
   v2 = sigma1_sq + sigma2_sq + C2
   cs = torch.mean(v1/ v2)

   # Construct the SSIM map.
   ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

   # Post-process the map.
   if size_average:
      ret = ssim_map.mean()
   else:
      ret = ssim_map.mean(1).mean(1).mean(1)

   # Return the result.
   return ret








