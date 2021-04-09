#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import itertools
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mediavision.core.types import TensorReturnType

# Define the device to run computations on.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_filters: int,
               out_filters: int,
               *,
               kernel_size: Union[int, tuple] = (3, 3),
               stride: int = 1,
               padding: int = 1,
               dilation: int = 1):
   """A generic convolution block."""
   return nn.Sequential(
      nn.Conv2d(in_filters, out_filters, kernel_size = kernel_size,
                stride = stride, padding = padding, dilation = dilation, bias = True),
      nn.PReLU(out_filters)
   )

class Conv2(nn.Module):
   """Two convolution blocks."""
   def __init__(self, in_filters: int, out_filters: int, stride: int = 2):
      # Initialize the module.
      super(Conv2, self).__init__()

      # Construct the convolution blocks.
      self.conv1 = conv_block(in_filters, out_filters, kernel_size = (3, 3),
                              stride = stride, padding = 1)
      self.conv2 = conv_block(out_filters, out_filters, kernel_size = (3, 3),
                              stride = 1, padding = 1)

   def forward(self, x: torch.Tensor) -> TensorReturnType:
      # Sequential pass through the convolution blocks.
      x = self.conv1(x)
      x = self.conv2(x)
      return x

def upsample_block(in_filters: int,
                   out_filters: int,
                   *,
                   kernel_size: Union[tuple, int] = (4, 4),
                   stride: int = 2,
                   padding: int = 1):
   """A generic upsampling block."""
   return nn.Sequential(
      nn.ConvTranspose2d(in_filters, out_filters, kernel_size = kernel_size,
                         stride = stride, padding = padding, bias = True),
      nn.PReLU(out_filters)
   )

# Maintain a log of already-conducted warps.
warp_grid = {}

def warp(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> TensorReturnType:
   """Spatially warp an input tensor."""
   # Gather the tensor parameters.
   k = (str(input_tensor.device), str(input_tensor.size()))

   # Calculate the warp parameters.
   if k not in warp_grid:
      horizontal = torch.linspace(-1.0, 1.0, output_tensor.shape[3], device = device) \
                        .view(1, 1, 1, output_tensor.shape[3]) \
                        .expand(output_tensor.shape[0], -1, output_tensor.shape[2], -1)
      vertical = torch.linspace(-1.0, 1.0, output_tensor.shape[2], device = device) \
                      .view(1, 1, output_tensor.shape[2], 1) \
                      .expand(output_tensor.shape[0], -1, -1, output_tensor.shape[3])
      # Add the calculations to the grid.
      warp_grid[k] = torch.cat((horizontal, vertical), 1).to(device)

   # Re-calculate the output tensor.
   output_tensor = torch.cat((output_tensor[:, 0:1, :, :] / ((input_tensor.shape[3] - 1.0) / 2.0),
                              output_tensor[:, 1:2, :, :] / ((input_tensor.shape[2] - 1.0) / 2.0)), 1)

   # Calculate the parameter to which to warp the grid.
   g = (warp_grid[k] + output_tensor).permute(0, 2, 3, 1)

   # Return a grid sample of the output tensor.
   return F.grid_sample(input_tensor, g, mode = 'bilinear',
                        padding_mode = 'border', align_corners = True)

class IFBlock(nn.Module):
   """The IFNet core block."""
   def __init__(self, in_filters: int, scale: int = 1, out_filters: int = 64):
      # Initialize the module.
      super(IFBlock, self).__init__()

      # Validate and set the input parameters.
      self.scale = scale

      # Construct the convolution layers of the class.
      self.conv0 = nn.Sequential(
         conv_block(in_filters, out_filters, kernel_size = (3, 3),
                    stride = 2, padding = 1),
         conv_block(out_filters, 2 * out_filters, kernel_size = (3, 3),
                    stride = 2, padding = 1))
      self.convblock = nn.Sequential(
         conv_block(2 * out_filters, 2 * out_filters),
         conv_block(2 * out_filters, 2 * out_filters),
         conv_block(2 * out_filters, 2 * out_filters),
         conv_block(2 * out_filters, 2 * out_filters),
         conv_block(2 * out_filters, 2 * out_filters),
         conv_block(2 * out_filters, 2 * out_filters))
      self.conv1 = nn.ConvTranspose2d(
         2 * out_filters, 4, kernel_size = (4, 4), stride = 2, padding = 1)

   def forward(self, x: torch.Tensor) -> TensorReturnType:
      # Interpolate the tensor if necessary.
      if self.scale != 1:
         x = F.interpolate(x, scale_factor = 1. / self.scale,
                           mode = 'bilinear', align_corners = False)

      # Pass through the convolution blocks.
      x = self.conv0(x)
      x = self.convblock(x)
      x = self.conv1(x)

      # Interpolate the tensor if necessary.
      flow = x
      if self.scale != 1:
         flow = F.interpolate(flow, scale_factor = self.scale,
                              mode = 'bilinear', align_corners = False)

      # Return the tensor.
      return flow



