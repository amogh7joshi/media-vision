#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from mediavision.core.types import TensorReturnType

class ResidualDenseBlockSC(nn.Module):
   """A module representing an SC residual dense block."""
   def __init__(self, in_filters: int = 64,
                out_filters: int = 32, bias: bool = True):
      # Initialize the module.
      super(ResidualDenseBlockSC, self).__init__()

      # Construct the convolution layers of the block.
      self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size = (3, 3),
                             stride = 1, padding = 1, bias = bias)
      self.conv2 = nn.Conv2d(in_filters + out_filters, out_filters,
                             kernel_size = (3, 3),
                             stride = 1, padding = 1, bias = bias)
      self.conv3 = nn.Conv2d(in_filters + 2 * out_filters, out_filters,
                             kernel_size = (3, 3),
                             stride = 1, padding = 1, bias = bias)
      self.conv4 = nn.Conv2d(in_filters + 3 * out_filters, out_filters,
                             kernel_size = (3, 3),
                             stride = 1, padding = 1, bias = bias)
      self.conv5 = nn.Conv2d(in_filters + 4 * out_filters, in_filters,
                             kernel_size = (3, 3),
                             stride = 1, padding = 1, bias = bias)

      # Construct the output leaky relu layer.
      self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

   def forward(self, x: torch.Tensor) -> TensorReturnType:
      # Create the residual dense structure of the layers.
      x1 = self.lrelu(self.conv1(x))
      x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
      x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
      x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
      x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
      return x5 * 0.2 + x

class RRDB(nn.Module):
   """A module representing a residual-in-residual dense block."""
   def __init__(self, in_filters: int, out_filters: int = 32):
      # Initialize the module.
      super(RRDB, self).__init__()

      # Construct the residual blocks.
      self.RDB1 = ResidualDenseBlockSC(in_filters, out_filters)
      self.RDB2 = ResidualDenseBlockSC(in_filters, out_filters)
      self.RDB3 = ResidualDenseBlockSC(in_filters, out_filters)

   def forward(self, x: torch.Tensor) -> TensorReturnType:
      # Pass through the residual dense blocks.
      out = self.RDB1(x)
      out = self.RDB2(out)
      out = self.RDB3(out)
      return out * 0.2 + x


