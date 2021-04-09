#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from mediavision.upscale.esrgan.modules import RRDB
from mediavision.core.types import TensorReturnType
from mediavision.core.model_registry import register_upscaler
from mediavision.core.base import MediaVisionModelBase
from mediavision.weights import load_model_weights


class RRDBNet(nn.Module, MediaVisionModelBase):
   """Implementation of the RRDBNet model.

   Sourced from: https://github.com/xinntao/ESRGAN/, which
   itself is an implementation of the paper "Enhanced super-resolution
   generative adversarial networks (see https://arxiv.org/abs/1809.00219).
   """
   def __init__(self, in_filters: int, out_filters: int, final_filters: int,
                n_layers: int, growth_filters: int):
      # Initialize a module.
      super(RRDBNet, self).__init__()

      # Create the starting convolution layer.
      self.conv_first = nn.Conv2d(in_filters, out_filters, kernel_size = (3, 3),
                                  stride = 1, padding = 1, bias = True)

      # Create the middle trunk of convolution layers.
      self.RRDB_trunk = nn.Sequential(
         *(
            functools.partial(RRDB, out_filters, growth_filters)() for _ in range(n_layers)
         )
      )
      self.trunk_conv = nn.Conv2d(out_filters, out_filters, kernel_size = (3, 3),
                                  stride = 1, padding = 1, bias = True)

      # Construct the upsampling layers.
      self.upconv1 = nn.Conv2d(out_filters, out_filters, kernel_size = (3, 3),
                               stride = 1, padding = 1, bias = True)
      self.upconv2 = nn.Conv2d(out_filters, out_filters, kernel_size = (3, 3),
                               stride = 1, padding = 1, bias = True)
      self.HRconv = nn.Conv2d(out_filters, out_filters, kernel_size = (3, 3),
                              stride = 1, padding = 1, bias = True)
      self.conv_last = nn.Conv2d(out_filters, final_filters, kernel_size = (3, 3),
                                 stride = 1, padding = 1, bias = True)

      # Construct the output leaky relu layer.
      self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

   def forward(self, x: torch.Tensor) -> TensorReturnType:
      # Pass sequentially through the input convolution layers and stem.
      fea = self.conv_first(x)
      trunk = self.trunk_conv(self.RRDB_trunk(fea))
      fea = fea + trunk

      # Pass through the upsampling layers and get the model output.
      fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor = 2, mode = 'nearest')))
      fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor = 2, mode = 'nearest')))
      out = self.conv_last(self.lrelu(self.HRconv(fea)))

      # Return the model output.
      return out

@register_upscaler(name = 'esrgan_rrdbnet')
def load_esrgan_rrdbnet_model(pretrained = True) -> RRDBNet:
   """Constructs the ESRGAN RRDBNet upscaler and loads pretrained weights."""
   # Construct the model.
   _model = RRDBNet(3, 64, 3, 23, 32)

   # Load pretrained weights if requested to.
   if pretrained:
      _model = load_model_weights('esrgan_rrdbnet', _model)

   # Convert the model into an evaluation model.
   return _model.eval()

