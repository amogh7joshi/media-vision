#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from typing import Union

import torch.nn as nn

def build_double_convolution_block(inp_filters: int,
                                   out_filters: int,
                                   *,
                                   kernel_size: Union[int, list, tuple] = (3, 3),
                                   stride: Union[int, list, tuple] = 1,
                                   padding = 1,
                                   bias: bool = True) -> nn.Sequential:
   """Builds a double convolution block.

   In essence, the sequence of layers in a double convolution
   block consists of:

   Conv2D -> ReLU -> Conv2D -> ReLU -> BatchNormalization

   Parameters:
      - inp_filters: The initial input filters for the convolution layer.
      - out_filters: The filters used for the remaining convolution layers.
      - kernel_size: The size of the convolution kernel.
      - stride: The size of the strides of the convolution layers either an
                integer or a list of parameters to be used.
      - padding: The padding parameter.
      - bias: Whether to use a bias or not.
   """
   # Check the passed parameters.
   if isinstance(stride, (list, tuple)):
      stride_1 = stride[0]
      stride_2 = stride[1]
   else:
      stride_1 = stride
      stride_2 = stride

   # Construct the different layers of the model.
   layers = (
      nn.Conv2d(inp_filters, out_filters, kernel_size = kernel_size,
                stride = stride_1, padding = (padding, ), bias = bias),
      nn.ReLU(True),
      nn.Conv2d(out_filters, out_filters, kernel_size = kernel_size,
                stride = stride_2, padding = (padding, ), bias = bias),
      nn.ReLU(True),
      nn.BatchNorm2d(out_filters)
   )

   # Create a Sequential model with the layers and return it.
   current_stem = nn.Sequential(*layers)
   return current_stem

def build_triple_convolution_block(inp_filters: int,
                                   out_filters: int,
                                   *,
                                   kernel_size: Union[int, list, tuple] = (3, 3),
                                   stride: Union[int, list, tuple] = 1,
                                   dilation: int = 1,
                                   padding: Union[int, list, tuple] = 1,
                                   bias: bool = True) -> nn.Sequential:
   """Builds a triple convolution block.

   In essence, the sequence of layers in a double convolution
   block consists of:

   Conv2D -> ReLU -> Conv2D -> ReLU -> Conv2D -> ReLU -> BatchNormalization

   Parameters:
      - inp_filters: The initial input filters for the convolution layer.
      - out_filters: The filters used for the remaining convolution layers.
      - kernel_size: The size of the convolution kernel.
      - dilation: The dilation rate of the convolution layers.
      - stride: The size of the strides of the convolution layers either an
                integer or a list of parameters to be used.
      - padding: The padding parameter.
      - bias: Whether to use a bias or not.
   """
   # Check the passed parameters.
   if isinstance(stride, (list, tuple)):
      stride_1 = stride[0]
      stride_2 = stride[1]
      stride_3 = stride[2]
   else:
      stride_1 = stride
      stride_2 = stride
      stride_3 = stride

   # Construct the different layers of the model.
   layers = (
      nn.Conv2d(inp_filters, out_filters, kernel_size = kernel_size,
                dilation = (dilation, ), stride = stride_1,
                padding = padding, bias = bias),
      nn.ReLU(True),
      nn.Conv2d(out_filters, out_filters, kernel_size = kernel_size,
                dilation = (dilation, ), stride = stride_2,
                padding = padding, bias = bias),
      nn.ReLU(True),
      nn.Conv2d(out_filters, out_filters, kernel_size = kernel_size,
                dilation = (dilation, ), stride = stride_3,
                padding = padding, bias = bias),
      nn.ReLU(True),
      nn.BatchNorm2d(out_filters)
   )

   # Create a Sequential model with the layers and return it.
   current_stem = nn.Sequential(*layers)
   return current_stem
