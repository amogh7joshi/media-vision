#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from typing import Tuple, List, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mediavision.interpolate.rife.modules import IFBlock, Conv2, warp, upsample_block
from mediavision.core.types import TensorReturnType

# Define the device to run computations on.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IFNet(nn.Module):
   """The intermediate flow network of the RIFE model."""
   def __init__(self):
      # Initialize the module.
      super(IFNet, self).__init__()

      # Construct the network blocks.
      self.block0 = IFBlock(6, scale = 8, out_filters = 192)
      self.block1 = IFBlock(10, scale = 4, out_filters = 128)
      self.block2 = IFBlock(10, scale = 2, out_filters = 96)
      self.block3 = IFBlock(10, scale = 1, out_filters = 48)

   def forward(self, x: torch.Tensor, scale: int = 1.0) \
         -> Tuple[TensorReturnType, List[TensorReturnType]]:
      # Interpolate the tensor if necessary.
      if scale != 1.0:
         x = F.interpolate(x, scale_factor = scale, mode = 'bilinear',
                           align_corners = False)

      # Pass through the different convolution blocks, then interpolate the
      # result and warp both of these other results for each of the IF Blocks.
      flow0 = self.block0(x)
      F1 = flow0
      F1_large = F.interpolate(F1, scale_factor = 2.0, mode = 'bilinear',
                               align_corners = False) * 2.0
      warped_0 = warp(x[:, :3], F1_large[:, :2])
      warped_1 = warp(x[:, 3:], F1_large[:, 2:4])
      flow1 = self.block1(torch.cat((warped_0, warped_1, F1_large), 1))
      F2 = (flow0 + flow1)
      F2_large = F.interpolate(F2, scale_factor = 2.0, mode = 'bilinear',
                               align_corners = False) * 2.0
      warped_0 = warp(x[:, :3], F2_large[:, :2])
      warped_1 = warp(x[:, 3:], F2_large[:, 2:4])
      flow2 = self.block1(torch.cat((warped_0, warped_1, F2_large), 1))
      F3 = (flow0 + flow1 + flow2)
      F3_large = F.interpolate(F3, scale_factor = 2.0, mode = 'bilinear',
                               align_corners = False) * 2.0
      warped_0 = warp(x[:, :3], F2_large[:, :2])
      warped_1 = warp(x[:, 3:], F2_large[:, 2:4])
      flow3 = self.block1(torch.cat((warped_0, warped_1, F3_large), 1))
      F4 = (flow0 + flow1 + flow2 + flow3)

      # Re-interpolate the tensor if necessary.
      if scale != 1.0:
         F4 = F.interpolate(F4, scale_factor = 1 / scale,
                            mode = 'bilinear', align_corners = False) / scale

      # Return the different tensors.
      return F4, [F1, F2, F3, F4]

class ContextNet(nn.Module):
   """The context network of the RIFE model."""
   def __init__(self):
      # Initialize the module.
      super(ContextNet, self).__init__()

      # Build the convolution layers.
      c = 32
      self.conv0 = Conv2(3, c)
      self.conv1 = Conv2(c, c)
      self.conv2 = Conv2(c, 2 * c)
      self.conv3 = Conv2(2 * c, 4 * c)
      self.conv4 = Conv2(4 * c, 8 * c)

   def forward(self, x: torch.Tensor, flow: torch.Tensor) -> List[TensorReturnType]:
      # Pass through the convolution blocks and warp the tensors.
      x = self.conv0(x)
      x = self.conv1(x)
      flow = F.interpolate(flow, scale_factor = 0.5, mode = 'bilinear',
                           align_corners = False) * 0.5
      f1 = warp(x, flow)
      x = self.conv2(x)
      flow = F.interpolate(flow, scale_factor = 0.5, mode = 'bilinear',
                           align_corners = False) * 0.5
      f2 = warp(x, flow)
      x = self.conv3(x)
      flow = F.interpolate(flow, scale_factor = 0.5, mode = 'bilinear',
                           align_corners = False) * 0.5
      f3 = warp(x, flow)
      x = self.conv4(x)
      flow = F.interpolate(flow, scale_factor = 0.5, mode = 'bilinear',
                           align_corners = False) * 0.5
      f4 = warp(x, flow)
      return [f1, f2, f3, f4]

class FusionNet(nn.Module):
   """The fusion network of the RIFE model."""
   def __init__(self):
      # Initialize the module.
      super(FusionNet, self).__init__()

      # Construct the downsampling vanilla convolution blocks.
      c = 32
      self.conv0 = Conv2(10, c)
      self.down0 = Conv2(c, 2 * c)
      self.down1 = Conv2(4 * c, 4 * c)
      self.down2 = Conv2(8 * c, 8 * c)
      self.down3 = Conv2(16 * c, 16 * c)
      self.up0 = upsample_block(32 * c, 8 * c)
      self.up1 = upsample_block(16 * c, 4 * c)
      self.up2 = upsample_block(8 * c, 2 * c)
      self.up3 = upsample_block(4 * c, c)
      self.conv = nn.ConvTranspose2d(c, 4, 4, 2, 1)

   def forward(self, img0: torch.Tensor, img1: torch.Tensor, flow: torch.Tensor,
               c0: torch.Tensor, c1: torch.Tensor, flow_gt: torch.Tensor) -> Tuple:
      # Warp the input image.
      warped_0 = warp(img0, flow[:, :2])
      warped_1 = warp(img1, flow[:, 2:4])

      # Warp the ground truth.
      if flow_gt is None:
         warped_0_gt, warped_1_gt = None, None
      else:
         warped_0_gt = warp(img0, flow_gt[:, :2])
         warped_1_gt = warp(img1, flow_gt[:, 2:4])

      # Pass through the class downsampling and upsampling layers.
      x = self.conv0(torch.cat((warped_0, warped_1, flow), 1))
      s0 = self.down0(x)
      s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
      s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
      s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
      x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
      x = self.up1(torch.cat((x, s2), 1))
      x = self.up2(torch.cat((x, s1), 1))
      x = self.up3(torch.cat((x, s0), 1))
      x = self.conv(x)

      # Return all of the images.
      return x, warped_0, warped_1, warped_0_gt, warped_1_gt







