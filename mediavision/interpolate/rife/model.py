#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from mediavision.interpolate.rife.networks import IFNet, ContextNet, FusionNet
from mediavision.core.model_registry import register_interpolator
from mediavision.core.types import TensorReturnType
from mediavision.core.base import MediaVisionModelBase
from mediavision.weights import load_model_weights

# Define the device to run computations on.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RIFEModel(MediaVisionModelBase):
   """Implementation of the RIFE model.

   Source from https://github.com/hzwer/arXiv2020-RIFE, which
   is an implementation of the paper "RIFE: Real-Time Intermediate
   Flow Estimation for Video Frame Interpolation" (see
   https://arxiv.org/abs/2011.06294).
   """
   def __init__(self, local_rank: int = -1):
      # Construct the three intermediate networks.
      self.flownet = IFNet()
      self.contextnet = ContextNet()
      self.fusionnet = FusionNet()

      # Construct the device.
      self.device()

      # Transfer the data to a distributed device.
      if local_rank != -1:
         self.flownet = DDP(self.flownet, device_ids = [local_rank], output_device = local_rank)
         self.contextnet = DDP(self.contextnet, device_ids = [local_rank], output_device = local_rank)
         self.fusionnet = DDP(self.fusionnet, device_ids = [local_rank], output_device = local_rank)

   def device(self):
      """Connect the networks to a device."""
      self.flownet.to(device)
      self.contextnet.to(device)
      self.fusionnet.to(device)
      return self

   def eval(self):
      """Convert the networks into evaluation networks."""
      self.flownet.eval()
      self.contextnet.eval()
      self.fusionnet.eval()
      return self

   def load_model(self, path: str, rank: int = -1):
      """Load the weights into the model intermediate networks."""
      # Ensure the path exists.
      if not os.path.exists(path):
         raise FileNotFoundError(f"The provided weight path {path} does not exist.")

      # Create a helper function for weight conversion.
      def convert(param):
         # Process the input paths.
         if rank == -1:
            return {
               k.replace("module.", ""): v
               for k, v in param.items()
               if "module." in k
            }
         else:
            return param

      # Load the weights into the intermediate networks.
      if rank <= 0:
         self.flownet.load_state_dict(convert(
            torch.load("{}/flownet.pkl".format(path), map_location = device)))
         self.contextnet.load_state_dict(convert(
            torch.load("{}/contextnet.pkl".format(path), map_location = device)))
         self.fusionnet.load_state_dict(convert(
            torch.load("{}/fusionnet.pkl".format(path), map_location = device)))

   def predict(self, images: torch.Tensor, flow: torch.Tensor,
               flow_gt: torch.Tensor = None) -> TensorReturnType:
      """Predict the frame output for an input set."""
      # Get the images.
      image0 = images[:, :3]
      image1 = images[:, 3:]

      # Get the context from the images, and interpolate the flow.
      c0 = self.contextnet(image0, flow[:, :2])
      c1 = self.contextnet(image1, flow[:, 2:4])
      flow = F.interpolate(flow, scale_factor = 2.0, mode = 'bilinear',
                           align_corners = False) * 2.0

      # Refine the outputs from the fusion network.
      refine_output, warped_0, warped_1, warped_0_gt, warped_1_gt = self.fusionnet(
         image0, image1, flow, c0, c1, flow_gt)

      # Create the output prediction frame.
      res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
      mask = torch.sigmoid(refine_output[:, 3:4])
      merged = warped_0 * mask + warped_1 * (1 - mask)
      pred = merged + res
      pred = torch.clamp(pred, 0, 1)

      # Return the prediction fame.
      return pred

   def inference(self, image0: torch.Tensor, image1: torch.Tensor,
                 scale: int = 1.0) -> TensorReturnType:
      """The main exposed inference method for constructing frames."""
      images = torch.cat((image0, image1), 1)
      flow, _ = self.flownet(images, scale)
      return self.predict(images, flow)

@register_interpolator(name = "rife_net")
def load_rife_model(pretrained = True) -> RIFEModel:
   """Constructs the rife frame interpolator and loads pretrained weights."""
   # Initialize the backend for inferencing.
   torch.set_grad_enabled(False)
   if torch.cuda.is_available():
      # Initialize the GPU backend.
      from torch.backends import cudnn
      cudnn.enabled = True
      cudnn.benchmark = True
      # Initialize the fp16 mode for lightweight
      # inference on a GPU with tensor cores.
      torch.set_default_tensor_type(torch.HalfTensor)

   # Construct the model.
   _model = RIFEModel()

   # Load the model weights.
   if pretrained:
      _model = load_model_weights("rife_nets", _model, load_param = "load_model")

   # Convert the model into an evaluation model.
   return _model.eval()



