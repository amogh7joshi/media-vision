#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import torch
import torch.nn as nn

from mediavision.colorize.cic.modules import build_double_convolution_block
from mediavision.colorize.cic.modules import build_triple_convolution_block
from mediavision.core.model_registry import register_colorizer
from mediavision.core.types import TensorReturnType
from mediavision.core.base import MediaVisionModelBase
from mediavision.weights import load_model_weights

class CICECCV16Model(nn.Module, MediaVisionModelBase):
   """Implementation of the ECCV16 model.

   Sourced from: https://github.com/richzhang/colorization/, which
   itself is an implementation of the paper "Colorful Image Colorization"
   (see https://arxiv.org/abs/1603.08511).
   """
   def __init__(self):
      # Initialize the current model.
      super(CICECCV16Model, self).__init__()

      # Construct the model stems.
      self.model1 = build_double_convolution_block(
         1, 64, stride = (1, 2))
      self.model2 = build_double_convolution_block(
         64, 128, stride = (1, 2))
      self.model3 = build_triple_convolution_block(
         128, 256, stride = (1, 1, 2))
      self.model4 = build_triple_convolution_block(
         256, 512, stride = (1, 1, 1))
      self.model5 = build_triple_convolution_block(
         512, 512, dilation = 2, stride = (1, 1, 1), padding = 2)
      self.model6 = build_triple_convolution_block(
         512, 512, dilation = 2, stride = (1, 1, 1), padding = 2)
      self.model7 = build_triple_convolution_block(
         512, 512, stride = (1, 1, 1), padding = 2)

      # The final step will contain a transposed convolution layer
      # as well as a custom output channel layer, so construct that.
      self.model8 = nn.Sequential(
         *(
            nn.ConvTranspose2d(512, 256, kernel_size = (4, 4), stride = (2, ),
                               padding = (1, ), bias = True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, ),
                      padding = (1, ), bias = True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, ),
                      padding = (1, ), bias = True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size = (1, 1), stride = (1, ),
                      padding = (0, ), bias = True)
         )
      )

      # Finally, create the output softmax and upsampling layers.
      self.softmax = nn.Softmax(1)
      self.model_out = nn.Conv2d(313, 2, kernel_size = (1, 1), padding = (0, ),
                                 dilation = (1, ), stride = (1, ), bias = False)
      self.upsample = nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = True)

   @staticmethod
   def normalize_L(in_: torch.Tensor) -> TensorReturnType:
      """Normalizes the input L grayscale channel."""
      return (in_ - 50.) / 100.

   @staticmethod
   def unnormalize_AB(in_: torch.Tensor) -> TensorReturnType:
      """Unnormalizes the output AB color channels."""
      return in_ * 110.

   def forward(self, x: torch.Tensor) -> TensorReturnType:
      # Normalize the input.
      x = self.normalize_L(x)

      # Forward pass through each model stem.
      for i in range(1, 9, 1):
         x = getattr(self, f"model{i}")(x)

      # Get the model output.
      x = self.model_out(self.softmax(x))
      x = self.upsample(x)

      # Return the unnormalized output.
      return self.unnormalize_AB(x)

class CICSIGGRAPHModel(nn.Module, MediaVisionModelBase):
   """Implementation of the SIGGRAPH17 model.

   Sourced from: https://github.com/richzhang/colorization/, which
   itself is an implementation of the paper "Colorful Image Colorization"
   (see https://arxiv.org/abs/1603.08511).
   """
   def __init__(self):
      # Initialize the current model.
      super(CICSIGGRAPHModel, self).__init__()

      # Construct the sequential model stems.
      self.model1 = build_double_convolution_block(4, 64)
      self.model2 = build_double_convolution_block(64, 128)
      self.model3 = build_triple_convolution_block(128, 256)
      self.model4 = build_triple_convolution_block(256, 512)
      self.model5 = build_triple_convolution_block(
         512, 512, dilation = 2, padding = 2)
      self.model6 = build_triple_convolution_block(
         512, 512, dilation = 2, padding = 2)
      self.model7 = build_triple_convolution_block(512, 512)

      # Construct the branching stems of the model.
      self.model8up = nn.Sequential(nn.ConvTranspose2d(
         512, 256, kernel_size = (4, 4), stride = (2, ), padding = (1, )))
      self.model3short8 = nn.Sequential(nn.Conv2d(
         256, 256, kernel_size = (3, 3), stride = (1, ), padding = (1, )))
      self.model8 = nn.Sequential(
         nn.ReLU(True), *build_double_convolution_block(256, 256).children())
      self.model9up = nn.Sequential(nn.ConvTranspose2d(
         256, 128, kernel_size = (4, 4), stride = (2, ), padding = (1, )))
      self.model2short9 = nn.Sequential(nn.Conv2d(
         128, 128, kernel_size = (3, 3), stride = (1, ), padding = (1, )))
      self.model9 = nn.Sequential(
         nn.ReLU(True), nn.Conv2d(128, 128, kernel_size = (3, 3), stride = (1, ), padding = (1, )),
         nn.ReLU(True), nn.BatchNorm2d(128))
      self.model10up = nn.Sequential(nn.ConvTranspose2d(
         128, 128, kernel_size = (4, 4), stride = (2, ), padding = (1, )))
      self.model1short10 = nn.Sequential(nn.Conv2d(
         64, 128, kernel_size = (3, 3), stride = (1, ), padding = (1, )))
      self.model10 = nn.Sequential(
         nn.ReLU(True),
         nn.Conv2d(128, 128, kernel_size = (3, 3), dilation = (1, ), stride = (1, ), padding = (1, )),
         nn.LeakyReLU(negative_slope = 0.2))

      # Create the model classification and regression outputs.
      self.model_class = nn.Sequential(nn.Conv2d(
         256, 529, kernel_size = (1, 1), dilation = (1, ), stride = (1, ), padding = (0, )))
      self.model_out = nn.Sequential(
         nn.Conv2d(128, 2, kernel_size = (1, 1), dilation = (1, ), stride = (1, ), padding = (0, )),
         nn.Tanh())

      # Create the upsampling and softmax layers.
      self.upsample4 = nn.Sequential(nn.Upsample(
         scale_factor = 4, mode = 'bilinear', align_corners = True))
      self.softmax = nn.Sequential(nn.Softmax(dim = 1))

   @staticmethod
   def normalize_L(in_: torch.Tensor) -> TensorReturnType:
      """Normalizes the input L grayscale channel."""
      return (in_ - 50.) / 100.

   @staticmethod
   def normalize_AB(in_: torch.Tensor) -> TensorReturnType:
      """Normalizes the input AB color channels."""
      return in_ / 110.

   @staticmethod
   def unnormalize_AB(in_: torch.Tensor) -> TensorReturnType:
      """Unnormalizes the output AB color channels."""
      return in_ * 110.

   def forward(self, x: torch.Tensor, inp2: torch.Tensor = None,
               mask: torch.Tensor = None) -> TensorReturnType:
      # Construct the second/mask inputs.
      if inp2 is None:
         inp2 = torch.cat((x * 0, x * 0), dim = 1)
      if mask is None:
         mask = x * 0

      # Construct the complete normalized input.
      x = torch.cat((self.normalize_L(x), self.normalize_AB(inp2), mask), dim = 1)

      # Forward pass through each of the sequential stems.
      conv1_2 = self.model1(x)
      conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
      conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
      conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
      conv5_3 = self.model5(conv4_3)
      conv6_3 = self.model6(conv5_3)
      conv7_3 = self.model7(conv6_3)

      # Construct the concatenated stems.
      conv8_3 = self.model8(self.model8up(conv7_3) + self.model3short8(conv3_3))
      conv9_3 = self.model9(self.model9up(conv8_3) + self.model2short9(conv2_2))
      conv10_2 = self.model10(self.model10up(conv9_3) + self.model1short10(conv1_2))

      # Create the output of the model and return it.
      out_reg = self.model_out(conv10_2)
      return self.unnormalize_AB(out_reg)

@register_colorizer(name = 'cic_eccv16')
def load_cic_eccv16_colorizer(pretrained: bool = True) -> CICECCV16Model:
   """Constructs the cic_eccv16 colorizer and loads pretrained weights."""
   # Construct the model.
   _model = CICECCV16Model()

   # Load pretrained weights if requested to.
   if pretrained:
      _model = load_model_weights('cic_eccv16', _model)

   # Convert the model into an evaluation model.
   return _model.eval()

@register_colorizer(name = 'cic_siggraph17')
def load_cic_siggraph17_colorizer(pretrained: bool = True) -> CICSIGGRAPHModel:
   """Constructs the cic_siggraph17 colorizer and loads pretrained weights."""
   # Construct the model.
   _model = CICSIGGRAPHModel()

   # Load pretrained weights if requested to.
   if pretrained:
      _model = load_model_weights('cic_siggraph17', _model)

   # Convert the model into an evaluation model.
   return _model.eval()


