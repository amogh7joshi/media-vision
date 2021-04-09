#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import json

from functools import lru_cache

import torch
import torch.utils.model_zoo as zoo

from mediavision.core.types import AnyModel
from mediavision.core.exceptions import DataFileMissing

# Ensure that the path to the weights file exists.
weights_path = os.path.join(os.path.dirname(__file__), 'weights.json')
if not os.path.exists(weights_path):
   raise DataFileMissing("containing pretrained weights", weights_path)

# Methods that actually perform the weight loading process.
@lru_cache(maxsize = 1)
def construct_weight_dict() -> dict:
   """Loads in the weights dict from the JSON file."""
   with open(weights_path, 'r') as json_file:
      return json.load(json_file)

def load_model_weights(name: str, model: AnyModel, **kwargs) -> AnyModel:
   """Load pretrained weights into a provided model."""
   # Construct the weight dictionary.
   _weights_dict = construct_weight_dict()

   # Ensure that a valid model name has been provided, and get the path.
   if name not in _weights_dict.keys():
      raise ValueError(f"Invalid name {name} received, expected on"
                       f"of the following: {[*_weights_dict.keys()]}.")
   path = _weights_dict[name]

   # Check if the file is a local path or whether it needs to be downloaded.
   if os.path.exists(path):
      download = False
   else:
      download = True

   # Load corresponding weights into their correct model.
   if path.endswith('pth'):
      # The model is a torch model.
      if download:
         _weights = zoo.load_url(path, map_location = 'cpu', check_hash = True)
      else:
         _weights = torch.load(path, map_location = 'cpu')

      # Check if any user parameters are provided.
      if "load_params_torch" in kwargs:
         _weights = (_weights[kwargs['load_params_torch']])

      # Load the model weights either into the model itself or into a
      # container class with the model (a `load_weights` type method).
      if "load_param" in kwargs:
         model = getattr(model, kwargs["load_param"])(_weights)
      else:
         model.load_state_dict(_weights) # noqa

   # Return the model with loaded weights.
   return model


