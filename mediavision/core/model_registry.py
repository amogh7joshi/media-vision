#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import inspect
from collections.abc import Callable

import torch.nn as nn

from mediavision.core.exceptions import InvalidModelName

# Create a holder list of valid models.
VALID_MODELS = []

def register_model(name: str) -> None:
   """Register a model as valid and serialize it into the list
   of valid models. This should not be called outside of this module."""
   # Add the model to the list of valid models.
   VALID_MODELS.append(name)

# Create a holder list of valid colorizers.
VALID_COLORIZERS = {}

def register_colorizer(name: str) -> Callable:
   """Register a colorizer as valid for the recoloring
   module and serialize it into the list of valid colorizers."""
   def decorator(f: Callable):
      # Add the model to the list of valid colorizers.
      VALID_COLORIZERS[name] = f
      # Add the model to the overarching valid model list.
      register_model(name)
      # Return the function.
      return f
   # Return the decorator.
   return decorator

# Create a holder list of valid upscalers.
VALID_UPSCALERS = {}

def register_upscaler(name: str) -> Callable:
   """Register an upscaler as valid for the recoloring
   module and serialize it into the list of valid upscalers."""
   def decorator(f: Callable):
      # Add the model to the list of valid upscalers.
      VALID_UPSCALERS[name] = f
      # Add the model to the overarching valid model list.
      register_model(name)
      # Return the function.
      return f
   # Return the decorator.
   return decorator

# Create a holder list of valid interpolators.
VALID_INTERPOLATORS = {}

def register_interpolator(name: str) -> Callable:
   """Register an interpolator as valid for the recoloring
   module and serialize it into the list of valid interpolators."""
   def decorator(f: Callable):
      # Add the model to the list of valid upscalers.
      VALID_INTERPOLATORS[name] = f
      # Add the model to the overarching valid model list.
      register_model(name)
      # Return the function.
      return f
   # Return the decorator.
   return decorator

def get_model(model_name: str) -> nn.Module:
   """Return the relevant function to get a model."""
   # Validate the model.
   if model_name not in VALID_MODELS:
      raise InvalidModelName(model_name)

   # Return the model function from its relevant module list.
   frm = inspect.stack()[1]
   module = inspect.getmodule(frm[0])
   if "colorize" in module.__name__:
      return VALID_COLORIZERS[model_name]
   elif "upscale" in module.__name__:
      return VALID_UPSCALERS[model_name]
   elif "interpolate" in module.__name__:
      return VALID_INTERPOLATORS[model_name]

   # Raise an error for invalid modules.
   raise ValueError(f"The model {model_name} does not belong to any modules.")

