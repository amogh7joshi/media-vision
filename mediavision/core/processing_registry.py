#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from collections.abc import Callable

import torch.nn as nn

from mediavision.core.exceptions import InvalidProcessorName

# Create a holder list of valid processors.
VALID_PROCESSORS = {}

def register_processor(name: str) -> Callable:
   """Register a processor as a valid processor and
   serialize it into the list of valid processors."""
   def decorator(f: Callable):
      # Add the processor to the list of valid colorizers.
      VALID_PROCESSORS[name] = f
      # Return the function.
      return f
   # Return the decorator.
   return decorator

def get_processor(processor_name: str) -> nn.Module:
   """Return the relevant function to get a processor."""
   # Validate the processor.
   if processor_name not in VALID_PROCESSORS.keys():
      raise InvalidProcessorName(processor_name)

   # Return the processor function from its relevant module list.
   return VALID_PROCESSORS[processor_name]

