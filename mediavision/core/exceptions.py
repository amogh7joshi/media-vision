#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from mediavision.core.types import AnyPath

class DataFileMissing(FileNotFoundError):
   """When an included library data file is missing."""
   def __init__(self, purpose: str, fname: AnyPath):
      # Set the class variables.
      self.purpose = purpose
      self.fname = fname

   def __str__(self):
      # Format the string with the filename.
      return f"The data file {self.purpose} which should " \
             f"be at {self.fname} is missing."

class InvalidModelName(NameError):
   """When a provided model name is invalid."""
   def __init__(self, name: str):
      # Set the class variables.
      self.name = name

   def __str__(self):
      # Format the string with the filename.
      return f"Received invalid model name {self.name}."

class InvalidProcessorName(NameError):
   """When a provided processor  name is invalid."""
   def __init__(self, name: str):
      # Set the class variables.
      self.name = name

   def __str__(self):
      # Format the string with the filename.
      return f"Received invalid processor name {self.name}."
