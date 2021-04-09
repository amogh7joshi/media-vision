#!/usr/bin/env python3
# -*- coding = utf-8 -*-
"""A loose collection of methods to check system properties."""
import os
import re
import subprocess
from functools import lru_cache

from mediavision.core.types import AnyPath

@lru_cache(maxsize = 1)
def is_ffmpeg_installed() -> bool:
   """Check whether FFMPEG is installed on the system."""
   # Try and execute the version command.
   try:
      out = subprocess.check_output(['ffmpeg', '-version'])
   except OSError:
      # If an error has been encountered, by default it must
      # not be installed, or something is broken such that it
      # cannot be accessed, so return false regardless.
      return False

   # Otherwise, check the output. If there is a an output
   # containing the regex string `FFMPEG`, then it is installed,
   # otherwise return false since no output exists.
   if re.search(b'ffmpeg', out):
      return True
   return False

@lru_cache(maxsize = 1)
def is_ffprobe_installed() -> bool:
   """Check whether FFPROBE is installed on the system."""
   # Try and execute the version command.
   try:
      out = subprocess.check_output(['ffprobe', '-version'])
   except OSError:
      # If an error has been encountered, by default it must
      # not be installed, or something is broken such that it
      # cannot be accessed, so return false regardless.
      return False

   # Otherwise, check the output. If there is a an output
   # containing the regex string `FFMPEG`, then it is installed,
   # otherwise return false since no output exists.
   if re.search(b'ffprobe', out):
      return True
   return False

def has_audio(filename: AnyPath) -> bool:
   """Determines whether a provided video path has audio."""
   # Run the ffprobe subprocess to check the video output.
   result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=nb_streams", "-of",
                            "default=noprint_wrappers=1:nokey=1", filename],
                           stdout = subprocess.PIPE,
                           stderr = subprocess.STDOUT)

   # Return a boolean representing the output.
   return bool(int(result.stdout) - 1)

