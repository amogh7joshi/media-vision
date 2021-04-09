#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import numpy as np

from mediavision.core.types import ImageOrPath
from mediavision.core.model_registry import get_model
from mediavision.core.processing_registry import get_processor
from mediavision.core.utils import is_video_path

def colorize(path: ImageOrPath, model: str = "cic_siggraph17", **kwargs) -> np.ndarray:
   """Recolor a provided image."""
   # Load in the model.
   model = get_model(model)(pretrained = True)

   # Choose either the image or video processor.
   if is_video_path(path):
      processor = get_processor('cic_colorizer_video')
   else:
      processor = get_processor('cic_colorizer_image')

   # Check whether an output path has been provided (for video).
   if "output_path" in kwargs:
      out = kwargs["output_path"]
   elif "out_path" in kwargs:
      out = kwargs["out_path"]
   elif "out" in kwargs:
      out = kwargs["out"]
   else:
      out = None

   # Return the colorized output.
   if out is not None:
      return processor(path, model, out)
   else:
      return processor(path, model)


