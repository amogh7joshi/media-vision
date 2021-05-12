#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from mediavision.core.types import ImageOrPath
from mediavision.core.model_registry import get_model
from mediavision.core.processing_registry import get_processor
from mediavision.core.utils import is_video_path, load_image

def interpolate(*paths: ImageOrPath, output_path: ImageOrPath = None,
                model: str = "rife_net", **kwargs) -> None:
   """Interpolate a provided image."""
   # Load in the model.
   model = get_model(model)(pretrained = True)

   # Check whether to use the image or video processor.
   if len(paths) == 1:
      if is_video_path(paths[0]): # noqa
         # Use the video processor.
         processor = get_processor('rife_interpolator_video')
         return processor(paths[0], output_path, model, **kwargs)
      else: # Can't interpolate only one image.
         raise ValueError("Expected either a video path or two image paths.")
   else:
      # Use the image processor.
      processor = get_processor('rife_interpolator_image')
      return processor(
         [load_image(img) for img in paths], output_path, model, **kwargs) # noqa
