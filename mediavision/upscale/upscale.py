#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from mediavision.core.types import ImageOrPath
from mediavision.core.model_registry import get_model
from mediavision.core.processing_registry import get_processor

def upscale(image: ImageOrPath, model: str = "esrgan_rrdbnet"):
   """Upscale a provided image."""
   # Get the processor.
   processor = get_processor('esrgan_upscaler')

   # Load in the model.
   model = get_model(model)(pretrained = True)

   # Return the colorized output.
   return processor(image, model)

