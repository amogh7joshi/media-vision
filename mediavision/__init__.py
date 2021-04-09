#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import logging as _logging
import importlib as _importlib

# Configure the logger.
_logging.basicConfig(level = _logging.WARNING,
                     format = '%(levelname)s - %(message)s')

# At the initialization of the library, all of the different
# modules containing models need to be initialized and their
# relevant modules registered, which will be achieved through
# a dummy import of those modules using importlib. The same
# process then needs to be repeated for all of the processors.
def _initialize_registry():
   _MODEL_CONTAINING_MODULES = ['mediavision.colorize.cic.models',
                                'mediavision.upscale.esrgan.model',
                                'mediavision.interpolate.rife.model']
   for _module in _MODEL_CONTAINING_MODULES:
      _ = _importlib.__import__(_module)
   _PROCESSOR_CONTAINING_MODULES = ['mediavision.colorize.cic.utils',
                                    'mediavision.upscale.esrgan.utils',
                                    'mediavision.interpolate.rife.utils']
   for _module in _PROCESSOR_CONTAINING_MODULES:
      _ = _importlib.__import__(_module)
_initialize_registry()

# Run the system validation methods to check whether the system
# has the non-critical dependencies (FFMPEG primarily) and is
# functioning as expected in most methods/areas. To do this,
# the `mediavision.system` module is imported and any method
# matching the pattern 'is_*_installed' is run. Also, to prevent
# the same warning message from being repeated multiple times, a
# cache is kept of all the warnings already administered, but it
# is imported from another module to prevent issues with reloading.
def _check_system_properties():
   import re as _re
   from mediavision.core.initialization \
      import _ADMINISTERED_WARNINGS # noqa
   _mod = _importlib.import_module('mediavision.system')
   _meths = _re.findall('(is_(.*?)_installed)',
                        "$".join(list(vars(_mod).keys())))
   for (_meth, _name) in _meths:
      if not getattr(_mod, _meth):
         if _name not in _ADMINISTERED_WARNINGS:
            _logging.warning(f"{_name.upper()} is not installed "
                             f"on your system. This may cause issues "
                             f"when executing certain methods.")
            _ADMINISTERED_WARNINGS.append(_name)
_check_system_properties()

# Mid/low-level methods should be imported directly from their
# relevant submodule, but the high-level image methods will be
# directly imported from this top-level module.
from mediavision.colorize.colorize import colorize
from mediavision.upscale.upscale import upscale
from mediavision.interpolate.interpolate import interpolate

# Import the different codes from the code module, so that they
# are easily accessible to the user in different method uses.
from mediavision.codes import *

