#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import re
import sys

# A base list of colors which are used in the toolbox.
_MEDIAVISION_COLORS = {
   'blue': '\033[94m',
   'red': '\033[91m',
   'green': '\033[92m',
   'end': '\033[0m'
}

# Create the actual methods.
def colorize_text(inp: str, color: str) -> str:
   """Colorizes the actual text as necessary."""
   # Validate the input color.
   if color not in _MEDIAVISION_COLORS.keys():
      raise ValueError(f"Invalid color {color} received.")

   # Return the formatted string as necessary.
   return "".join(
      [_MEDIAVISION_COLORS[color], inp, _MEDIAVISION_COLORS['end']])

def print_info(inp: str, color: str = None) -> None:
   """Handles the printing of information in a colored format.

   This method can take in inputs in two formats: a single
   string can be provided with a single color `color`:

   >>> print_info("my example string", color = 'red')

   Otherwise, for multicolored outputs or other representations,
   the colors can be inserted directly in the string.

   >>> print_info("{red}my{/red} {blue}colorful{/blue} {red}text{/red}")
   """
   # Parse through the string for any braces which represent
   # individual colors, and differentiate the string as such.
   if color is None and "{" in inp:
      # Create a pattern to match the colored segments.
      c_pattern = re.compile("\\{(.*?)\\}.*?\\{/(.*?)\\}") # noqa

      # Create an output string.
      output_string = inp[:inp.index("{")]
      inp = inp[inp.index("{"):]

      # Iterate through the string portions until it is empty.
      while len(inp) != 0:
         # Extract the latest match of the colored pattern.
         portion = list(re.finditer(c_pattern, inp))[0]
         pattern = re.findall(c_pattern, inp)[0][0]

         # Check whether it is the first index of the string, if not,
         # then append the part of the string up to that index into
         # the current output string and then parse.
         if portion.start() != 0:
            output_string += inp[:portion.start()]
            inp = inp[portion.start():]

         # Colorize the portion of the string that needs to be colorized.
         emp_txt = re.sub('\\{.*?\\}', '', portion.group(0)) # noqa
         output_string += colorize_text(emp_txt, pattern)

         # Reduce the size of the input string.
         inp = inp[portion.end():]
   elif color is not None and "{" not in inp:
      # A singular color case where there are no braces in the input.
      output_string = colorize_text(inp, color = color)
   else: # Raise an error for an invalid input format.
      raise ValueError("Invalid input format received.")

   # Write the output string to the standard output.
   sys.stdout.write(output_string)

