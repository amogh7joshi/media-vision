# The construction of the `imagevision.models` directory
# is done here with the `make build` command. After the
# pretrained model files are added to the directory, this
# command runs a Python script which in turn reconstructs
# the weights JSON file and unzips the downloaded files.
.PHONY: build
build:
	@if cd scripts; then python3 build_models.py; \
   	else \
   	  echo "\033[91mCould not enter the \`scripts\` directory. Check your Makefile path.\033[0m"; \
   	  exit 1; \
   	fi


