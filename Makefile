# Enable running all commands in the same subshell
.ONESHELL:

# Define the name of the virtual environment
VENV := venv

# Detect the operating system (Windows vs Unix-like)
OSFLAG :=
ifeq ($(OS),Windows_NT)
    OSFLAG := windows
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OSFLAG := linux
    endif
    ifeq ($(UNAME_S),Darwin)
        OSFLAG := macos
    endif
endif

# Default target: setup the environment, install dependencies, open VSCode, and show activation instructions
.PHONY: start
start: venv install open_vscode print_activation_instructions

# Create the virtual environment only if it doesn't exist
.PHONY: venv
venv:
ifeq ($(OSFLAG),windows)
	if not exist $(VENV)\Scripts\activate (
		python -m venv $(VENV)
	)
else
	if [ ! -f "$(VENV)/bin/activate" ]; then \
		python3 -m venv $(VENV); \
	fi
endif

# Install dependencies in the activated environment
.PHONY: install
install:
ifeq ($(OSFLAG),windows)
	cmd /C "$(VENV)\Scripts\activate && pip install -r requirements.txt"
else
	. $(VENV)/bin/activate && pip install -r requirements.txt
endif

# Open VS Code after environment setup
.PHONY: open_vscode
open_vscode:
	code .

# Print activation instructions for the user at the end
.PHONY: print_activation_instructions
print_activation_instructions:
ifeq ($(OSFLAG),windows)
	@echo "To activate the virtual environment, run: .\\$(VENV)\\Scripts\\activate"
else
	@echo "To activate the virtual environment, run: source $(VENV)/bin/activate"
endif

# Run pylint on the source code
.PHONY: lint
lint:
ifeq ($(OSFLAG),windows)
	pylint src/**/*.py
else
	pylint src/**/*.py
endif

# Clean up by removing the virtual environment
.PHONY: clean
clean:
ifeq ($(OSFLAG),windows)
	if exist $(VENV) rmdir /S /Q $(VENV)
else
	rm -rf $(VENV)
endif
