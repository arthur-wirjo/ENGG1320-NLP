#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PYTHON_EXECUTABLE="python3" # Or just "python" if that's your preferred command
VENV_DIR="venv"           # Name for the virtual environment directory

echo "--- Setting up Python virtual environment in '$VENV_DIR' ---"

# Check if Python command exists
if ! command -v $PYTHON_EXECUTABLE &> /dev/null
then
    echo "Error: '$PYTHON_EXECUTABLE' command not found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_EXECUTABLE -m venv $VENV_DIR
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

echo "--- Activating virtual environment ---"
# Note: Activation within the script only lasts for the script's execution.
# You still need to activate manually in your terminal later.
source "$VENV_DIR/bin/activate"

echo "--- Installing dependencies from requirements.txt ---"

# Upgrade pip first
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in the current directory."
    # Deactivate environment if script fails here
    deactivate || true # Use '|| true' in case deactivate isn't found (e.g., outside venv)
    exit 1
fi

# Install packages
pip install -r requirements.txt

echo "--- Installation Complete ---"
echo "To run your Python script, first activate the environment in your terminal:"
echo "source $VENV_DIR/bin/activate"
echo "Then run your script:"
echo "python process_message.py"

# Deactivate environment at the end of the script (optional, cleans up script's own activation)
# deactivate || true

exit 0