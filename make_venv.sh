#!/usr/bin/bash

python3 -m venv .venv_cs3415
source .venv_cs3415/bin/activate
pip install --upgrade pip

# Install from pyproject.toml
pip install ./
pip install ./[dev]