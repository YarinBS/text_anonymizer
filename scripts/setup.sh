#!/bin/bash

cd "$(dirname "$0")/.." || exit 1

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
