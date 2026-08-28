#!/usr/bin/env bash
apt-get update && apt-get install -y libgl1 libglib2.0-0t64 libzbar0 libgomp1
pip install -r requirements.txt