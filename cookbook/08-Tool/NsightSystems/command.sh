#!/bin/bash

rm -rf ./*.plan
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o model-BuildAndRun python3 main.py
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o model-OnlyRun     python3 main.py
