#!/usr/bin/env bash
# Build the image once, then drop into the container with the workspace mounted.
set -e
cd "$(dirname "$0")"
docker build -t f1tenth-ros2 .
docker run -it --rm -v "$(pwd)":/ws f1tenth-ros2 bash
