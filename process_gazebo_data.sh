#!/bin/bash

# Ensure a filename is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <output_filename>"
    exit 1
fi

# Set output file name with enforced .xml extension
OUTPUT_FILE="$(pwd)/gazebo_data/$1.xml"

# Run the command with the new output path
gz log -e -f ~/.gazebo/log/*/gzserver/state.log -z 30 --filter burger/base_footprint.pose > "$OUTPUT_FILE"
rm -rf ~/.gazebo/log/*
echo "Filtered log saved to: $OUTPUT_FILE"
