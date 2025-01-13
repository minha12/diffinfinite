#!/bin/bash

echo "Finding mismatches between images and masks..."

# Direct comparison using process substitution and printing only unique entries
# Only show files that exist in masks but not in images
comm -13 \
    <(find ../pathology-datasets/DRSK/full_dataset/images -type f -printf "%f\n" | sort) \
    <(find ../pathology-datasets/DRSK/full_dataset/masks -type f -printf "%f\n" | sort)