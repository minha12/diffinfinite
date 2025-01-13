# Dataset Mismatch Analysis

## Summary
Analysis of the DRSK pathology dataset revealed 5 files that exist in the masks directory but are missing from the images directory.

## Dataset Statistics
- Total mask files: 246,755
- Total image files: 246,750
- Difference: 5 files (masks - images)

## Missing Files
The following files are present in `../pathology-datasets/DRSK/full_dataset/masks` but missing from `../pathology-datasets/DRSK/full_dataset/images`:

```text
16355d5be68e1c55a49c67f320aa516f58dd855641c5e435652fd95d042402e4.jpg
4687cc7c8ccb1773f3e480d7e6e197b475cb41b904cfe13c467877e49d77c671.jpg
6e47be1dd0408a6b86c07a31b939d2a48aa931b72668007e1480e1746f38da9f.jpg
eae86f981405116c5c0536fdb2601bc6cf1d1549b39545d8a5bee27117238866.jpg
ee5d6de71a730aae7c8aa75448c42623c70d237fa5057bad302fab47cf875dba.jpg
```

## Implications
- These 5 mask files have no corresponding original images
- This mismatch needs to be addressed before using the dataset for training
- Possible actions:
  1. Locate the missing original images
  2. Remove the orphaned mask files
  3. Document this discrepancy in the dataset metadata
