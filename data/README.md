# Fakeddit Dataset Setup

## Download Instructions
The Fakeddit dataset is available on GitHub: [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)

1. Download the CSV files (e.g., `fakeddit_processed.csv` or the specific split files).
2. Download the associated images.

## Directory Structure
Place the files in the following structure:
```text
data/
└── raw/
    ├── fakeddit_processed.csv  # The main CSV file containing text, image paths, and labels
    └── images/                 # Directory containing all images
        ├── img1.jpg
        ├── img2.png
        └── ...
```

## Label Mapping
Fakeddit uses various label schemes. The baseline implementation expects a mapping to target classes (e.g., REAL, FAKE). 
Ensure the `label` column in the CSV contains the identifiers used in the `baseline.yaml` configuration.
