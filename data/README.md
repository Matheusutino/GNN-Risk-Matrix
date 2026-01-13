# Data Directory

This directory contains all data files used by the GNN-Risk-Matrix project.

## Directory Structure

```
data/
├── raw/                  # Raw data files (see download instructions below)
├── datasets/             # Preprocessed datasets ready for training
└── results/              # Model results and predictions
    ├── csv/              # Results in CSV format
    └── pkl/              # Results in pickle format
```

## Required Raw Data Files

The following files must be placed in the `data/raw/` directory before running the preprocessing:

### 1. injuries_expansion_taxonomy.csv
- **Description**: Taxonomy of injury types and related search queries
- **Format**: CSV with semicolon separator (`;`)
- **Columns**: injury type, query string
- **Download**:
  ```bash
  # Using gdown
  gdown --id 1Cc5WK-vk8hk00ufs_CEpfi1E6ZTfN4gG -O data/raw/injuries_expansion_taxonomy.csv
  ```

### 2. IncidentReports.csv
- **Description**: Consumer product safety incident reports from SaferProducts database
- **Format**: CSV (ISO-8859-2 encoding)
- **Source**: [SaferProducts.gov](https://www.saferproducts.gov/)
- **Download**:
  ```bash
  # Download and extract SPDB.zip
  wget https://www.saferproducts.gov/SPDB.zip
  unzip SPDB.zip

  # Move the IncidentReports.csv file to data/raw/
  mv IncidentReports.csv data/raw/
  ```

## Quick Setup

To download all required files:

```bash
# Install required tools (if not already installed)
pip install gdown wget

# Download injuries taxonomy
gdown --id 1Cc5WK-vk8hk00ufs_CEpfi1E6ZTfN4gG -O data/raw/injuries_expansion_taxonomy.csv

# Download SaferProducts database
wget https://www.saferproducts.gov/SPDB.zip
unzip SPDB.zip
mv IncidentReports.csv data/raw/
rm SPDB.zip  # Optional: remove the zip file
```

## Notes

- The raw data files are **not** included in the git repository due to their size
- Make sure both files are present in `data/raw/` before running the preprocessing
- The DataManager class will automatically check for these files and raise an error if they're missing
