# Tissue Analysis and Survival Analysis Pipeline

This project provides a pipeline for analyzing tissue composition and survival outcomes in colorectal cancer patients using TCGA data. It includes tools for calculating tissue fractions from probability matrices and performing survival analysis.

## Project Structure

```
.
├── README.md
├── code/
│   ├── generate_combined_data.py  # Script for generating initial combined dataset
│   ├── main.py                    # Main analysis script
│   ├── tissue_analysis.py         # Functions for tissue fraction calculation
│   └── survival_analysis.py       # Survival analysis implementations
├── TCGA_CRC/                      # Download TCGA_CRC processed .npy files
```

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - lifelines
  - matplotlib

## Data Requirements

The pipeline expects the following data structure:

1. Clinical information file (TCGA_CRC_info.csv):
   - Contains patient clinical data including survival information
   - Required columns: PATIENT, OS, OS.time, DSS, DSS.time, vital_status, MSI

2. NPY files directory:
   - Contains .npy files with tissue probability matrices
   - Each file should be named with the TCGA patient identifier
   - Matrices should contain probability scores for different tissue types:
     - Stroma (class 7)
     - Muscle (class 5)
     - Tumor (class 8)

## Pipeline Steps

### Step 0: Download TCGA_CRC processed .npy files
Download link: https://cloud.tsinghua.edu.cn/f/a1c87bb480eb4eae9bd2/?dl=1
Then extract the files to the current path.

### Step 1: Generate Combined Dataset
Run generate_combined_data.py to create the initial combined dataset:
```bash
python generate_combined_data.py
```
This script will:
- Calculate tissue fractions from .npy files
- Merge with clinical data
- Generate new_TCGA_CRC_clinic.csv
- Display summary statistics

### Step 2: Perform Survival Analysis
Run main.py to perform survival analysis:
```bash
python main.py
```
This script will:
- Load the combined dataset
- Perform multivariate Cox regression
- Generate visualization plots
- Display statistical results

## Output Files

1. new_TCGA_CRC_clinic.csv:
   - Combined dataset with clinical information and tissue fractions
   - Generated by generate_combined_data.py

2. multivariate_forest_plot.png:
   - Forest plot showing hazard ratios from Cox regression
   - Includes confidence intervals and p-values

3. significant_var_survival.png:
   - Kaplan-Meier survival curve for the most significant variable
   - Includes logrank test p-value

## Code Description

- generate_combined_data.py:
  - Processes .npy files to calculate tissue fractions
  - Merges results with clinical data
  - Includes progress tracking and error checking

- main.py:
  - Orchestrates survival analysis workflow
  - Generates visualization plots
  - Displays statistical results

- tissue_analysis.py:
  - Contains functions for calculating tissue fractions
  - Handles different tissue types (stroma, muscle, tumor)

- survival_analysis.py:
  - Implements Cox regression analysis
  - Creates forest plots and Kaplan-Meier curves
  - Performs statistical testing

## Usage Notes

1. Data Preparation:
   - Ensure all required data files are in place
   - Check file paths in respective scripts

2. Running the Pipeline:
   - First run generate_combined_data.py if combined dataset doesn't exist
   - Then run main.py for survival analysis

3. Output Interpretation:
   - Forest plot shows hazard ratios for all variables
   - Survival curve displays the most significant prognostic factor
   - Statistical summaries are printed to console

## Contact

For questions or issues, please create an issue in the repository or contact the maintainers.
