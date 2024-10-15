# Neuro Signal Processing Mini Project 1

## Overview

This project explores neural processing of emotionally provocative auditory stimuli using fMRI data. The analysis focuses on identifying brain regions activated when subjects listen to positive versus negative emotional music, as well as non-emotional stimuli (pure tones). The project is divided into two main parts:

1. **Part 1**: General Linear Model (GLM) analysis to find brain activations.
2. **Part 2**: Independent Component Analysis (ICA) to identify spatial patterns in brain activity.

The dataset used is from OpenNeuro ([Dataset: ds000171](https://openneuro.org/datasets/ds000171/versions/00001)).

## Project Structure

```bash

📁 neuro-signal-processing-project/
│
├── 📁 data/              # Raw and processed data
│   ├── raw/              # Raw fMRI data from OpenNeuro
│   ├── processed/        # Preprocessed fMRI data (motion correction, smoothing)
│   └── metadata/         # Experiment details, participant info, design matrices
│
├── 📁 src/               # Source code for preprocessing and analysis
│   ├── preprocess/       # Preprocessing scripts (motion correction, smoothing, etc.)
│   ├── glm/              # Scripts for General Linear Model (GLM) analysis
│   ├── ica/              # Independent Component Analysis (ICA) scripts
│   ├── utils/            # Helper functions (loading data, plotting, etc.)
│   └── analysis/         # Second-level analysis and results
│
├── 📁 notebooks/         # Jupyter notebooks for step-by-step analysis
│   ├── preprocessing.ipynb
│   ├── glm_analysis.ipynb
│   └── ica_analysis.ipynb
│
├── 📁 results/           # GLM and ICA outputs
│   ├── glm/              # GLM analysis outputs (beta maps, contrast maps, etc.)
│   ├── ica/              # ICA component maps and analyses
│   ├── figures/          # Figures like design matrices, brain region maps
│   └── logs/             # Processing logs
│
├── 📁 docs/              # Documentation and report
│   └── report/           # Final report files and theoretical answers
│
├── environment.yml       # Conda environment configuration file
├── .gitignore            # Ignored files and directories
├── README.md             # Project overview

```

## Dataset

The dataset used in this project is publicly available on OpenNeuro:

- **Name**: fMRI Study of Emotional and Non-Emotional Auditory Stimuli
- **Dataset ID**: [ds000171](https://openneuro.org/datasets/ds000171/versions/00001)
- **Paper Reference**: Lepping, et al., 2015

Subjects listened to blocks of positive or negative emotional music interleaved with pure tones (neutral stimuli). Only control subjects and the task-music_run are used for this project.

## Setup

### Prerequisites

- Python 3.x
- Conda or pip for environment management
- fMRIPrep or FSL for preprocessing

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neuro-signal-processing-project.git
   cd neuro-signal-processing-project 

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate neuro-signal-processing
3. Download the dataset from OpenNeuro and place it in the data/raw/ directory.

### Usage

#### 1. Data Preprocessing

Preprocess the fMRI data using motion correction and smoothing steps.

- Script location: `src/preprocess/preprocess.py`
- Jupyter Notebook: `notebooks/preprocessing.ipynb`

#### 2. GLM Analysis

Run the General Linear Model (GLM) to find beta maps and contrast maps for positive versus negative music.

- Script location: `src/glm/glm_analysis.py`
- Jupyter Notebook: `notebooks/glm_analysis.ipynb`

#### 3. ICA Analysis (Variant 2)

Perform Independent Component Analysis (ICA) to find spatial patterns in the fMRI data.

- Script location: `src/ica/ica_analysis.py`
- Jupyter Notebook: `notebooks/ica_analysis.ipynb`

### Results

Processed data and results can be found in the `results/` directory:

- **GLM results**: `results/glm/`
- **ICA results**: `results/ica/`
- **Figures**: `results/figures/`

### Documentation

Detailed explanations and answers to theoretical questions can be found in the `docs/report/` directory.



