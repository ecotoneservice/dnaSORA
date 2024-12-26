# dnaSORA - Unified Denoiser-Classifier for Chromo Data

This repository contains implementation of a unified denoiser-classifier model for processing and analyzing chromo data. The model predicts misrepresented token point clouds for the hawaiian experiment data. See the reseaarch paper for the details.

## Setup

### Hardware requirements

* GPU 24GB VRAM
* 64 GB RAM
* 4 Cores CPU
* 10 GB free disk space

### Environment Setup
1. Create a Linux Conda environment with Python 3.10:
```bash
conda create -n chromo-env python=3.10
conda activate chromo-env
```

2. Install dependencies 

pip packages:
```bash
pip install -r requirements.txt
```

For exporting plots to images
```bash
conda install -c conda-forge firefox geckodriver
```

Requred for rendering plots with custom fonts
```bash
sudo apt install msttcorefonts -qq
```

## Project Structure
```
├── notebooks/
│   ├── RD_Cache/          # Raw chromo data files (*.txt)
│   ├── preprocessing.ipynb
│   ├── MD.ipynb
│   ├── denoiser.ipynb
│   ├── denoiser_analysis.ipynb
│   ├── conv_classifier.ipynb
│   ├── unified_classifier.ipynb
│   └── classifier_analysis.ipynb
├── configs/
│   ├── denoiser/         # Denoiser model configurations
│   └── classifier/       # Unified classifier model configurations
├── src/                 # Shared generated source code
├── data/                # Processed data files (*.pkl)
├── checkpoints/         # Trained model checkpoints
└── ablation/           # Ablation study results and data
```

## Workflow

### 1. Data Preprocessing
1. Make sure hawaiian experiments data in `notebooks/RD_Cache/` and gausian overrides, measured misrepresented token positions for validation are in `preprocessing/gausian_overrides.json`. Default data is suffcient for reproducing paper results
2. Run `preprocessing.ipynb` to process the raw data
   - Output: Processed data files (*.pkl) in cached folders

### 2. Mock Data Generation
1. Run `MD.ipynb` to generate mock gausian data for training

### 3. Model Training

#### Denoiser Training
1. Configure model parameters in `configs/denoiser/`
   - To skip specific configurations, set `skip: true` in the config file
2. Run `denoiser.ipynb` to train all denoiser base models
3. For analysis:
   - Run `denoiser_analysis.ipynb`
   - Set the correct model configuration file
   - Execute to perform all available analyses for the denoiser base model

#### Classifier Training
1. Configure models in `configs/classifier/`
2. Run `unified_classifier.ipynb` to train all unified classifiers for evaluation
3. Run `conv_classifier.ipynb` to train conv classifier for evaluation
4. For analysis:
   - Run `classifier_analysis_create.ipynb`
      - This will evaluate all trained unified classifiers, extract performance metrics and save them into a separate file in metrics_results
   - Run `classifier_analysis_parse.ipynb`
      - This fill read latest from metrics_results and create .md files with generalised results: all models perf, categorical perf and detailed mu preditions

## Model Configuration

The project uses configuration files located in:
- `configs/denoiser/`: Denoiser model definitions
- `configs/classifier/`: Unified classifier model definitions
- `configs/MD/`: MD datasets definitions
- `configs/MDDense/`: DenseMD datasets definitions

All configurations are linked and set by default, but you can modify them according to your needs.