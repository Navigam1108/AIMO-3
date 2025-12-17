# AIMO3 System-2 Data Foundry

This repository contains the **data construction pipeline** used for AIMO3 System-2 training.

## What This Repo Contains
- Ingestion pipelines for Numina, TIR, Nvidia OpenMath
- Safety decontamination (MinHash LSH)
- Recursive error–correction synthesis
- Curriculum mixing logic

## What This Repo Does NOT Contain
- Raw datasets
- Processed datasets
- Final training JSONL

Final datasets are distributed separately via Google Drive.

## Setup
```bash
pip install -r requirements.txt
