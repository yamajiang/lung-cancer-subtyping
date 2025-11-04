# Problem Definition

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Accurate subtype classification plays a critical role in diagnosis and treatment planning. Non–small cell lung cancer (NSCLC) accounts for the majority of lung cancers, with main subtypes: adenocarcinoma (LUAD), squamous cell carcinoma (LUSC), and large cell carcinoma (LCLC).

Traditional subtype diagnosis requires expert pathologist annotation, which is time-consuming and subjective. This project aims to develop a **weakly supervised learning model** that can automatically classify whole-slide histopathology images. Since LCLC is relatively rare and not well-represented in public datasets, this project focuses on weakly supervised classification between the two most common NSCLC subtypes: **LUAD vs LUSC**. A normal (non-tumor) class may be included if time allows. This approach leverages the abundance of unlabeled regions within pathology slides, reducing the dependence on costly manual annotations while maintaining diagnostic accuracy.

# Dataset

This project uses datasets from **The Cancer Genome Atlas (TCGA)** and the **National Cancer Institute**.

- [TCGA-LUAD](https://www.cancerimagingarchive.net/collection/tcga-luad/) – Lung adenocarcinoma whole-slide images  
- [TCGA-LUSC](https://www.cancerimagingarchive.net/collection/tcga-lusc/) – Lung squamous cell carcinoma whole-slide images  
- [GDC Data Portal](https://portal.gdc.cancer.gov/analysis_page?app=) – Provides slide-level labels

# Preprocessing
Navigate to data/preprocess:
- Run index.py to build index for LUAD and LUSC WSI classfication. This fetches TCGA slides from the GDC API, stores the metadata, and splits into train/val/test (75/15/10)
- Run download_index.py to download the TCGA WSI slides which will be saved under data/ data/raw/{project_id}/slides/{file_name}
- Run download_gdc_metadata.py to download the clinical and metadata in json formate from GDC API
