# Alzheimer's Disease Classification Using EEG Data

## Update: 
Link Google Colab for new data: https://colab.research.google.com/drive/1IX_4SG4O9QPjYyclkc1Jh1M-X1yegcPB?usp=sharing

This repository is a modification of the original repository by **tsyoshihara**. The work combines EEG signal processing with supervised machine learning methods to classify Alzheimer's disease patients.

## Dataset
The EEG dataset used in this project was sourced from Fiscon et al.’s 2014 paper and consists of 3 classes: **AD**: Alzheimer's disease, **MCI**: Mild Cognitive Impairment, **HC**: Healthy Control

### Data Processing
The data is preprocessed using a **Fast Fourier Transform (FFT)** to convert the EEG signals from the time domain to the frequency domain. Each sample contains EEG signals recorded from multiple electrodes and is transformed for classification.

## Classification Models
1. **Fisher’s Discriminant Analysis (FDA)**
2. **Relevance Vector Classifier (RVC)**
3. **Random Forest (RF)**
4. **Logistic Regression (LR)**

## Modifications and Improvements
This repository includes the following modifications and improvements to the original code:
- Updated packages to ensure compatibility with modern Python versions.
- Saved the visual results of classification (e.g., ROC curves, confusion matrices) as images for easy visualization.

## Results
The results from the classification models are saved in the `/new_image` folder as images and include: Confusion matrices, ROC curves for each classifier

## Acknowledgments
- Original repository: https://github.com/tsyoshihara/Alzheimer-s-Classification-EEG
- The dataset and preprocessing methods are based on the work by Fiscon et al. (2018) - **Fiscon, G., et al. (2018). Combining EEG signal processing with supervised methods for Alzheimer’s patients classification. BMC Medical Informatics and Decision Making.**
