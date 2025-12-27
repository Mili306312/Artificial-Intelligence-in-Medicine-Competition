# Artificial Intelligence in Medicine Competition – EEG Seizure Detection

**Short Description:**  
This project provides a comprehensive EEG-based seizure detection pipeline. It combines advanced signal processing, spectral and temporal feature extraction techniques, and a hybrid CNN–LSTM deep learning model with an attention mechanism. The result is a high-accuracy system that not only classifies seizures but also offers interpretable estimates of seizure onset and offset.

## 1. Introduction

This repository contains an end-to-end implementation for automated seizure detection on multi-channel EEG data, developed for the Artificial Intelligence in Medicine Competition. The goal is to build a robust, interpretable, and computationally efficient pipeline capable of detecting epileptic seizures in EEG recordings and pinpointing their start and end times with high precision.

The workflow is organized into three main phases:

- **Data Extraction & Preprocessing:** Handling raw EEG recordings, filtering noise, and preparing labeled datasets.  
- **Model Training:** Building and training a deep learning model (CNN + LSTM with attention) for seizure classification and onset/offset regression.  
- **Visualization & Explainability:** Generating visualizations and interpretability insights (e.g., attention heatmaps) to understand the model’s decisions.

This project demonstrates how classical biomedical signal processing can be integrated with modern deep learning architectures to analyze temporal physiological data (like EEG) for seizure detection in an interpretable way.

## 2. Key Features

**Signal Processing & Data Preparation**  
- **Noise Filtering:** Applied a 50 Hz IIR notch filter to remove power line interference (mains hum).  
- **Band-Pass Filtering:** Used a 0.5–70 Hz FIR band-pass filter to retain relevant EEG frequency bands while filtering out irrelevant frequencies.  
- **Resampling:** Standardized all EEG signals to a 250 Hz sampling rate for consistency across recordings.  
- **Segmentation:** Extracted a fixed-length segment (first 300 seconds) from each EEG recording to have uniform input duration.  
- **Data Storage:** Saved the cleaned EEG signals and their corresponding seizure labels (including onset and offset times) as NumPy `.npy` files for efficient loading.

**Feature Engineering**  
- **Spectral Features:** Computed power spectral density (PSD) using Welch’s method for each EEG segment. From the PSD, calculated band power features for key frequency ranges: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), and Gamma (30–70 Hz).  
- **Time-Domain Features:** Extracted statistical measures from the time-domain signal, including mean, standard deviation, root mean square (RMS), skewness, and kurtosis, to capture signal characteristics that differentiate seizure vs. non-seizure activity.

**Deep Learning Model – CNN + LSTM + Attention**  
- **Hybrid Architecture:** Implemented a neural network that combines Convolutional Neural Network (CNN) layers (to capture localized temporal patterns in the EEG) with Long Short-Term Memory (LSTM) layers (to learn long-term dependencies across time). An attention mechanism is integrated on top of the LSTM outputs to highlight the most informative time steps for the seizure prediction.  
- **Multi-Task Learning:** The model is trained to perform two related tasks simultaneously: (1) **Seizure Classification** (predicting whether a segment contains a seizure or not) and (2) **Onset/Offset Regression** (predicting the timing of seizure start and end within the segment).  
- **Loss Function:** Utilizes a combined loss to train these tasks together – Binary Cross-Entropy (BCE) for the classification and a weighted Mean Squared Error (MSE) for the onset/offset regression. The total loss is computed as **Total Loss = BCE + 0.5 × MSE**, balancing both objectives.

**Explainability (XAI)**  
- **Attention Analysis:** During inference, the attention weights from the model are extracted and upsampled to match the original signal length (in time). This produces a time series of attention values aligned with the EEG signal.  
- **Heatmap Overlay:** By overlaying the attention weights on the EEG waveform, the framework provides a heatmap visualization indicating which parts of the signal the model focused on when making its decision.  
- **Interpretation:** These visual explanations help identify the EEG regions most relevant to the model’s predictions, offering insight into why the model labeled a segment as seizure or non-seizure and how it determined the predicted onset/offset.

## 3. Installation

Before running the notebooks, ensure you have **Python 3.x** installed on your system. Then install the required dependencies. You can install all necessary packages (with specific versions as used in this project) using pip:

```bash
pip install numpy==1.24.0 pandas==1.5.3 scipy==1.9.3 tqdm==4.64.0 \
            torch>=1.10.0 matplotlib==3.6.3 seaborn==0.11.2 \
            scikit-learn==1.0.2 mne==1.3.1
