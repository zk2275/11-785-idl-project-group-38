# Blood Pressure Estimation from ECG and PPG Signals

**Author:** Zhuodiao Kuang
**Date:** October 24, 2025

## Project Overview

This project, contained in a Google Colab notebook, implements and evaluates two baseline machine learning models for estimating Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) from raw Electrocardiogram (ECG) and Photoplethysmogram (PPG) signals. The data is processed from large zip archives and fed into two different models: a classical Linear Regression model based on engineered features and a deep learning (LSTM) model based on signal sequences. The performance of both models is evaluated against the British Hypertension Society (BHS) Grade A standard.

## Features

* **Data Processing:** Loads and processes large `.zip` archives (`train.zip`, `val.zip`, `test.zip`) directly from Google Drive.
* **Two Model Baselines:** Implements two distinct approaches for comparison:
    1.  **Baseline 1 (Linear Regression):** Extracts signal-processing features (e.g., Pulse Transit Time (PTT), Heart Rate (HR), PPG Amplitude) to train a `sklearn.linear_model.LinearRegression` model.
    2.  **Baseline 2 (LSTM):** Uses raw, normalized signal windows (250 samples) to train a `Bidirectional(LSTM)` deep learning model.
* **Evaluation:** Calculates the Mean Absolute Error (MAE) and Standard Deviation (SD) of the prediction errors for both SBP and DBP, comparing them to the BHS Grade A requirements ($\le 5$ mmHg MAE, $\le 8$ mmHg SD).
* **Debug Mode:** Includes a `DEBUG_MODE` flag. When set to `True`, the script processes only a small subset of files (`DEBUG_FILE_LIMIT`) to allow for rapid end-to-end pipeline testing without waiting hours for the full dataset to load.

## Requirements

The notebook is designed for Google Colab and requires the following main Python libraries:
* `tensorflow` (for Keras/LSTM)
* `scikit-learn` (for Linear Regression and `StandardScaler`)
* `pandas` (for reading CSVs)
* `numpy` (for data manipulation)
* `scipy` (for `find_peaks` in signal processing)
* `google.colab` (for mounting Google Drive)

## Data Preprocessing

Download the raw dataset **directly from the official MIMIC-III website** following their access requirements and instructions.  

1. Initial inspection & filtering (`basic stat and clean data.ipynb`): List and verify the downloaded files/variables, and filter the records to retain the three target data types used in our study.

2. Pairwise correlation (`simple corr.ipynb`): computes pairwise correlation coefficients among the three selected data types to quantify their relationships and provide a quick idea on signal relations.

3. Subject-wise split & temporal slicing (`split_dataset_split time.ipynb`): partitions the data by subject into three subsets and performs slicing on each record.

## How to Run

1.  **Upload Data:** Upload your `train.zip`, `val.zip`, and `test.zip` files to a folder in your Google Drive (e.g., `/MyDrive/11785FinalData/`).
2.  **Open in Colab:** Open the `colab_baseline_models (1).ipynb` notebook in Google Colab.
3.  **Mount Drive:** Run the second cell (`# === 1. Mount Google Drive ===`) to connect your Google Drive.
4.  **Edit Data Paths:** In the cell with the comment `# !!! EDIT THESE PATHS !!!`, update the `train_zip_path`, `val_zip_path`, and `test_zip_path` variables to match the exact location of your data in Google Drive.
5.  **Run All:** Run all the cells in the notebook sequentially.
    * Cell 3 defines all data loading and processing functions.
    * Cell 5 runs the Linear Regression baseline.
    * Cell 6 runs the LSTM baseline.

### Using Debug Mode

To quickly test if your data paths and model code are working, set `DEBUG_MODE = True` in the "Baseline 1: Linear Regression" cell. This will run the entire process on only 100 files, which should complete in under a minute. Once you confirm it works, set `DEBUG_MODE = False` to run the full experiment.

## Model Details

### Baseline 1: Linear Regression

* **Data:** Uses `mode='features'`.
* **Feature Engineering:** The `get_features_and_labels` function is used to find ECG R-peaks and PPG peaks to calculate:
    * Pulse Transit Time (PTT)
    * Heart Rate (HR)
    * PPG Amplitude
* **Model:** A `StandardScaler` is applied to the features, which are then used to train a simple `LinearRegression` model.

### Baseline 2: LSTM Model

* **Data:** Uses `mode='sequence'`.
* **Sequence Preparation:** The `create_sequences` function cuts the ECG and PPG signals into overlapping windows of 250 samples (2 seconds) with a step of 125 samples (1 second). The signals in each window are individually normalized.
* **Model:** A Keras `Sequential` model is used, with the core being a `Bidirectional(LSTM(64))` layer, followed by two `Dense` layers for regression.

## Results

Based on the execution logs, the models performed as follows:

* **Linear Regression (Debug Run):** The debug run on 100 files completed successfully, proving the data pipeline was fixed. The results (SBP MAE: 20.86 mmHg, DBP MAE: 12.66 mmHg) were not representative due to the tiny dataset but confirmed the code's functionality.
* **LSTM Model (Full Run):** The full run on over 1.7 million sequences achieved a **SBP MAE of 9.52 mmHg** and a **DBP MAE of 6.17 mmHg**.

**Conclusion:** Neither baseline model met the strict BHS Grade A standard, suggesting that more advanced architectures or features are necessary for this task.
