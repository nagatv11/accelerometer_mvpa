## Summary

- The code is designed to train and evaluate accelerometer-based activity classifiers.  
- It includes HMM smoothing for time series label consistency.  
- There is support for different models with model-specific corrections.  
- It saves and loads models in tar archives for portability.  
- It includes utility functions to classify raw acceleration into activity categories based on thresholds.  
- Evaluation is participant-based with detailed report generation.


### 1. **train_hmm_parameters**  
Estimates Hidden Markov Model (HMM) prior and transition probabilities from labeled data.

- **Inputs:**  
  - `y_proba`: classifier probabilities (not used here, but might be used in an extended version)  
  - `y_true`: true labels as a 1D array  
  - `labels`: list of possible labels  
  - `uniform_prior`: whether to assume uniform priors or estimate from data

- **Outputs:**  
  - Dictionary with `prior` (initial state probabilities) and `trans` (transition matrix)

- **How it works:**  
  Counts how often each label occurs at the start of sequences (to estimate priors) and how often transitions happen between labels (to estimate transition probabilities). If `uniform_prior=True`, it sets equal probabilities for all states.

---

### 2. **_apply_hmm_viterbi**  
Applies the Viterbi algorithm to smooth a sequence of predicted labels.

- **Inputs:**  
  - `predictions`: raw predicted labels (1D numpy array)  
  - `hmm_params`: dictionary with `prior` and `trans` matrices

- **Outputs:**  
  - Smoothed labels as a pandas Series

- **How it works:**  
  Converts labels to indices, computes log probabilities, runs the dynamic programming to find the most likely hidden state sequence given the priors and transitions, then converts back to labels.

---

### 3. **remove_spurious_sleep_epochs**  
Fixes isolated single epochs of one activity surrounded by another, based on model-specific rules.

- **Inputs:**  
  - `predicted_labels`: pandas Series of predicted activity labels  
  - `model_name`: either `"walmsley"` or `"chan"` — selects the correction rule

- **Outputs:**  
  - Corrected label Series

- **How it works:**  
  Checks if an epoch is sandwiched by the same label on both sides but different from the middle epoch, then replaces the middle one with a smoother label based on the correction map for that model.

---

### 4. **classify_by_cutpoints**  
Classifies activity based on ENMO acceleration cutpoints.

- **Inputs:**  
  - `enmo_series`: pandas Series with ENMO values  
  - `cutpoints`: dict with keys `'LPA'`, `'MPA'`, `'VPA'` and numeric thresholds  
  - `valid_mask`: optional boolean mask for valid epochs

- **Outputs:**  
  - One-hot encoded DataFrame of categories (`sleep`, `LPA`, `MPA`, `VPA`, `vigorous`)

- **How it works:**  
  Assigns categories by binning ENMO values according to the thresholds, treating invalid or zero ENMO as `'sleep'`. Returns one-hot encoded DataFrame with all expected activity columns.

---

### 5. **_extract_file_from_tar & _resolve_model_path**  
Helper functions for dealing with model files packaged as tar archives.

- `_extract_file_from_tar`: extracts a specific file from a tarball into a temporary directory  
- `_resolve_model_path`: resolves model file paths by alias or direct path

---

### 6. **_save_model_to_tar**  
Saves the classifier, labels, features, HMM parameters, and MET estimates into a tar archive.

- Used for packaging trained models for distribution or future loading.

---

### 7. **_apply_chan_adjustments**  
Specific logic for the `"chan"` model to relabel `'sedentary'` epochs to `'light'` activity.

---

### 8. **_evaluate_model_on_test_data**  
Runs predictions on test participants and saves classification reports.

- Uses the HMM smoothing on raw predictions before evaluation.  
- Saves reports per participant.

---

### 9. **_perform_cross_validation**  
Performs participant-wise cross-validation on the dataset.

- Splits participants into folds, trains classifier on train folds, evaluates on test folds.  
- Trains HMM parameters on training data only.

---
