"""Building machine learning models to stratify accelerometer-derived physical activity.

Uses a Balanced Random Forest classifier combined with a Hidden Markov Model (HMM) for
temporal smoothing. Supports training new models and applying pre-trained models.

Features:
- Loading and saving models packaged as tar archives.
- Cut-point based activity classification.
- HMM smoothing with Viterbi algorithm.
- Handling spurious sleep epochs.
"""

import os
import json
import logging
import tempfile
import tarfile
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, f1_score, make_scorer
import sklearn.metrics as metrics

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def classify_activity_epochs(
    epoch_data: pd.DataFrame,
    model_path_or_name: str = "walmsley",
    cutpoint_thresholds: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Classify activity states from epoch feature data using a pre-trained model.

    Args:
        epoch_data (pd.DataFrame): Dataframe of processed epoch data with features.
        model_path_or_name (str): Path to tar file containing the model or model name alias.
        cutpoint_thresholds (Optional[Dict[str, float]]): Thresholds for cutpoint classification
            {'LPA': float, 'MPA': float, 'VPA': float} in mg units (default thresholds used if None).

    Returns:
        Tuple[pd.DataFrame, List[str]]: Tuple of the original dataframe enriched with
        one-hot encoded activity labels and the full list of activity state labels.
    """
    model_path = _resolve_model_path(model_path_or_name)

    # Load model components from tar
    feature_columns = joblib.load(_extract_file_from_tar(model_path, 'featureCols'))
    model = joblib.load(_extract_file_from_tar(model_path, 'model'))
    hmm_params = joblib.load(_extract_file_from_tar(model_path, 'hmmParams'))
    labels = joblib.load(_extract_file_from_tar(model_path, 'labels')).tolist()

    # Prepare feature matrix and handle invalid rows
    X = epoch_data[feature_columns].to_numpy()
    valid_rows = np.isfinite(X).all(axis=1)
    n_invalid = len(epoch_data) - np.sum(valid_rows)
    if n_invalid > 0:
        logger.warning(f"{n_invalid} rows contain NaN or infinite values and will be skipped.")

    # Predict raw classes
    predictions = pd.Series(index=epoch_data.index, dtype=object)
    if valid_rows.any():
        raw_preds = model.predict(X[valid_rows])
        predictions.loc[valid_rows] = _apply_hmm_viterbi(raw_preds, hmm_params)

    # Optional model-specific logic (e.g., 'chan' model)
    if model_path_or_name.lower() == 'chan':
        _apply_chan_adjustments(epoch_data, predictions, labels)

    # Remove spurious short sleep epochs
    predictions = remove_spurious_sleep_epochs(predictions, model_path_or_name)

    # One-hot encode predicted activity labels
    one_hot_labels = pd.DataFrame(
        data=(predictions.to_numpy()[:, None] == labels).astype(float),
        index=epoch_data.index,
        columns=labels,
    )
    epoch_data = epoch_data.join(one_hot_labels)

    # Add predicted MET values if available
    met_values = joblib.load(_extract_file_from_tar(model_path, 'METs'))
    if met_values is not None:
        epoch_data['MET'] = predictions.replace(met_values)

    # Perform cutpoint classification on non-sleep epochs
    cutpoints = cutpoint_thresholds or {'LPA': 45/1000, 'MPA': 100/1000, 'VPA': 400/1000}
    non_sleep_mask = ~(predictions == 'sleep')  # Avoid NaNs comparison issues
    cutpoint_one_hot = classify_by_cutpoints(
        enmo_series=epoch_data['enmoTrunc'],
        cutpoints=cutpoints,
        valid_mask=non_sleep_mask
    )
    epoch_data = epoch_data.join(cutpoint_one_hot)

    # Append cutpoint labels to overall labels list
    all_labels = labels + cutpoint_one_hot.columns.tolist()

    return epoch_data, all_labels


def train_activity_classification_model(
    training_csv_path: str,
    feature_list_path: str = "activityModels/features.txt",
    output_dir: str = "model/",
    label_column: str = "label",
    participant_column: str = "participant",
    annotation_column: str = "annotation",
    met_column: Optional[str] = "MET",
    n_trees: int = 1000,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    cross_validation_folds: Optional[int] = None,
    test_participants: Optional[List[str]] = None,
    n_jobs: int = 1
) -> None:
    """
    Train an activity classification model from labeled epoch data and save the model.

    Args:
        training_csv_path (str): Path to CSV file with training data.
        feature_list_path (str): Path to text file listing feature columns to use.
        output_dir (str): Directory to save the trained model and reports.
        label_column (str): Column name containing activity labels.
        participant_column (str): Column name with participant IDs.
        annotation_column (str): Column name with textual annotations.
        met_column (Optional[str]): Column name for MET values. If None, METs are not estimated.
        n_trees (int): Number of trees for Random Forest.
        max_depth (Optional[int]): Maximum depth of trees.
        min_samples_leaf (int): Minimum samples per leaf.
        cross_validation_folds (Optional[int]): Number of folds for cross-validation. If None, no CV.
        test_participants (Optional[List[str]]): Participant IDs reserved for testing.
        n_jobs (int): Number of parallel jobs.

    Returns:
        None. Model and reports are saved to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_columns = np.loadtxt(feature_list_path, dtype=str)
    required_columns = [participant_column, label_column, annotation_column] + feature_columns.tolist()
    if met_column:
        required_columns.append(met_column)

    data = pd.read_csv(training_csv_path, usecols=required_columns)
    with pd.option_context('mode.use_inf_as_null', True):
        data.dropna(inplace=True)

    # Split into train/test sets if specified
    if test_participants:
        test_participants_set = set(test_participants)
        test_data = data[data[participant_column].isin(test_participants_set)].copy()
        train_data = data[~data[participant_column].isin(test_participants_set)].copy()
    else:
        train_data = data
        test_data = None

    X_train = train_data[feature_columns].to_numpy()
    y_train = train_data[label_column].to_numpy()
    groups = train_data[participant_column].to_numpy()

    # Define classifier factory function
    def create_classifier(**kwargs) -> BalancedRandomForestClassifier:
        return BalancedRandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            replacement=True,
            sampling_strategy='not minority',
            random_state=42,
            **kwargs
        )

    logger.info("Training Balanced Random Forest model...")
    model = create_classifier(n_jobs=n_jobs, verbose=1).fit(X_train, y_train)
    model.verbose = 0

    labels = model.classes_

    logger.info("Generating cross-validated predictions for HMM training...")
    njobs_per_cv = min(2, n_jobs)
    y_proba_cv = cross_val_predict(
        create_classifier(n_jobs=njobs_per_cv),
        X_train,
        y_train,
        groups=groups,
        cv=10,
        n_jobs=max(1, n_jobs // njobs_per_cv),
        method="predict_proba",
        verbose=3
    )

    logger.info("Training Hidden Markov Model parameters...")
    hmm_params = train_hmm_parameters(y_proba_cv, y_train, labels)

    met_estimates = None
    if met_column:
        met_estimates = {
            label: train_data.loc[y_train == label].groupby(annotation_column)[met_column].mean().mean()
            for label in labels
        }

    model_tar_path = os.path.join(output_dir, 'model.tar')
    _save_model_to_tar(
        model_tar_path,
        model=model,
        labels=labels,
        feature_columns=feature_columns,
        hmm_params=hmm_params,
        met_estimates=met_estimates
    )
    logger.info(f"Trained model saved to: {model_tar_path}")

    # Evaluate on test participants if provided
    if test_data is not None and not test_data.empty:
        _evaluate_model_on_test_data(
            model, hmm_params, test_data, feature_columns, label_column,
            participant_column, output_dir
        )

all data
if cross\_validation\_folds:
\_perform\_cross\_validation(
create\_classifier, data, feature\_columns, label\_column,
participant\_column, cross\_validation\_folds, n\_jobs, output\_dir
)

def train\_hmm\_parameters(
y\_proba: np.ndarray,
y\_true: np.ndarray,
labels: Union\[List\[str], np.ndarray],
uniform\_prior: bool = True
) -> Dict\[str, np.ndarray]:
"""
Train HMM prior and transition probabilities using labeled data.

```
Args:
    y_proba (np.ndarray): Predicted class probabilities from classifier.
    y_true (np.ndarray): True class labels.
    labels (List[str] or np.ndarray): List of class labels.
    uniform_prior (bool): Whether to use uniform prior probabilities or estimate from data.

Returns:
    Dict[str, np.ndarray]: Dictionary with keys 'prior' and 'trans' containing
        prior distribution and transition matrix.
"""
label_to_index = {label: idx for idx, label in enumerate(labels)}
n_labels = len(labels)

# Initialize counts
prior_counts = np.zeros(n_labels)
transition_counts = np.zeros((n_labels, n_labels))

# Map true labels to indices
y_true_indices = np.array([label_to_index[label] for label in y_true])

# Estimate prior counts from first label in each sequence (assuming grouped sequences)
unique_groups, group_start_indices = np.unique(y_true_indices, return_index=True)
for start_idx in group_start_indices:
    prior_counts[y_true_indices[start_idx]] += 1

if not uniform_prior:
    prior_probs = prior_counts / prior_counts.sum()
else:
    prior_probs = np.full(n_labels, 1.0 / n_labels)

# Estimate transition counts
for i in range(len(y_true_indices) - 1):
    current_label = y_true_indices[i]
    next_label = y_true_indices[i + 1]
    transition_counts[current_label, next_label] += 1

# Normalize transition matrix rows to probabilities
trans_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
trans_probs = np.nan_to_num(trans_probs)  # Replace NaNs from division by zero

return {'prior': prior_probs, 'trans': trans_probs}
```

def \_apply\_hmm\_viterbi(
predictions: np.ndarray,
hmm\_params: Dict\[str, np.ndarray]
) -> pd.Series:
"""
Apply Viterbi algorithm to smooth predicted labels using HMM parameters.

```
Args:
    predictions (np.ndarray): Raw predicted labels from classifier.
    hmm_params (dict): Dictionary containing HMM 'prior' and 'trans' matrices.

Returns:
    pd.Series: Smoothed predicted labels as a pandas Series.
"""
labels = np.unique(predictions)
label_to_idx = {label: i for i, label in enumerate(labels)}
idx_to_label = {i: label for label, i in label_to_idx.items()}

states = np.array([label_to_idx[label] for label in predictions])

n_states = len(labels)
T = len(states)

prior = hmm_params['prior']
trans = hmm_params['trans']

SMALL_NUMBER = 1e-16
log_prior = np.log(prior + SMALL_NUMBER)
log_trans = np.log(trans + SMALL_NUMBER)

# Initialize arrays for dynamic programming
dp = np.full((T, n_states), -np.inf)
prev = np.zeros((T, n_states), dtype=int)

dp[0, :] = log_prior

for t in range(1, T):
    for j in range(n_states):
        prob = dp[t-1, :] + log_trans[:, j]
        prev[t, j] = np.argmax(prob)
        dp[t, j] = np.max(prob)

# Backtrack to find most likely state sequence
states_seq = np.zeros(T, dtype=int)
states_seq[-1] = np.argmax(dp[-1, :])
for t in reversed(range(1, T)):
    states_seq[t-1] = prev[t, states_seq[t]]

smoothed_labels = pd.Series([idx_to_label[idx] for idx in states_seq])
return smoothed_labels
```

def remove\_spurious\_sleep\_epochs(
predicted\_labels: pd.Series,
model\_name: str = "walmsley"
) -> pd.Series:
"""
Correct spurious isolated sleep epochs according to defined mapping rules.

```
Args:
    predicted_labels (pd.Series): Series of predicted activity labels.
    model_name (str): Name of the classification model used.

Returns:
    pd.Series: Corrected predicted labels.
"""
if model_name.lower() == "walmsley":
    correction_map = {
        "sleep": "sedentary",
        "sedentary": "light",
        "light": "moderate",
        "moderate": "vigorous",
        "vigorous": "moderate"
    }
elif model_name.lower() == "chan":
    correction_map = {
        "sleep": "sedentary",
        "sedentary": "light",
        "light": "moderate",
        "moderate": "light",
        "vigorous": "moderate"
    }
else:
    return predicted_labels

corrected_labels = predicted_labels.copy()
for idx in range(1, len(corrected_labels) - 1):
    if (
        corrected_labels.iloc[idx - 1] == corrected_labels.iloc[idx + 1]
        and corrected_labels.iloc[idx] != corrected_labels.iloc[idx - 1]
    ):
        corrected_labels.iloc[idx] = correction_map.get(corrected_labels.iloc[idx], corrected_labels.iloc[idx])

return corrected_labels
```

def classify\_by\_cutpoints(
enmo\_series: pd.Series,
cutpoints: Dict\[str, float],
valid\_mask: Optional\[pd.Series] = None
) -> pd.DataFrame:
"""
Classify epochs into activity categories based on cutpoints on ENMO values.

```
Args:
    enmo_series (pd.Series): ENMO (Euclidean Norm Minus One) acceleration data.
    cutpoints (dict): Dictionary with keys 'LPA', 'MPA', 'VPA' and thresholds in g.
    valid_mask (Optional[pd.Series]): Boolean mask indicating which epochs to classify.

Returns:
    pd.DataFrame: One-hot encoded DataFrame with columns ['sleep', 'LPA', 'MPA', 'VPA'].
"""
if valid_mask is None:
    valid_mask = pd.Series(True, index=enmo_series.index)

categories = pd.Series("sleep", index=enmo_series.index)

categories[valid_mask] = pd.cut(
    enmo_series[valid_mask],
    bins=[-np.inf, cutpoints['LPA'], cutpoints['MPA'], cutpoints['VPA'], np.inf],
    labels=['LPA', 'MPA', 'VPA', 'vigorous']
).astype(str)

# Sleep category for invalid mask or zero ENMO
categories.loc[~valid_mask] = 'sleep'
categories.loc[enmo_series == 0] = 'sleep'

one_hot = pd.get_dummies(categories, prefix='', prefix_sep='')
# Ensure all expected columns are present
for col in ['sleep', 'LPA', 'MPA', 'VPA', 'vigorous']:
    if col not in one_hot.columns:
        one_hot[col] = 0
return one_hot
```

def \_extract\_file\_from\_tar(tar\_path: str, filename: str) -> str:
"""
Extract a file from a tar archive and return its path.

```
Args:
    tar_path (str): Path to tar archive.
    filename (str): File name inside tar archive.

Returns:
    str: Path to extracted file.
"""
temp_dir = tempfile.mkdtemp()
with tarfile.open(tar_path, "r") as tar:
    tar.extract(filename, path=temp_dir)
return os.path.join(temp_dir, filename)
```

def \_resolve\_model\_path(model\_name\_or\_path: str) -> str:
"""
Resolve model path from a name or a path.

```
Args:
    model_name_or_path (str): Model alias or path.

Returns:
    str: Absolute path to model tar archive.
"""
if os.path.exists(model_name_or_path):
    return model_name_or_path
# Define known model name aliases here, e.g.
known_models = {
    "walmsley": "activityModels/walmsley/model.tar",
    "chan": "activityModels/chan/model.tar"
}
if model_name_or_path.lower() in known_models:
    return known_models[model_name_or_path.lower()]
else:
    raise FileNotFoundError(f"Model '{model_name_or_path}' not found.")
```

def \_save\_model\_to\_tar(
tar\_path: str,
model,
labels: List\[str],
feature\_columns: List\[str],
hmm\_params: Dict\[str, np.ndarray],
met\_estimates: Optional\[Dict\[str, float]] = None
) -> None:
"""
Save model components to a tar archive.

```
Args:
    tar_path (str): Output tar file path.
    model: Trained classifier object.
    labels (List[str]): List of activity labels.
    feature_columns (List[str]): List of feature column names.
    hmm_params (Dict[str, np.ndarray]): HMM prior and transition matrices.
    met_estimates (Optional[Dict[str, float]]): Mean MET estimates per label.

Returns:
    None
"""
with tempfile.TemporaryDirectory() as tmpdir:
    # Save each component
    joblib.dump(feature_columns, os.path.join(tmpdir, "featureCols"))
    joblib.dump(model, os.path.join(tmpdir, "model"))
    joblib.dump(hmm_params, os.path.join(tmpdir, "hmmParams"))
    joblib.dump(labels, os.path.join(tmpdir, "labels"))
    joblib.dump(met_estimates, os.path.join(tmpdir, "METs"))

    with tarfile.open(tar_path, "w") as tar:
        for fname in ["featureCols", "model", "hmmParams", "labels", "METs"]:
            tar.add(os.path.join(tmpdir, fname), arcname=fname)
```

def \_apply\_chan\_adjustments(
epoch\_data: pd.DataFrame,
predictions: pd.Series,
labels: List\[str]
) -> None:
"""
Specific logic for Chan model: set epochs labeled as 'sedentary' to 'light' activity.

```
Args:
    epoch_data (pd.DataFrame): Epoch data (unused but kept for potential use).
    predictions (pd.Series): Predicted labels, modified in place.
    labels (List[str]): List of labels.

Returns:
    None
"""
predictions.loc[predictions == 'sedentary'] = 'light'
```

def \_evaluate\_model\_on\_test\_data(
model,
hmm\_params: Dict\[str, np.ndarray],
test\_data: pd.DataFrame,
feature\_columns: List\[str],
label\_column: str,
participant\_column: str,
output\_dir: str
) -> None:
"""
Evaluate the model on test participants and save performance reports.

```
Args:
    model: Trained classifier.
    hmm_params (dict): HMM parameters.
    test_data (pd.DataFrame): Test set data.
    feature_columns (List[str]): Features to use.
    label_column (str): Ground truth label column.
    participant_column (str): Participant ID column.
    output_dir (str): Directory to save reports.

Returns:
    None
"""
participants = test_data[participant_column].unique()
reports = []
for participant in participants:
    p_data = test_data[test_data[participant_column] == participant]
    X_test = p_data[feature_columns].to_numpy()
    y_true = p_data[label_column].to_numpy()

    y_pred_raw = model.predict(X_test)
    y_pred = _apply_hmm_viterbi(y_pred_raw, hmm_params)

    f1 = f1_score(y_true, y_pred, average='macro')
    report_str = classification_report
    reports.append((participant, f1, report_str))

        # Save report file
        report_file = os.path.join(output_dir, f"{participant}_classification_report.txt")
        with open(report_file, "w") as f:
            f.write(report_str)

def _perform_cross_validation(
    create_classifier,
    data: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    participant_column: str,
    folds: int,
    n_jobs: int,
    output_dir: str
) -> None:
    """
    Perform participant-wise cross-validation training and evaluation.

    Args:
        create_classifier (callable): Function returning a fresh classifier instance.
        data (pd.DataFrame): Full dataset.
        feature_columns (List[str]): Feature column names.
        label_column (str): Label column name.
        participant_column (str): Participant ID column.
        folds (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel jobs.
        output_dir (str): Output directory for reports.

    Returns:
        None
    """
    participants = data[participant_column].unique()
    np.random.shuffle(participants)

    fold_size = len(participants) // folds
    fold_indices = [participants[i * fold_size:(i + 1) * fold_size] for i in range(folds)]

    for i, test_participants in enumerate(fold_indices):
        train_participants = np.setdiff1d(participants, test_participants)
        train_data = data[data[participant_column].isin(train_participants)]
        test_data = data[data[participant_column].isin(test_participants)]

        clf = create_classifier()
        clf.fit(train_data[feature_columns], train_data[label_column])

        y_train_pred = clf.predict(train_data[feature_columns])
        hmm_params = train_hmm_parameters(
            y_proba=None,
            y_true=train_data[label_column].to_numpy(),
            labels=np.unique(train_data[label_column]),
            uniform_prior=True
        )

        _evaluate_model_on_test_data(
            clf, hmm_params, test_data,
            feature_columns, label_column, participant_column, output_dir
        )

        print(f"Fold {i+1}/{folds} completed.")
