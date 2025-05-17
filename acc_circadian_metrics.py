"""Calculate circadian rhythm metrics from accelerometer data."""

import numpy as np
import pandas as pd
from scipy import fftpack, optimize
from datetime import timedelta
from typing import List, Dict, Union


def calculatePSD(
    e: pd.DataFrame,
    epochPeriod: int,
    fourierWithAcc: bool,
    labels: List[str],
    summary: Dict[str, Union[float, str]],
) -> None:
    """
    Calculate the power spectral density at 1 cycle/day from Fourier analysis.

    :param e: DataFrame with epoch data, must contain 'acc' column if fourierWithAcc is True,
              else activity state columns from labels.
    :param epochPeriod: Epoch size in seconds.
    :param fourierWithAcc: If True, use acceleration data; else use sleep activity state.
    :param labels: List of activity state labels (column names).
    :param summary: Dictionary to update with key 'PSD-1/day' for the PSD value.
    """
    if epochPeriod <= 0:
        raise ValueError("epochPeriod must be positive.")

    if fourierWithAcc:
        y = e['acc'].values
    else:
        # Binary signal: sleep = 1, others = 0, then centered to [-1, 1]
        y = (e[labels].idxmax(axis=1) == 'sleep').astype(int) * 2 - 1

    n = len(y)
    if n == 0:
        summary['PSD-1/day'] = 'NA_no_data'
        return

    # Compute FFT
    fft_y = fftpack.fft(y)
    freqs = fftpack.fftfreq(n, d=epochPeriod)

    # Frequency corresponding to 1 cycle/day (Hz)
    target_freq = 1 / (24 * 3600)

    # Find index closest to target_freq
    idx = np.argmin(np.abs(freqs - target_freq))

    # Power spectral density at 1 cycle/day (normalize by length)
    PSD = (np.abs(fft_y[idx]) ** 2) / n

    summary['PSD-1/day'] = PSD


def calculateFourierFreq(
    e: pd.DataFrame,
    epochPeriod: int,
    fourierWithAcc: bool,
    labels: List[str],
    summary: Dict[str, Union[float, str]],
) -> None:
    """
    Calculate the dominant frequency (cycles per day) from Fourier analysis.

    :param e: DataFrame with epoch data, must contain 'acc' column if fourierWithAcc is True,
              else activity state columns from labels.
    :param epochPeriod: Epoch size in seconds.
    :param fourierWithAcc: If True, use acceleration data; else use sleep activity state.
    :param labels: List of activity state labels.
    :param summary: Dictionary to update with key 'fourier-frequency-1/day' for dominant frequency.
    """
    if epochPeriod <= 0:
        raise ValueError("epochPeriod must be positive.")

    if fourierWithAcc:
        y = e['acc'].values
    else:
        y = (e[labels].idxmax(axis=1) == 'sleep').astype(int) * 2 - 1

    n = len(y)
    if n == 0:
        summary['fourier-frequency-1/day'] = 'NA_no_data'
        return

    fft_y = np.abs(fftpack.fft(y))
    freqs = fftpack.fftfreq(n, d=epochPeriod)

    # Consider positive frequencies only (excluding zero frequency)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_y[pos_mask]

    if len(pos_fft) == 0:
        summary['fourier-frequency-1/day'] = 'NA_no_positive_freqs'
        return

    k_max_idx = np.argmax(pos_fft)
    k_max = pos_freqs[k_max_idx] * n  # Corresponding "k" for minimize_scalar bracket

    def func(k):
        return -np.abs(np.sum(np.exp(-2j * np.pi * k * np.arange(n) / n) * y) / n)

    bracket_low = max(k_max - 1, 0)
    bracket_high = k_max + 1
    res = optimize.minimize_scalar(func, bracket=(bracket_low, bracket_high))

    freq_cpd = res.x / (n * epochPeriod / (24 * 3600))  # cycles per day

    summary['fourier-frequency-1/day'] = freq_cpd


def calculateM10L5(
    e: pd.DataFrame, epochPeriod: int, summary: Dict[str, Union[float, str]]
) -> None:
    """
    Calculate M10 L5 relative amplitude from average acceleration data.

    :param e: DataFrame with datetime index and 'acc' column.
    :param epochPeriod: Epoch size in seconds.
    :param summary: Dictionary to update with key 'M10L5-rel_amp' for relative amplitude.
    """
    if epochPeriod <= 0:
        raise ValueError("epochPeriod must be positive.")

    TEN_HOURS = int(10 * 3600 / epochPeriod)
    FIVE_HOURS = int(5 * 3600 / epochPeriod)

    if 'acc' not in e.columns:
        raise ValueError("Input DataFrame must contain 'acc' column.")

    if not isinstance(e.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    daily_acc = e['acc'].resample('1D').apply(list)

    if len(daily_acc) == 0:
        summary['M10L5-rel_amp'] = 'NA_too_few_days'
        return

    M10_values = []
    L5_values = []

    for day_acc in daily_acc.dropna():
        arr = np.array(day_acc)
        if len(arr) < max(TEN_HOURS, FIVE_HOURS):
            continue
        # Sliding window sums using convolution
        ten_hr_sums = np.convolve(arr, np.ones(TEN_HOURS), mode='valid')
        five_hr_sums = np.convolve(arr, np.ones(FIVE_HOURS), mode='valid')

        M10_values.append(ten_hr_sums.max() / TEN_HOURS)
        L5_values.append(five_hr_sums.min() / FIVE_HOURS)

    if len(M10_values) == 0 or len(L5_values) == 0:
        summary['M10L5-rel_amp'] = 'NA_too_few_days'
        return

    M10 = np.mean(M10_values)
    L5 = np.mean(L5_values)

    rel_amp = (M10 - L5) / (M10 + L5)

    summary['M10L5-rel_amp'] = rel_amp
