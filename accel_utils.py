"""Provide generic utilities for other accelerometer_mvpa modules."""

import datetime
import json
import math
import os
import re
from collections import OrderedDict

import pandas as pd
from tqdm.auto import tqdm

DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
TIME_SERIES_COL = 'time'

def formatNum(num, decimalPlaces):
    return round(num, decimalPlaces)

def meanSDstr(mean, std, decimalPlaces):
    """
    Return a string of mean and standard deviation formatted to given decimal places.
    """
    return f"{formatNum(mean, decimalPlaces)} ({formatNum(std, decimalPlaces)})"

def meanCIstr(mean, std, n, decimalPlaces):
    """
    Return a string of mean and 95% confidence interval formatted to given decimal places.
    """
    stdErr = std / math.sqrt(n)
    lowerCI = mean - 1.96 * stdErr
    upperCI = mean + 1.96 * stdErr
    return f"{formatNum(mean, decimalPlaces)} ({formatNum(lowerCI, decimalPlaces)} - {formatNum(upperCI, decimalPlaces)})"

def toScreen(msg):
    """Print message with timestamp."""
    timeFormat = '%Y-%m-%d %H:%M:%S'
    print(f"\n{datetime.datetime.now().strftime(timeFormat)}\t{msg}")

def writeCmds(accDir, outDir, cmdsFile='list-of-commands.txt', accExt="cwa", cmdOptions="", filesCSV=None):
    """
    Generate a command list for processing accelerometer files.
    """
    accExt = accExt.lower()
    accDir = accDir.rstrip("/")
    outDir = outDir.rstrip("/")

    if filesCSV and filesCSV in os.listdir(accDir):
        filesCSV = pd.read_csv(os.path.join(accDir, filesCSV), index_col="fileName")
        filesCSV.index = accDir + "/" + filesCSV.index.astype('str')
        filePaths = filesCSV.index.to_numpy()
    else:
        filePaths = [
            os.path.join(root, file)
            for root, _, files in os.walk(accDir)
            for file in files
            if file.lower().endswith(tuple([accExt, accExt + ext for ext in [".gz", ".zip", ".bz2", ".xz"]]))
        ]

    with open(cmdsFile, 'w') as f:
        for filePath in filePaths:
            outputFolder = filePath.replace(accDir, outDir).split(".")[0]
            cmd = f"accProcess '{filePath}' --outputFolder '{outputFolder}' {cmdOptions}"
            if filesCSV is not None:
                cmdOptionsCSV = ' '.join([f"--{col} {filesCSV.loc[filePath, col]}" for col in filesCSV.columns])
                cmd += f" {cmdOptionsCSV}"
            f.write(cmd + '\n')
    print(f"List of commands written to {cmdsFile}")

def collateSummary(resultsDir, outputCsvFile="all-summary.csv"):
    """
    Merge all *-summary.json files in a directory into one summary CSV.
    """
    print(f"Scanning {resultsDir} for summary files...")
    sumfiles = [
        os.path.join(root, file)
        for root, _, files in os.walk(resultsDir)
        for file in files if file.lower().endswith("-summary.json")
    ]
    print(f"Found {len(sumfiles)} summary files...")

    jdicts = [json.load(open(file), object_pairs_hook=OrderedDict) for file in tqdm(sumfiles)]
    summary = pd.DataFrame.from_dict(jdicts)
    summary['eid'] = summary['file-name'].str.extract(r"([^/]+)\.\w+$")
    summary.to_csv(outputCsvFile, index=False)
    print(f"Summary for {len(summary)} participants written to: {outputCsvFile}")

def identifyUnprocessedFiles(filesCsv, summaryCsv, outputFilesCsv):
    """
    Identify and list files that do not have summary data.
    """
    fileList = pd.read_csv(filesCsv)
    summary = pd.read_csv(summaryCsv)

    output = fileList[~fileList['fileName'].isin(summary['file-name'])]
    output = output.loc[:, ~output.columns.str.startswith('Unnamed')]
    output.to_csv(outputFilesCsv, index=False)
    print(f"Reprocessing list for {len(output)} participants written to: {outputFilesCsv}")

def updateCalibrationCoefs(inputCsvFile, outputCsvFile):
    """
    Suggest new calibration coefficients for files with poor calibration.
    """
    d = pd.read_csv(inputCsvFile)
    goodCal = d[(d['quality-calibratedOnOwnData'] == 1) & (d['quality-goodCalibration'] == 1)].sort_values('file-startTime')
    badCal = d[(d['quality-calibratedOnOwnData'] == 1) & (d['quality-goodCalibration'] == 0)].sort_values('file-startTime')

    calCols = [
        'calibration-xOffset(g)', 'calibration-yOffset(g)', 'calibration-zOffset(g)',
        'calibration-xSlope(g)', 'calibration-ySlope(g)', 'calibration-zSlope(g)',
        'calibration-xTemp(C)', 'calibration-yTemp(C)', 'calibration-zTemp(C)',
        'calibration-meanDeviceTemp(C)'
    ]

    counts = {"previousUses": 0, "nextUses": 0, "noOtherUses": 0}
    with open(outputCsvFile, 'w') as f:
        f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
        for _, row in badCal.iterrows():
            fileName, device, startTime = row['file-name'], int(row['file-deviceID']), row['file-startTime']
            prev = goodCal[(goodCal['file-deviceID'] == device) & (goodCal['file-startTime'] < startTime)].tail(1)
            if not prev.empty:
                values = prev[calCols].iloc[0]
                counts["previousUses"] += 1
            else:
                next_ = goodCal[(goodCal['file-deviceID'] == device) & (goodCal['file-startTime'] > startTime)].head(1)
                if next_.empty:
                    print(f"No other uses for device {device}: {fileName}")
                    counts["noOtherUses"] += 1
                    continue
                values = next_[calCols].iloc[0]
                counts["nextUses"] += 1

            offset = ' '.join(map(str, values[:3]))
            slope = ' '.join(map(str, values[3:6]))
            temp = ' '.join(map(str, values[6:9]))
            meanTemp = str(values[9])
            f.write(f"{fileName},{offset},{slope},{temp},{meanTemp}\n")

    print(f"previousUses: {counts['previousUses']}, nextUses: {counts['nextUses']}, noOtherUses: {counts['noOtherUses']}")
    print(f"Reprocessing for {counts['previousUses'] + counts['nextUses']} participants written to: {outputCsvFile}")

def writeFilesWithCalibrationCoefs(inputCsvFile, outputCsvFile):
    """
    Write files.csv with calibration coefs from summary CSV.
    """
    d = pd.read_csv(inputCsvFile)
    calCols = [
        'calibration-xOffset(g)', 'calibration-yOffset(g)', 'calibration-zOffset(g)',
        'calibration-xSlope(g)', 'calibration-ySlope(g)', 'calibration-zSlope(g)',
        'calibration-xTemp(C)', 'calibration-yTemp(C)', 'calibration-zTemp(C)',
        'calibration-meanDeviceTemp(C)'
    ]

    with open(outputCsvFile, 'w') as f:
        f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
        for _, row in d.iterrows():
            values = row[calCols]
            offset = ' '.join(map(str, values[:3]))
            slope = ' '.join(map(str, values[3:6]))
            temp = ' '.join(map(str, values[6:9]))
            meanTemp = str(values[9])
            f.write(f"{row['file-name']},{offset},{slope},{temp},{meanTemp}\n")
    print(f"Files with calibration info written to: {outputCsvFile}")
