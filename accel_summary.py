import sys
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from accelerometer_mvpa import accel_utils, classification, circadian


def getActivitySummary(
    epochFile,
    summary,
    activityClassification=True,
    timeZone='Europe/London',
    startTime=None,
    endTime=None,
    epochPeriod=30,
    stationaryStd=13,
    minNonWearDuration=60,
    mgCpLPA=45,
    mgCpMPA=100,
    mgCpVPA=400,
    activityModel="walmsley",
    intensityDistribution=False,
    imputation=True,
    psd=False,
    fourierFrequency=False,
    fourierWithAcc=False,
    m10l5=False
):
    """
    Main interface to compute summary metrics from accelerometer epoch data.
    Returns the cleaned data, activity labels, and populates the summary dict.
    """
    utils.toScreen("=== Summarizing ===")

    data = _loadData(epochFile, timeZone)
    data = _trimTimeRange(data, timeZone, startTime, endTime)
    _populateFileMetadata(data, summary)

    checkQuality(data, summary)

    data['acc'] = data['enmoTrunc'] * 1000

    data = resolveInterrupts(data, epochPeriod, summary)
    data = resolveNonWear(data, stationaryStd, minNonWearDuration, summary)

    labels = []
    if activityClassification:
        data, labels = classification.activityClassification(
            data, activityModel, mgCpLPA, mgCpMPA, mgCpVPA
        )

    if intensityDistribution:
        calculateECDF(data['acc'], summary)

    if any([psd, fourierFrequency, m10l5]):
        circadianData = imputeMissing(data[['acc'] + labels])
        if psd:
            circadian.calculatePSD(circadianData, epochPeriod, fourierWithAcc, labels, summary)
        if fourierFrequency:
            circadian.calculateFourierFreq(circadianData, epochPeriod, fourierWithAcc, labels, summary)
        if m10l5:
            circadian.calculateM10L5(circadianData, epochPeriod, summary)

    writeMovementSummaries(data, labels, summary)

    return data, labels


def _loadData(epochFile, timeZone):
    if isinstance(epochFile, pd.DataFrame):
        return epochFile
    return pd.read_csv(
        epochFile,
        index_col='time',
        parse_dates=['time'],
        date_parser=utils.date_parser
    )


def _trimTimeRange(data, timeZone, startTime, endTime):
    if startTime:
        data = data.loc[pd.Timestamp(startTime, tz=timeZone):]
    if endTime:
        data = data.loc[:pd.Timestamp(endTime, tz=timeZone)]
    if data.empty:
        print("No rows remaining after start/end time removal")
        sys.exit(-9)
    return data


def _populateFileMetadata(data, summary):
    startTime = data.index[0]
    endTime = data.index[-1]
    summary['file-startTime'] = utils.date_strftime(startTime)
    summary['file-endTime'] = utils.date_strftime(endTime)
    summary['file-firstDay(0=mon,6=sun)'] = startTime.weekday()


def checkQuality(data, summary):
    summary['totalReads'] = data['rawSamples'].sum().item()
    dstDiff = data.index[0].dst() - data.index[-1].dst()
    summary['quality-daylightSavingsCrossover'] = int(np.sign(dstDiff))
    summary['clipsBeforeCalibration'] = data['clipsBeforeCalibr'].sum().item()
    summary['clipsAfterCalibration'] = data['clipsAfterCalibr'].sum().item()


def resolveInterrupts(data, epochPeriod, summary):
    epochPeriod = pd.Timedelta(seconds=epochPeriod)
    gaps = data.index.to_series().diff() > epochPeriod
    summary['errs-interrupts-num'] = gaps.sum()
    summary['errs-interrupt-mins'] = data.index.to_series().diff()[gaps].sum().total_seconds() / 60
    data = data.asfreq(epochPeriod, fill_value=None)
    data['missing'] = data.isna().any(axis=1)
    return data


def resolveNonWear(data, stdTol, patience, summary):
    stdTol /= 1000.0
    stationary = (data[['xStd', 'yStd', 'zStd']] < stdTol).all(axis=1)
    group = ((stationary != stationary.shift()).cumsum().where(stationary))
    durations = group.groupby(group, dropna=True).apply(lambda g: g.index[-1] - g.index[0])
    longDurations = durations[durations > pd.Timedelta(minutes=patience)]
    nonWear = group.isin(longDurations.index)
    missing = nonWear | data['missing']
    data = data.mask(missing)
    data['missing'] = missing

    freq = to_offset(pd.infer_freq(data.index))
    epochInDays = pd.to_timedelta(freq).total_seconds() / (60 * 60 * 24)
    nonWearTime = missing.sum() * epochInDays
    wearTime = (len(data) - missing.sum()) * epochInDays

    isGoodCoverage = not missing.groupby(missing.index.hour).all().any()
    isGoodWearTime = wearTime >= 3

    summary['wearTime-numNonWearEpisodes(>1hr)'] = len(longDurations)
    summary['wearTime-overall(days)'] = wearTime
    summary['nonWearTime-overall(days)'] = nonWearTime
    summary['quality-goodWearTime'] = int(isGoodCoverage and isGoodWearTime)

    return data


def imputeMissing(data, extrapolate=True):
    if extrapolate:
        data = data.reindex(
            pd.date_range(
                data.index[0].floor('D'),
                data.index[-1].ceil('D'),
                freq=to_offset(pd.infer_freq(data.index)),
                closed='left',
                name='time'
            ),
            method='nearest',
            tolerance=pd.Timedelta('1m'),
            limit=1
        )

    def fillna(subframe):
        if isinstance(subframe, pd.Series):
            x = subframe.to_numpy()
            if np.isnan(x).any() and not np.isnan(x).all():
                x[np.isnan(x)] = np.nanmean(x)
                return x
            return subframe

    for keys in [
        [data.index.weekday, data.index.hour, data.index.minute],
        [data.index.weekday >= 5, data.index.hour, data.index.minute],
        [data.index.hour, data.index.minute]
    ]:
        data = data.groupby(keys).transform(fillna)

    return data


def calculateECDF(x, summary):
    bins = np.concatenate([
        np.arange(1, 21, 1),
        np.arange(25, 105, 5),
        np.arange(125, 525, 25),
        np.arange(600, 2100, 100)
    ])
    ecdf = [(b, np.nanmean(x < b)) for b in bins]
    for b, val in ecdf:
        summary[f'acc-ecdf-below{b}mg'] = val


def writeMovementSummaries(data, labels, summary):  # noqa: C901
    """Write overall summary stats for each activity type to summary dict

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics
    :param bool imputation: Impute missing data using data from other days around the same time

    :return: Write dict <summary> keys for each activity type 'overall-<avg/sd>',
        'week<day/end>-avg', '<day..>-avg', 'hourOfDay-<hr..>-avg',
        'hourOfWeek<day/end>-<hr..>-avg'
    :rtype: void
    """

    data = data.copy()
    data['wearTime'] = ~data['missing']
    freq = to_offset(pd.infer_freq(data.index))

    # Hours of activity for each recorded day
    epochInHours = pd.Timedelta(freq).total_seconds() / 3600
    cols = ['wearTime'] + labels
    dailyStats = (
        data[cols].astype('float')
        .groupby(data.index.date)
        .sum()
        * epochInHours
    ).reset_index(drop=True)

    for i, row in dailyStats.iterrows():
        for col in cols:
            summary[f'day{i}-recorded-{col}(hrs)'] = row.loc[col]

    # In the following, we resample, pad and impute the data so that we have a
    # multiple of 24h for the stats calculations
    tStart, tEnd = data.index[0], data.index[-1]
    cols = ['acc', 'wearTime'] + labels
    if 'MET' in data.columns:
        cols.append('MET')
    data = imputeMissing(data[cols].astype('float'))

    # Overall stats (no padding, i.e. only within recording period)
    overallStats = data[tStart:tEnd].apply(['mean', 'std'])
    for col in overallStats:
        summary[f'{col}-overall-avg'] = overallStats[col].loc['mean']
        summary[f'{col}-overall-sd'] = overallStats[col].loc['std']

    dayOfWeekStats = (
        data
        .groupby([data.index.weekday, data.index.hour])
        .mean()
    )
    dayOfWeekStats.index = dayOfWeekStats.index.set_levels(
        dayOfWeekStats
        .index.levels[0].to_series()
        .replace({0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'})
        .to_list(),
        level=0
    )
    dayOfWeekStats.index.set_names(['DayOfWeek', 'Hour'], inplace=True)

    # Week stats
    for col, value in dayOfWeekStats.mean().items():
        summary[f'{col}-week-avg'] = value

    # Stats by day of week (Mon, Tue, ...)
    for col, stats in dayOfWeekStats.groupby(level=0).mean().to_dict().items():
        for dayOfWeek, value in stats.items():
            summary[f'{col}-{dayOfWeek}-avg'] = value

    # Stats by hour of day
    for col, stats in dayOfWeekStats.groupby(level=1).mean().to_dict().items():
        for hour, value in stats.items():
            summary[f'{col}-hourOfDay-{hour}-avg'] = value

    # (not included but could be) Stats by hour of day AND day of week
    # for col, stats in dayOfWeekStats.to_dict().items():
    #     for key, value in stats.items():
    #         dayOfWeek, hour = key
    #         summary[f'{col}-hourOf{dayOfWeek}-{hour}-avg'] = value

    weekdayOrWeekendStats = (
        dayOfWeekStats
        .groupby([
            dayOfWeekStats.index.get_level_values('DayOfWeek').str.contains('sat|sun'),
            dayOfWeekStats.index.get_level_values('Hour')
        ])
        .mean()
    )
    weekdayOrWeekendStats.index = weekdayOrWeekendStats.index.set_levels(
        weekdayOrWeekendStats
        .index.levels[0].to_series()
        .replace({True: 'Weekend', False: 'Weekday'})
        .to_list(),
        level=0
    )
    weekdayOrWeekendStats.index.set_names(['WeekdayOrWeekend', 'Hour'], inplace=True)

    # Weekday/weekend stats
    for col, stats in weekdayOrWeekendStats.groupby(level=0).mean().to_dict().items():
        for weekdayOrWeekend, value in stats.items():
            summary[f'{col}-{weekdayOrWeekend.lower()}-avg'] = value

    # Stats by hour of day AND by weekday/weekend
    for col, stats in weekdayOrWeekendStats.to_dict().items():
        for key, value in stats.items():
            weekdayOrWeekend, hour = key
            summary[f'{col}-hourOf{weekdayOrWeekend}-{hour}-avg'] = value

    return




