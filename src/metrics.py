import pandas as pd


def moving_average(series: pd.Series, window: int = 10):
    return series.rolling(window=window, min_periods=1).mean()
