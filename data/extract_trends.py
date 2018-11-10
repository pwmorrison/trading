import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 150)

"""
Extracts the trends of a given nature from a dataset.
Useful for creating ML targets.
"""


class Feature:
    def __init__(self, feature_name, relative_day, column_name):
        self.feature_name = feature_name
        self.relative_day = relative_day
        self.column_name = column_name

    def extract_feature(self, df):
        """
        Extracts the feature from the given DataFrame. The feature is shifted, so that it aligns with the days in the
        DataFrame.
        """
        feature = df[self.column_name]
        # Since the feature is in the past, shift it forward, so that it aligns with the relevant day.
        feature = feature.shift(self.relative_day)

        return feature


def extract_trend_features(df, features):
    # Create a new column for each feature.
    for feature in features:
        feature_column = feature.extract_feature(df)
        df[feature.feature_name] = feature_column

    return df


def extract_trend_target(df, n_days, increase_thresh, start="open", end="close"):
    """
    Extracts the target to indicate if a trend is occurring.
    The target is positive if the price increases by a certain amount in the given timeframe.
    :param df:
    :param n_days: the timeframe of the increase
    :param increase_thresh: the amount by which the price needs to increase
    :param start: the column at which we buy, on the first day
    :param end: the column at which sell, on the last day
    :return: a DataFrame containing the target
    """
    # Shift the end series backwards by the given timeframe, so it aligns with the start series.
    df['end_shift'] = df[end].shift(-n_days)
    # The diff (money made/lost if the period were traded).
    df['diff'] = df['end_shift'] - df[start]
    df['diff_percent'] = df['diff'] / df[start]
    # The target (whether the trend was profitable enough).
    df['y'] = df['diff_percent'] > increase_thresh

    return df


def main():
    csv_filename = r"C:\Users\Paul\data\WIKI_PRICES\tickers\AAPL.csv"
    df = pd.read_csv(csv_filename)
    print(df.head())

    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]

    # Trade at the open on the first day.
    start = "open"
    # Trade at the close on the last day.
    end = "close"
    # Trade for this many days.
    n_days = 5
    # Only trade if the profit is greater than this percent.
    increase_thresh = 0.05

    df_trends = extract_trend_target(df, n_days, increase_thresh, start, end)
    # print(df_trends[[start, end, 'end_shift', 'diff', 'diff_percent', 'y']].head())

    # df_trends['diff_percent'].plot()
    # plt.show()

    # The features used to predict whether to trade.
    # The close values on the previous 5 days.
    features = []
    features.append(Feature("close_1", 1, "close"))
    features.append(Feature("close_2", 2, "close"))
    features.append(Feature("close_3", 3, "close"))

    df_features = extract_trend_features(df_trends, features)
    print(df_features.head(10))

if __name__ == '__main__':
    main()
