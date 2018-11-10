import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

    # Drop the last n_days rows, since we don't have the future data for these.
    df_trends = df_trends.dropna()
    # df_trends = df_trends.drop(df_trends.index[-n_days:])

    # df_trends['diff_percent'].plot()
    # plt.show()

    # The features used to predict whether to trade.
    # The close values on the previous 5 days.
    features = []
    features.append(Feature("close_1", 1, "close"))
    features.append(Feature("close_2", 2, "close"))
    features.append(Feature("close_3", 3, "close"))

    df_features = extract_trend_features(df_trends, features)

    df_features = df_features.dropna()

    # Convert the boolean to 0/1.
    df_features['y'] = df_features['y'].astype(int)

    print(df_features.head(10))
    print(df_features.tail(10))

    # Split into train and test DataFrames.
    n_train_rows = int(df_features.shape[0] * 0.8)
    df_train = df_features.iloc[:n_train_rows]
    df_test = df_features.iloc[n_train_rows:]
    print(df_features.shape, df_train.shape, df_test.shape)

    class_counts = df_test['y'].value_counts()
    print(class_counts)

    # Weights for the samples.
    # negative_weight = 1 - float(class_counts[0]) / df_test.shape[0]
    # positive_weight = 1 - float(class_counts[1]) / df_test.shape[0]
    # df_train['weights'] = df_train['y'].apply(lambda x: negative_weight if x == 0 else positive_weight)

    prediction_features = ['open', 'close_1', 'close_2', 'close_3']
    X =df_train[prediction_features]
    y = df_train['y']

    # Standarize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced').fit(X_std, y)

    X_test = df_test[prediction_features]
    y_test = df_test['y']
    X_test_std = scaler.transform(X_test)

    y_pred = clf.predict(X_test_std)
    score = clf.score(X_test_std, y_test)
    print(score)

    # A confusion matrix shows that it's always predicting '0'.
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(conf_matrix)
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))

    print(clf.get_params())

    if 0:
        # A simple linear regression, just to see if it's working.
        # Should give ~1 for the 'close_3' coefficient, since it would be pretty similar to the close.
        # Should give a 0 coefficient to 'volume'.
        reg = LinearRegression().fit(df_train[['close_3', 'volume']], df_train['close'])
        r_2 = reg.score(df_test[['close_3', 'volume']], df_test['close'])
        print('Coefficients: \n', reg.coef_)
        print(r_2)
        y_pred = reg.predict(df_test[['close_3', 'volume']])
        plt.scatter(y_pred, df_test['close'])
        plt.show()


if __name__ == '__main__':
    main()
