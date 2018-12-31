import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

"""
A simple backtester, used to validate strategies built in other frameworks.
"""

class Trade():
    def __init__(self, asset, open_price, open_date, close_price=None, close_date=None, trailing_stop=None):
        self.asset = asset
        self.open_price = open_price
        self.close_price = close_price
        self.open_date = open_date
        self.close_date = close_date
        self.trailing_stop = trailing_stop


class Strategy:

    def __init__(self, start_date, end_date, lookback):
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.cols = [6, 7, 8, 4]  # OHLC
        self.asset_dir = r"C:\Users\Paul\data\WIKI_PRICES\stocks_reformat_reverse_adjusted"
        self.index_asset = "SPX"
        #self.assets = ["A", "AA", "AAPL", "MSFT"]
        self.assets = ["AAPL"]
        self.asset_data = {}

        self.load_assets()

    def load_asset_data(self, filename):

        asset_df = pd.read_csv(filename, index_col=0)
        # Only keep our columns, and reorder them
        asset_df = asset_df.iloc[:, self.cols]
        asset_df.columns = ['open', 'high', 'low', 'close']
        # Reverse the rows, to make them from oldest to newest.
        asset_df = asset_df.iloc[::-1]
        # Convert index to datetime.
        asset_df.index = pd.to_datetime(asset_df.index)
        # Only keep the date range we're testing.
        asset_df = asset_df.loc[self.start_date - datetime.timedelta(weeks=self.lookback): self.end_date]

        # print(asset_df.head(10))
        # print(asset_df.dtypes)
        # print(asset_df.index.dtype)
        # print(asset_df.shape)

        return asset_df

    def resample_weekly(self, df):
        # Resample to weekly data.
        weekly_df = pd.DataFrame()
        weekly_df['open'] = df['open'].resample('W').first()
        weekly_df['high'] = df['high'].resample('W').max()
        weekly_df['low'] = df['low'].resample('W').min()
        weekly_df['close'] = df['close'].resample('W').last()

        # The resampling assigns an index of the last day of the week (Sunday).
        # Make it the Friday instead.
        weekly_df.index = weekly_df.index - pd.Timedelta('2 days')
        # print(weekly_df.head(10))

        return weekly_df

    def add_index_indicators(self, df):
        df['10_wk_sma'] = df['close'].rolling(10).mean()
        return df

    def add_asset_indicators(self, df):
        df['20_wk_high'] = df['high'].rolling(20).max()
        df['20_wk_roc'] = df['close'].pct_change(20)
        return df

    def load_assets(self):
        # Load the assets we're trading.
        for asset in self.assets:
            asset_filename = os.path.join(self.asset_dir, "{}.csv".format(asset))
            asset_df = self.load_asset_data(asset_filename)
            asset_df = self.resample_weekly(asset_df)
            asset_df = self.add_asset_indicators(asset_df)
            self.asset_data[asset] = asset_df
        # Load the index, used to determine market direction.
        index_filename = os.path.join(self.asset_dir, "{}.csv".format(self.index_asset))
        index_data = self.load_asset_data(index_filename)
        index_data = self.resample_weekly(index_data)
        index_data = self.add_index_indicators(index_data)
        self.index_data = index_data

    def run(self):
        # Calculate the first Friday.
        current_date = self.start_date
        current_date += datetime.timedelta(days=(4 - current_date.weekday() + 7) % 7)

        # Iterate over the Friday of every week in the timeframe.
        self.open_trades = []
        self.closed_trades = []
        asset_trade_open = []
        while current_date < self.end_date:
            print(current_date)
            print("Index data:", self.index_data.loc[current_date, :])
            print("Asset data:", self.asset_data['AAPL'].loc[current_date, :])

            is_index_uptrending = \
                self.index_data.loc[current_date, 'close'] > self.index_data.loc[current_date, '10_wk_sma']
            #print(is_index_uptrending, self.index_data.loc[current_date, 'close'], self.index_data.loc[current_date, '10_wk_sma'])

            # Close any trades.
            for trade in self.open_trades:
                asset_data = self.asset_data[trade.asset]
                close_price = asset_data.loc[current_date, 'close']
                asset_high = asset_data.loc[current_date, '20_wk_high']

                if is_index_uptrending:
                    trailing_percent = 0.4
                else:
                    trailing_percent = 0.1

                # Increase the trailing loss if it's more than the required % less than the highest price during the week.
                trailing_thresh = asset_high * (1 - trailing_percent)

                if trade.trailing_stop is None:
                    # This is the end of the first week of the trade.
                    trade.trailing_stop = trailing_thresh
                elif trailing_thresh > trade.trailing_stop:
                    # The trailing stop needs to be moved up.
                    trade.trailing_stop = trailing_thresh

                # Exit the trade if required.
                if close_price < trade.trailing_stop:
                    trade.close_price = close_price
                    trade.close_date = current_date
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                    asset_trade_open.remove(asset)
                    print("{} Exiting trade of asset {}. Open price: {:.3f}. Close price: {:.3f}"
                          .format(current_date, asset, trade.open_price, close_price))

            # Enter trades.
            if is_index_uptrending:
                for asset in self.asset_data.keys():
                    asset_data = self.asset_data[asset]

                    close_price = asset_data.loc[current_date, 'close']
                    prev_high = asset_data.loc[current_date - datetime.timedelta(weeks=1), '20_wk_high']
                    is_stock_at_high = close_price > prev_high
                    roc = asset_data.loc[current_date, '20_wk_roc']
                    is_stock_high_roc = roc > 0.3

                    if is_stock_at_high and is_stock_high_roc and asset not in asset_trade_open:
                        # Enter the trade.
                        trade = Trade(asset, close_price, current_date)
                        self.open_trades.append(trade)
                        asset_trade_open.append(asset)
                        print("{} Entering trade of asset {}. Most recent close price: {:.3f}."
                              .format(current_date, asset, close_price))

            current_date += datetime.timedelta(days=7)
            print()


    def plot_asset_data(self):
        # print("hi")
        # self.index_data['close'].plot()
        # self.index_data['10_wk_sma'].plot()


        fig, axes = plt.subplots(nrows=2, ncols=1)
        axes[0].plot(self.index_data.index, self.index_data['close'])
        axes[0].plot(self.index_data.index, self.index_data['10_wk_sma'])
        axes[0].grid()

        asset_axes = {}
        for i, asset in enumerate(self.asset_data.keys()):
            axis = i + 1
            asset_data = self.asset_data[asset]
            asset_axes[asset] = axis
            axes[axis].plot(asset_data.index, asset_data['close'])
            axes[axis].plot(asset_data.index, asset_data['20_wk_high'])

            axes[axis].grid()

        for trade in self.closed_trades:
            asset = trade.asset
            axis = asset_axes[asset]
            axes[axis].scatter(trade.open_date, trade.open_price, c='green')
            axes[axis].scatter(trade.close_date, trade.close_price, c='red')


        plt.show()



def main():
    start_date = datetime.datetime(2012, 1, 1)#'01-01-2012'
    end_date = datetime.datetime(2018, 1, 1)
    lookback = 20
    strategy = Strategy(start_date, end_date, lookback)
    strategy.run()
    strategy.plot_asset_data()


if __name__ == '__main__':
    main()
