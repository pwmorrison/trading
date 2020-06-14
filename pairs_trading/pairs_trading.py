import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
from typing import List
import csv
import pandas_market_calendars as mcal
from pathlib import Path

from IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
from ib.ext.ScannerSubscription import ScannerSubscription


"""
https://github.com/anthonyng2/ib/blob/master/IbPy%20Demo%20v2018-04-05.ipynb

TWS API guide: http://interactivebrokers.github.io/tws-api/
"""


"""
TODO:
* Preload data from files.
* Retrieve historical data from IB to make up for gaps in preloaded data.
* Update functions.
* Trade logic and IB interfacing for orders.
* Reporting.
* Avoid trading in holidays.
"""

ACCOUNT_NAME = "DU1272702"
HOST = ""
PORT = 4002
CLIENT_ID = 5563

tz = pytz.timezone('US/Eastern')
nyse = mcal.get_calendar('NYSE')


class IB:
    """
    Interface to Interactive Brokers.
    """

    # Definition of the tick data "field" value.
    tick_type = {
        0 : "BID_SIZE",
        1 : "BID_PRICE",
        2 : "ASK_PRICE",
        3 : "ASK_SIZE",
        4 : "LAST_PRICE",
        5 : "LAST_SIZE",
        6 : "HIGH",
        7 : "LOW",
        8 : "VOLUME",
        9 : "CLOSE_PRICE",
        10 : "BID_OPTION_COMPUTATION",
        11 : "ASK_OPTION_COMPUTATION",
        12 : "LAST_OPTION_COMPUTATION",
        13 : "MODEL_OPTION_COMPUTATION",
        14 : "OPEN_PRICE",
        15 : "LOW_13_WEEK",
        16 : "HIGH_13_WEEK",
        17 : "LOW_26_WEEK",
        18 : "HIGH_26_WEEK",
        19 : "LOW_52_WEEK",
        20 : "HIGH_52_WEEK",
        21 : "AVG_VOLUME",
        22 : "OPEN_INTEREST",
        23 : "OPTION_HISTORICAL_VOL",
        24 : "OPTION_IMPLIED_VOL",
        27 : "OPTION_CALL_OPEN_INTEREST",
        28 : "OPTION_PUT_OPEN_INTEREST",
        29 : "OPTION_CALL_VOLUME"}

    def __init__(self, account_name: str, host: str, port: int, client_id: int):
        self.account_name = account_name
        # Instantiate IBWrapper callback.
        self.callback = IBWrapper()
        # Instantiate EClientSocket and return data to callback
        self.tws = EClientSocket(self.callback)
        # Connect to TWS.
        self.tws.eConnect(host, port, client_id)

         # Instantiate contract class.
        self.create = contract()
        self.callback.initiate_variables()

    def request_account_data(self):
        """
        Requests the relevant account data.
        http://interactivebrokers.github.io/tws-api/account_updates.html
        http://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html#aea1b0d9b6b85a4e0b18caf13a51f837f
        """
        self.callback.update_AccountValue = []
        self.callback.update_Portfolio = []
        self.callback.update_AccountTime = None
        print(f'Requesting account data.')
        self.tws.reqAccountUpdates(subscribe=True, acctCode=self.account_name)
        # Retrieve the data from the callback when it arrives.
        while True:
            # Don't check for portfolio, as it may be empty to begin with.
            if all([len(self.callback.update_AccountValue) > 0,
                    self.callback.update_AccountTime
                    ]):
                account_data = pd.DataFrame(
                    self.callback.update_AccountValue,
                    columns=['key', 'value', 'currency', 'accountName'])
                portfolio = pd.DataFrame(
                    self.callback.update_Portfolio,
                    columns=['Contract ID','Currency',
                              'Expiry','Include Expired',
                              'Local Symbol','Multiplier',
                              'Primary Exchange','Right',
                              'Security Type','Strike',
                              'Symbol','Trading Class',
                              'Position','Market Price','Market Value',
                              'Average Cost', 'Unrealised PnL', 'Realised PnL',
                              'Account Name'])
                account_time = self.callback.update_AccountTime
                break
            print(f'Waiting for account data. ')
            time.sleep(2)
        # Unsubscribe.
        self.tws.reqAccountUpdates(subscribe=False, acctCode=self.account_name)
        self.callback.update_AccountValue = []
        self.callback.update_Portfolio = []
        self.callback.update_AccountTime = None
        return account_data, portfolio, account_time

    def request_portfolio(self):
        """
        Requests the portfolio.
        http://interactivebrokers.github.io/tws-api/account_updates.html
        http://interactivebrokers.github.io/tws-api/interfaceIBApi_1_1EWrapper.html#a790ccbe25033df73996f36a79ce2ce5a
        """
        self.callback.update_AccountValue = []
        print(f'Requesting portfolio data.')
        self.tws.reqAccountUpdates(subscribe=True, acctCode=self.account_name)
        # Retrieve the data from the callback when it arrives.
        while True:
            if len(self.callback.update_AccountValue) > 0:
                data = pd.DataFrame(
                    self.callback.update_AccountValue,
                    columns=['key', 'value', 'currency', 'accountName'])
                break
            print(f'Waiting for portfolio data.')
            time.sleep(2)
        # Unsubscribe.
        self.tws.reqAccountUpdates(subscribe=False, acctCode=self.account_name)
        self.callback.update_AccountValue = []
        return data


    def request_tick_data(self, ticker, max_num_tries=10):
        """
        Requests current tick data for the given ticker.
        """
        # Create the contract to request tick data for.
        contract_info = self.create.create_contract(ticker, 'STK', 'SMART', 'USD')
        tickerId = 1004
        self.callback.tick_Price = []

        # Request the data.
        # Take a snapshot, so we don't continue to get data.
        self.tws.reqMktData(tickerId=tickerId, contract=contract_info, genericTickList="", snapshot=True)

        # Retrieve the data from the callback when it arrives.
        received_data = False
        num_tries = 0
        while num_tries < max_num_tries:
            if len(self.callback.tick_Price) > 0:
                data = pd.DataFrame(
                    self.callback.tick_Price, 
                    columns=['tickerId', 'field', 'price', 'canAutoExecute'])
                data["Type"] = data["field"].map(IB.tick_type)
                print(data)
                if 'LAST_PRICE' in data['Type'].values:
                    # We have the information we need.
                    break
                if 'CLOSE_PRICE' in data['Type'].values:
                    if not received_data:
                        # We have the close price. Wait a little longer to see if we get the last price.
                        received_data = True
                        time.sleep(2)
                        continue
                    break
            print(f'Waiting for tick data for ticker {ticker}, with ID {tickerId}, after try {num_tries} of {max_num_tries}.')
            num_tries += 1
            time.sleep(2)

        self.tws.cancelMktData(tickerId)

        # Attach the type of each row, based on the field.
        # data["Type"] = data["field"].map(IB.tick_type)

        # Reset the tick_Price data.
        self.callback.tick_Price = []

        if not received_data:
            data = None
        
        return data

    def request_tick_data_repeat(self, ticker, max_num_tries):
        """
        Wrapper around request_tick_data, that makes a given number of attempts.
        On each attempt, a new connection is established.
        """
        num_tries = 0
        while num_tries < max_num_tries:
            data = self.request_tick_data(ticker)
            if data is not None:
                break
            num_tries += 1
        return data

    def request_historical_data(self, ticker: str, num_days: int, last_date: datetime = None):
        """
        https://interactivebrokers.github.io/tws-api/historical_bars.html
        http://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html#aad87a15294377608e59aec1d87420594
        """
        # Create the contract to request tick data for.
        contract_info = self.create.create_contract(ticker, 'STK', 'SMART', 'USD')
        tickerId = 1004
        self.callback.historical_Data = []

        # Example: To get 3 days ending on 1/7/2013, use the last second of 1/7/2013 as the endDateTime, 
        # and 3D as the durationStr.

        data_endtime = '' if last_date is None else last_date.strftime("%Y%m%d %H:%M:%S")

        # Request the data.
        # reqHistoricalData(self, tickerId, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate):
        self.tws.reqHistoricalData(
            tickerId=tickerId, 
            contract=contract_info, 
            endDateTime=data_endtime,  # The request's end date and time (the empty string indicates current present moment).
            durationStr=f"{num_days} D", 
            barSizeSetting="1 day",
            whatToShow="TRADES",  # Hopefully this will be a bar summarising the trades...
            useRTH=1,  # "Regular trading hours". For some reason, 1 chops of the most recent day.
            formatDate=1,  # dates applying to bars returned in the format: yyyymmdd{space}{space}hh:mm:dd
            #keepUpToDate=False
            )

        # Retrieve the data from the callback when it arrives.
        while True:
            # I think the last row has "finished" in the "date" column, to indicate all data is retrieved.
            if len(self.callback.historical_Data) > 0 and 'finished' in self.callback.historical_Data[-1][1]:
                print(self.callback.historical_Data)
                data = pd.DataFrame(
                    self.callback.historical_Data[:-1],  # Removed "finished" row. 
                    columns = ["reqId", "date", "open",
                              "high", "low", "close", 
                              "volume", "count", "WAP", 
                              "hasGaps"])
                break
            print(f'Waiting for historical data with ID {tickerId}.')
            time.sleep(2)

        # Reset the historical_Data data.
        self.callback.historical_Data = []

        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date')
        # print(data.to_string())

        return data[['date', 'open', 'high', 'low', 'close', 'volume', 'count']]

    def close(self):
        self.tws.eDisconnect()


class Stock:
    def __init__(self, ticker: str, lookback: int):
        self.ticker = ticker
        self.lookback = lookback
        # Store the bars in a data frame, with a row per day.
        self.data = self._init_bars_data()
        
    def _init_bars_data(self):
        column_names = ['date', 'open', 'high', 'low', 'close', 'is_closed']
        df = pd.DataFrame(columns=column_names)
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'is_closed': bool})
        pd.to_datetime(df['date'])
        return df

    def get_tick(self, ib: IB, random_tick_stddev=0):
        print(f'Getting tick data for stock {self.ticker}')
        tick_data = ib.request_tick_data_repeat(self.ticker, max_num_tries=5)
        # TODO: Handle when we can't get tick data.
        # print(tick_data.to_string())
        print(tick_data)

        # Use the last price if it exists, else use the close price.
        if 'LAST_PRICE' in tick_data['Type'].values:
            tick_row = tick_data[tick_data['Type'] == 'LAST_PRICE']
        elif 'CLOSE_PRICE' in tick_data['Type'].values:
            tick_row = tick_data[tick_data['Type'] == 'CLOSE_PRICE']
        print(tick_row)
        tick_price = tick_row['price'].values[0]

        if random_tick_stddev != 0:
            noise = np.random.normal(0, tick_price * random_tick_stddev)
            tick_price += noise

        return tick_price

    def update_data(self, current_date: datetime, ib: IB, random_data_stddev=0, preload_filename=None):
        """
        Update the stock's daily bar data.
        """
        # Get recent NYSE trading days.
        date_format = '%Y-%m-%d'
        nyse_sched = nyse.schedule(
            start_date=(current_date - timedelta(days=40)).strftime(date_format), 
            end_date=current_date.strftime(date_format))
        dates = mcal.date_range(nyse_sched, frequency='1D')

        # Get the dates we want to load.
        # Load an extra day, in case we're trading today and the extra day is needed for later calculations.
        dates = dates[-(self.lookback + 1):]
        # print(dates)

        # Load historical data from IB.
        # last_date = dates[-1]
        # print(f'Requesting historical data for stock {self.ticker} at date {last_date.strftime("%Y%m%d %H:%M:%S")}')
        print(f'Requesting historical data for stock {self.ticker} at most recent date.')
        ib_data = ib.request_historical_data(self.ticker, 21, last_date=None)
        # print(ib_data.to_string())

        if random_data_stddev != 0:
            noise = np.random.normal(0, ib_data['close'] * random_data_stddev)
            ib_data['close'] += noise

        self.data = ib_data


class Pair:
    def __init__(self, stock_1: Stock, stock_2: Stock, capital: float, lookback: int, n_std_dev: float, out_dir: Path):
        self.stock_1 = stock_1
        self.stock_2 = stock_2
        self.capital = capital
        self.lookback = lookback
        self.n_std_dev = n_std_dev
        self.multiplier = 1000
        # self.trades = []
        # self.current_trade = None

        # Data is a pandas dataframe, with columns: OHLC, MA, upper_band, lower_band.
        # self.pair_data = None

        self.out_dir = out_dir

    def update_data(self, date: datetime, ib: IB, random_data_stddev=0):
        """
        Updates the pair data at the given date.
        Data is currently re-retrieved and calculated, rather than updated.
        """
        print(f'Updating data for pair ({self.stock_1.ticker}, {self.stock_2.ticker}) to date P={date}.')
        # Update the bars for both stocks.
        self.stock_1.update_data(date, ib, random_data_stddev)
        self.stock_2.update_data(date, ib, random_data_stddev)
        
        # Compute the pair data.
        stock_1_data = self.stock_1.data[['date', 'close']].rename(columns={'close': 'close_1'})
        stock_2_data = self.stock_2.data[['date', 'close']].rename(columns={'close': 'close_2'})
        pair_data = stock_1_data.merge(stock_2_data)

        # Update moving average, standard deviation, and bollinger bands.
        pair_data['close'] = (pair_data['close_1'] / pair_data['close_2']) * self.multiplier
        pair_data['mean'] = pair_data['close'].rolling(window=self.lookback).mean()
        pair_data['std_dev'] = pair_data['close'].rolling(window=self.lookback).std()
        pair_data['bollinger_top'] = pair_data['mean'] + self.n_std_dev * pair_data['std_dev']
        pair_data['bollinger_bottom'] = pair_data['mean'] - self.n_std_dev * pair_data['std_dev']

        # print('Pair data:')
        # print(pair_data.to_string())
        if 0:
            # Output the pair data to a file.
            pair_data.to_csv(self.out_dir / f'{self}_{date.strftime("%Y-%m-%d")}.csv', index=False)

        return pair_data

    def generate_trades(self, data_filename, pair_position_df, date, ib, random_tick_stddev=0):
        """
        Analyses the current trade (if it exists), the MA, and bollinger bands, and triggers any trades.
        """
        pair_data = pd.read_csv(data_filename)

        # Get current price.
        stock_1_tick = self.stock_1.get_tick(ib, random_tick_stddev)
        stock_2_tick = self.stock_2.get_tick(ib, random_tick_stddev)
        current_price = self.multiplier * stock_1_tick / stock_2_tick
        print(f'Ticks: {stock_1_tick}, {stock_2_tick}, price {current_price}')

        # Extract data to determine trade orders.
        last_row = pair_data.iloc[-1]
        last_date = last_row['date']
        last_mean = last_row['mean']
        last_bollinger_top = last_row['bollinger_top']
        last_bollinger_bottom = last_row['bollinger_bottom']

        is_current_position = pair_position_df.shape[0]

        # Determine if the pair has a trade to exit.
        exit_trade = None
        if is_current_position:
            is_long = pair_position_df['is_long'].item()
            if self.is_exit(current_price, last_mean, is_long):
                size_1 = pair_position_df['size_1'].item()
                size_2 = pair_position_df['size_2'].item()
                exit_trade = Trade(self, date, not is_long, current_price,
                                   last_mean, last_bollinger_top, last_bollinger_bottom,
                                   size_1, size_2)

        # Determine if the pair has a trade to enter.
        # Only enter if there is no current position or we're exiting the current position.
        enter_trade = None
        if not is_current_position or exit_trade is not None:
            enter, is_long = self.is_enter(current_price, last_bollinger_top, last_bollinger_bottom)
            if enter:
                # Create a new trade.
                size_1 = round(self.capital / 2 / stock_1_tick)
                size_2 = round(self.capital / 2 / stock_2_tick)
                enter_trade = Trade(self, date, is_long, current_price,
                                    last_mean, last_bollinger_top, last_bollinger_bottom,
                                    size_1, size_2)

        return exit_trade, enter_trade

    def is_exit(self, price, mean, is_long):
        """
        Determines whether the trade should be exited at the given price and mean.
        """
        if is_long and price > mean:
            return True
        if not is_long and price < mean:
            return True
        return False

    def is_enter(self, price, bollinger_top, bollinger_bottom):
        """
        Determines whether an entry should be made, and its direction.
        """
        if price > bollinger_top:
            # Short trade.
            return True, False
        if price < bollinger_bottom:
            # Long trade.
            return True, True
        return False, None

    def __repr__(self):
        return f'{self.stock_1.ticker}_{self.stock_2.ticker}'


class Trade:
    def __init__(self, pair, date, is_buy, trigger_price, last_mean, last_bollinger_top, last_bollinger_bottom, size_1, size_2):
        self.pair = pair
        self.date = date
        self.is_buy = is_buy
        self.trigger_price = trigger_price
        self.last_mean = last_mean
        self.last_bollinger_top = last_bollinger_top
        self.last_bollinger_bottom = last_bollinger_bottom
        self.size_1 = size_1
        self.size_2 = size_2


class PairsTradingStrategy:
    """
    Main trading strategy logic.
    """

    def __init__(self, pairs: List[Pair], out_dir, out_filename, positions_filename, ib, ignore_dt=False, random_data_stddev=0, random_tick_stddev=0):
        self.pairs = pairs
        self.out_dir = out_dir
        self.out_filename = out_filename
        self.positions_filename = positions_filename
        self.ib = ib
        self.ignore_dt = ignore_dt
        self.random_data_stddev = random_data_stddev
        self.random_tick_stddev = random_tick_stddev

        self.current_trades = []

    def run(self):
        # # Initial data update for all pairs
        # dt = datetime.now(tz=tz)
        # self.update_data(dt, self.ib)

        sleep_time_short = 30  # seconds
        sleep_time_long = 10 * 60
        daily_reset_done = False
        daily_data_updates_done = False
        daily_trades_done = False
        daily_position_updates_done = False

        if self.ignore_dt:
            # The number of the day into the simulation, when testing.
            dummy_day_num = 0

        while True:
            dt = datetime.now(tz=tz)
            print(dt)

            # Sleep through non-trade days.
            if not self.ignore_dt:
                date_format = '%Y-%m-%d'
                current_date_str = dt.strftime(date_format)
                nyse_sched = nyse.schedule(
                    start_date=(dt - timedelta(days=1)).strftime(date_format),
                    end_date=current_date_str)
                latest_trade_date = mcal.date_range(nyse_sched, frequency='1D')[-1]
                latest_trade_date_str = latest_trade_date.strftime(date_format)
                if current_date_str != latest_trade_date_str:
                    # This isn't a trade day. Sleep a while.
                    print(f'Not a trade day. Sleeping {sleep_time_long} seconds.')
                    time.sleep(sleep_time_long)
                    continue

            if self.ignore_dt:
                # Go to the next day, so we can simulate the progression of days.
                dt = dt + timedelta(days=dummy_day_num)

            # At 12:00AM reset the update flags.
            if (not daily_reset_done and dt.hour == 0 and dt.minute >= 0) or self.ignore_dt:
                print(f'Resetting update flags at time {dt}.')
                daily_reset_done = True
                daily_data_updates_done = False
                daily_orders_done = False
                daily_trade_updates_done = False

            # At 3:30PM update time series.
            if not (daily_data_updates_done and dt.hour == 15 and dt.minute >= 30) or self.ignore_dt:  # 15:30
                print(f'Updating data at time {dt}.')
                # When testing, Use the current dt to get historical data.
                update_dt = dt if not self.ignore_dt else dt - timedelta(days=dummy_day_num)
                self.update_data(update_dt, dt, self.ib)
                print(f'Finished updating data at time {dt}.')
                daily_data_updates_done = True

            # At 3:30PM place enter/exit orders.
            if (not daily_trades_done and dt.hour == 15 and dt.minute >= 30) or self.ignore_dt:  # 15:30
                print(f'Generating trades at time {dt}.')
                exit_trades, enter_trades = self.generate_trades(dt, self.ib)
                print(f'Finished generating trades at time {dt}.')
                daily_trades_done = True

            # At 4:30PM update trades.
            if (not daily_position_updates_done and dt.hour == 16 and dt.minute >= 30) or self.ignore_dt:
                print(f'Updating positions at time {dt}.')
                self.update_positions(dt, self.ib)
                print(f'Finished positions at time {dt}.')
                daily_position_updates_done = True
                daily_reset_done = False

            if self.ignore_dt:
                dummy_day_num += 1

            if not self.ignore_dt:
                time.sleep(sleep_time_short)
            # return

    def form_pair_data_filename(self, dt, pair):
        return str(self.out_dir / f'{dt.strftime("%Y-%m-%d")}_data_{pair.stock_1.ticker}-{pair.stock_2.ticker}.csv')

    def form_trades_filename(self, dt):
        return str(self.out_dir / f'{dt.strftime("%Y-%m-%d")}_trades.csv')

    def update_data(self, date: datetime, filename_date: datetime, ib: IB):
        """
        Updates all data for all pairs.
        """
        close_ib_connection = False
        if ib is None:
            # We need to open a new connection.
            ib = IB(ACCOUNT_NAME, HOST, PORT, CLIENT_ID)
            close_ib_connection = True

        for pair in self.pairs:
            pair_data = pair.update_data(date, ib, self.random_data_stddev)
            pair_data.to_csv(self.form_pair_data_filename(filename_date, pair), index=False)

        if close_ib_connection:
            ib.close()

    def generate_trades(self, date: datetime, ib: IB):
        """
        Generates orders for all pairs, using the pairs' current data.
        """
        close_ib_connection = False
        if ib is None:
            # We need to open a new connection.
            ib = IB(ACCOUNT_NAME, HOST, PORT, CLIENT_ID)
            close_ib_connection = True

        # Read current positions.
        positions_df = pd.read_csv(self.positions_filename)

        exit_trades = []
        enter_trades = []
        for pair in self.pairs:
            # The current position for this pair.
            pair_position_df = positions_df[
                (positions_df['stock_1'] == pair.stock_1.ticker) &
                (positions_df['stock_2'] == pair.stock_2.ticker) &
                (positions_df['is_open'] == 1)]
            pair_data_filename = self.form_pair_data_filename(date, pair)
            exit_trade, enter_trade = pair.generate_trades(pair_data_filename, pair_position_df, date, ib, self.random_tick_stddev)
            if exit_trade is not None:
                exit_trades.append(exit_trade)
            if enter_trade is not None:
                enter_trades.append(enter_trade)

        print(f'Exit trades: {exit_trades}')
        print(f'Enter trades: {enter_trades}')

        # TODO: Generate orders and send the orders to the IB interface.

        # Output trades.
        # TODO: Probably also record some kind of order ID returned by IB.
        with open(self.form_trades_filename(date), 'w') as f:
            f.write(f'stock_1,stock_2,is_exit,is_buy,trigger_price,last_mean,last_bollinger_top,last_bollinger_bottom,size_1,size_2\n')
            for trade in exit_trades:
                f.write(f'{trade.pair.stock_1.ticker},{trade.pair.stock_2.ticker},1,{1 if trade.is_buy else 0},'
                        f'{trade.trigger_price},{trade.last_mean},'
                        f'{trade.last_bollinger_top},{trade.last_bollinger_bottom},'
                        f'{trade.size_1},{trade.size_2}\n')
            for trade in enter_trades:
                f.write(f'{trade.pair.stock_1.ticker},{trade.pair.stock_2.ticker},0,{1 if trade.is_buy else 0},'
                        f'{trade.trigger_price},{trade.last_mean},'
                        f'{trade.last_bollinger_top},{trade.last_bollinger_bottom},'
                        f'{trade.size_1},{trade.size_2}\n')

        if close_ib_connection:
            ib.close()

        return exit_trades, enter_trades

    def update_positions(self, dt, ib):
        """
        Updates the trade data after the order should have been filled.
        """
        positions_df = pd.read_csv(self.positions_filename)
        trades_df = pd.read_csv(self.form_trades_filename(dt))

        # TODO: Check what price the MOC order got filled at.
        # TODO: For now, assume we fill the orders at the trigger price.
        trades_df['fill_price'] = trades_df['trigger_price']

        exit_trades_df = trades_df[trades_df['is_exit'] == 1]
        enter_trades_df = trades_df[trades_df['is_exit'] == 0]

        # Update the exit columns for the exit trades.
        for index, row in exit_trades_df.iterrows():
            position_indicator = (positions_df['stock_1'] == row.stock_1) & (positions_df['stock_2'] == row.stock_2) & (positions_df['is_open'] == 1)
            positions_df.loc[position_indicator, 'exit_date'] = dt.strftime("%Y-%m-%d")
            positions_df.loc[position_indicator, 'is_open'] = 0
            positions_df.loc[position_indicator, 'exit_price'] = row['fill_price']
            positions_df.loc[position_indicator, 'exit_trigger_price'] = row['trigger_price']

        # Add rows for the trades we're entering.
        for index, row in enter_trades_df.iterrows():
            positions_df = positions_df.append({
                'stock_1': row['stock_1'],
                'stock_2': row['stock_2'],
                'entry_date': dt.strftime("%Y-%m-%d"),
                'exit_date': '',
                'is_open': 1,
                'is_long': row['is_buy'],
                'entry_price': row['fill_price'],
                'exit_price': '',
                'entry_trigger_price': row['trigger_price'],
                'exit_trigger_price': '',
                'size_1': row['size_1'],
                'size_2': row['size_2'],
            }, ignore_index=True)

        # Record the positions.
        positions_df.to_csv(self.out_dir / f'{dt.strftime("%Y-%m-%d")}_positions.csv', index=False)
        positions_df.to_csv(self.positions_filename, index=False)

    def output_current_trades(self, dt):
        with open(self.out_filename, 'w') as f:
            for t in self.current_trades:
                f.write(f'{dt.strftime("%Y-%m-%d_%H-%M-%S")}, {t.pair.stock_1.ticker}, {t.pair.stock_2.ticker}, '
                        f'{t.enter_date.strftime("%Y-%m-%d_%H-%M-%S")}, {t.enter_price}, '
                        f'{t.exit_price}, {t.enter_date.strftime("%Y-%m-%d_%H-%M-%S") if t.exit_date else None}, '
                        f'{t.current_mean}, {t.current_bol_top}, {t.current_bol_bottom}\n')

    def output_trade(self, dt, trade):
        with open(self.out_filename, 'w') as f:
            f.write(f'{dt.strftime("%Y-%m-%d_%H-%M-%S")}, {trade.pair.stock_1.ticker}, {trade.pair.stock_2.ticker}, '
                    f'{trade.date.strftime("%Y-%m-%d_%H-%M-%S")}, {trade.price}, {trade.is_enter}, '
                    f'{trade.current_mean}, {trade.current_bol_top}, {trade.current_bol_bottom}\n')

    def daily_trade_updates_done(self):
        """
        After the market closes, check to see what trades we got on, and update data.
        """
        pass


def create_test_pairs():
    pairs = [
        Pair(Stock('AEE'), Stock('AEP')),
        Pair(Stock('EXR'), Stock('CUBE')),
        Pair(Stock('PH'), Stock('ROK')),
    ]
    return pairs


def create_pairs(filename, total_capital, lookback, n_std_dev, out_dir):
    pairs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            tickers, capital_portion = row
            ticker_1, ticker_2 = tickers.split('/')
            capital_portion = float(capital_portion)
            pair_capital = total_capital * capital_portion

            print(f'Creating pair ({ticker_1}, {ticker_2}), with capital portion {capital_portion}.')

            stock_1 = Stock(ticker_1, lookback)
            stock_2 = Stock(ticker_2, lookback)
            pair = Pair(stock_1, stock_2, pair_capital, lookback, n_std_dev, out_dir)

            pairs.append(pair)

    return pairs


def initialise_positions_file(positions_filename):
    positions_df = pd.DataFrame(columns=['stock_1', 'stock_2', 'entry_date', 'exit_date', 'is_open', 'is_long',
                                         'entry_price', 'exit_price',
                                         'entry_trigger_price', 'exit_trigger_price',
                                         'size_1', 'size_2'])
    positions_df.to_csv(positions_filename, index=False)


def main():
    total_capital = 100000
    lookback = 20
    n_std_dev = 1
    pairs_filename = r'D:/data/pairs_trading/pairs2020.csv'
    out_dir = Path(r'D:/data/pairs_trading/out')
    positions_filename = out_dir / 'positions.csv'

    ignore_dt = True

    # For testing, a use this std. dev. to apply random noise to the returned prices.
    # Portion of the stock price.
    random_data_stddev = 0.1
    random_tick_stddev = 0.1

    if not positions_filename.exists():
        initialise_positions_file(positions_filename)

    if 0:
        ib = IB(ACCOUNT_NAME, HOST, PORT, CLIENT_ID)
        time.sleep(2)
        account_data, portfolio, account_time = ib.request_account_data()
        print('Account data:')
        print(account_data.to_string())
        print('Portfolio:')
        print(portfolio.to_string())
        print(f'Account time: {account_time}')
    else:
        ib = None


    pairs = create_pairs(pairs_filename, total_capital, lookback, n_std_dev, out_dir)

    pairs = pairs[:1]

    out_filename = out_dir / 'trades.csv'

    strategy = PairsTradingStrategy(pairs, out_dir, out_filename, positions_filename, ib, ignore_dt=ignore_dt,
                                    random_data_stddev=random_data_stddev, random_tick_stddev=random_tick_stddev)
    strategy.run()
    ib.close()


if __name__ == '__main__':
    main()
