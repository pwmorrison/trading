import numpy as np
import pandas as pd
import os

pd.options.mode.chained_assignment = None  # default='warn'

DATA_FOLDER = r"C:\Users\pwmor\Resilio Sync\FXBootcamp"


def get_daily_OHLC_ticker(ticker):
    df = pd.read_csv(os.path.join(DATA_FOLDER, "daily", "{}.csv".format(ticker)), header=None)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
    df["Ticker"] = ticker
    df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def get_daily_OHLC(tickers):
    dfs = []
    for ticker in tickers:
        df = get_daily_OHLC_ticker(ticker)
        # print(df.head())
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df


def convert_common_quote_currency(prices_df, quote_currency):
    tickers = pd.unique(prices_df['Ticker'])
    converted_dfs = []
    for ticker in tickers:
        df_ticker = prices_df[prices_df['Ticker'] == ticker]
        base = ticker[:3]
        quote = ticker[3:]

        if quote == quote_currency:
            # The ticker already has the desired quote currency.
            converted_dfs.append(df_ticker)
        elif base == quote_currency:
            # The base currency is the desired quote currency. Swap them over.
            df_ticker['Ticker'] = quote + base
            df_ticker['Open'] = 1 / df_ticker['Open']
            new_high = 1 / df_ticker['Low']
            df_ticker['Low'] = 1 / df_ticker['High']
            df_ticker['High'] = new_high
            df_ticker['Close'] = 1 / df_ticker['Close']
            converted_dfs.append(df_ticker)
        else:
            # Ignore the pairs that don't have the desired quote currency.
            continue

    df_common_quote = pd.concat(converted_dfs, axis=0)

    return df_common_quote


def get_policy_rates(currencies):
    currency_dfs = []
    for currency in currencies:
        df = pd.read_csv(os.path.join(DATA_FOLDER, "{}.csv".format(currency)), header=0)
        df.columns = ['Date', 'Rate']
        df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
        df["Currency"] = currency
        df = df[['Currency', 'Date', 'Rate']]
        df = df.dropna()
        df = df.sort_values(by=['Date'])
        currency_dfs.append(df)
    currency_df = pd.concat(currency_dfs, axis=0)
    return currency_df


def get_unique_currencies(df):
    tickers = pd.unique(df['Ticker'])
    currencies = []
    for ticker in tickers:
        base = ticker[:3]
        quote = ticker[3:]
        if base not in currencies:
            currencies.append(base)
        if quote not in currencies:
            currencies.append(quote)
    return currencies


def append_interest_rate_differential(prices_df, policy_rates_df):
    tickers = pd.unique(prices_df['Ticker'])
    extended_prices_dfs = []
    for ticker in tickers:
        prices_df_ticker = prices_df[prices_df['Ticker'] == ticker]
        base = ticker[:3]
        quote = ticker[3:]

        prices_df_ticker['Base'] = base
        prices_df_ticker['Quote'] = quote

        prices_df_ticker = pd.merge(prices_df_ticker, policy_rates_df, how='left', left_on=['Base', 'Date'],
                                    right_on=['Currency', 'Date'])
        prices_df_ticker = pd.merge(prices_df_ticker, policy_rates_df, how='left', left_on=['Quote', 'Date'],
                                    right_on=['Currency', 'Date'])

        prices_df_ticker = prices_df_ticker.sort_values(by=['Date'])

        prices_df_ticker['Base_Rate'] = prices_df_ticker['Rate_x'].fillna(method='ffill') * 0.01
        prices_df_ticker['Quote_Rate'] = prices_df_ticker['Rate_y'].fillna(method='ffill') * 0.01
        prices_df_ticker['Rate_Diff'] = prices_df_ticker['Base_Rate'] - prices_df_ticker['Quote_Rate']
        prices_df_ticker['Daycount_Fraction'] = (prices_df_ticker['Date'] - prices_df_ticker['Date'].shift(1)).dt.days / 365
        # 5. Calcualte interest returns, the daycount fraction * the rate differential
        prices_df_ticker['Interest_Returns'] = prices_df_ticker['Daycount_Fraction'] * prices_df_ticker['Rate_Diff']
        # 6. Calculate Interest_Accrual_on_Spot, the interest that accrues since the last observation on a single unit of currency given the last spot rate, expressed in the quote currency.
        # This is Daycount_Fraction * Rate_Diff * closing exchange rate
        prices_df_ticker['Interest_Accrual_on_Spot'] = \
            prices_df_ticker['Daycount_Fraction'] * prices_df_ticker['Rate_Diff'] * prices_df_ticker['Close']
        # 7. Calculate Spot returns from the closing prices
        prices_df_ticker['Spot_Returns'] = prices_df_ticker['Close'] / prices_df_ticker['Close'].shift(1) - 1

        # 8. Remove records for which we don't have interest rates (after carrying forward)
        # drop raw rates before omit as they might still contain NAs, hence leading to loss of rows that were filled by na.locf
        prices_df_ticker = prices_df_ticker.drop(['Currency_x', 'Currency_y', 'Rate_x', 'Rate_y'], axis=1)
        prices_df_ticker = prices_df_ticker.dropna()

        # prices_df_ticker.to_csv("prices_df_ticker_{}.csv".format(ticker))

        # 9. Calculate total return indexes which assumes periodic compounding of interest (on each price observation)
        prices_df_ticker['Spot_Return_Index'] = (prices_df_ticker['Spot_Returns'] + 1).cumprod()
        prices_df_ticker['Interest_Return_Index'] = (prices_df_ticker['Interest_Returns'] + 1).cumprod()
        prices_df_ticker['Total_Return_Index'] = (prices_df_ticker['Spot_Returns'] + prices_df_ticker['Interest_Returns'] + 1).cumprod()

        extended_prices_dfs.append(prices_df_ticker)

    extended_prices = pd.concat(extended_prices_dfs, axis=0)
    return extended_prices


if __name__ == '__main__':
    raw_prices_df = get_daily_OHLC(["AUDUSD", "USDCAD"])
    usd_prices_df = convert_common_quote_currency(raw_prices_df, 'USD')
    print(usd_prices_df.head())
    print(usd_prices_df.tail())
