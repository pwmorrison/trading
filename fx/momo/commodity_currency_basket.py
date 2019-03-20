import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../tools/')
from data_utils import get_daily_OHLC, convert_common_quote_currency, get_unique_currencies, get_policy_rates, append_interest_rate_differential


def main():
    raw_prices_df = get_daily_OHLC(['AUDUSD','NZDUSD','USDNOK','USDCAD'])
    usd_prices_df = convert_common_quote_currency(raw_prices_df, 'USD')
    print(usd_prices_df.head())
    print(usd_prices_df.tail())

    raw_prices_df.to_csv("raw_prices_df_pandas.csv")
    usd_prices_df.to_csv("usd_prices_df_pandas.csv")

    currencies = get_unique_currencies(usd_prices_df)
    print(currencies)

    policy_rates_df = get_policy_rates(currencies)
    print(policy_rates_df.head())
    print(policy_rates_df.tail())

    policy_rates_df.to_csv("policy_rates_df_pandas.csv")

    usd_extended_prices_df = append_interest_rate_differential(usd_prices_df, policy_rates_df)

    # Restrict data set to 2009 onwards (for now).
    prices_df = usd_extended_prices_df[usd_extended_prices_df['Date'].dt.year >= 2009]

    usd_extended_prices_df.to_csv("usd_extended_prices_df_pandas.csv")

    # Total return charts.
    fig, axes = plt.subplots(2, 2, squeeze=True)
    axes = axes.flatten()
    print(axes)
    tickers = pd.unique(usd_extended_prices_df['Ticker'])
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        df = usd_extended_prices_df[usd_extended_prices_df['Ticker'] == ticker]
        ax.plot(df['Date'], df['Interest_Return_Index'])
        ax.plot(df['Date'], df['Spot_Return_Index'])
        ax.plot(df['Date'], df['Total_Return_Index'])
        ax.grid()
        ax.set_title(ticker)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
