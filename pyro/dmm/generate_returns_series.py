import pandas as pd
import os

"""
Takes price series data, and generates returns data useful for machine learning input.
"""


def main():
    # input_filename = r"C:\Users\pwmor\Zorro\History\EURUSD_hourly_2006to2017.csv"
    input_filename = r"C:\Users\pwmor\Zorro\History\EURUSD_daily_2006to2017.csv"
    output_filename = r"EURUSD_daily_2006to2017_returns.csv"

    price_df = pd.read_csv(input_filename, index_col=0)

    # The hourly returns, based on close.
    price_df['Returns'] = price_df['Close'].pct_change(1)
    price_df = price_df.drop(price_df.index[0])

    print(price_df.head(10))

    price_df.to_csv(output_filename)

if __name__ == '__main__':
    main()
