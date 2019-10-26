import pandas as pd
from pathlib import Path

def main():
    prices_filename = r'prices_df.csv'
    out_dir = Path('tickers')

    df = pd.read_csv(prices_filename)
    # Drop index column.
    df = df.drop(df.columns[[0]], axis=1)
    print(df.info())

    tickers = pd.unique(df['ticker'])
    print(tickers)

    # Save the tickers in separate csv files.
    for ticker in tickers:
        print(ticker)
        df_ticker = df.loc[df['ticker'] == ticker]
        df_ticker = df_ticker.drop('ticker', axis=1)
        df_ticker.to_csv(out_dir / f'{ticker}_prices.csv', index=False)


if __name__ == '__main__':
    main()
