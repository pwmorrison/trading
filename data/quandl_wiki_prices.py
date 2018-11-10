import pandas as pd
import os

def split_csv(filename, out_dir):
    """
    Splits the csv according to ticker names.
    """
    f = open(filename, 'r')
    for line in f:
        header = line
        break

    current_ticker = None
    out_file = None
    for line in f:
        print(line)
        ticker = line.split(',')[0]
        if current_ticker is None or ticker != current_ticker:
            # This is the first line of a new ticker.
            if out_file is not None:
                out_file.close()
                # return
            out_file = open(os.path.join(out_dir, "{}.csv".format(ticker)), 'w')
            out_file.write(header)

        out_file.write(line)
        current_ticker = ticker

        # return


if __name__ == '__main__':
    csv_filename = r"C:\Users\Paul\data\WIKI_PRICES\WIKI_PRICES.csv"
    out_dir = r"C:\Users\Paul\data\WIKI_PRICES\tickers"

    split_csv(csv_filename, out_dir)
