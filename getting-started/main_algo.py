"""
This is an example algorithm that shows what is required for the judges to run your code

The program will take in the data (by passing in its path), and will output a numpy file

use this program like: `python3 main_algo.py -p train_data_50.csv`
"""
from eval_algo import eval_actions
import pandas as pd
from pathlib import Path
import argparse
import numpy as np


import talib as ta


# Read the data from the csv file
def readData(path):
    df = pd.read_csv(path)

    # change date into datetime objects
    df["Date"] = pd.to_datetime(df["Date"])

    # set indexes
    df.set_index(["Ticker", "Date"], inplace=True)

    return df


# Used to convert the dataframe into a numpy array
# col: the column to convert into a numpy array
#      in this case it can be "Open", "Close", "High", "Low", "Volume", etc
def pricesToNumpyArray(df, col="Open"):
    tickers = sorted(df.index.get_level_values("Ticker").unique())

    prices = []

    for ticker in tickers:
        stock_close_data = df.loc[ticker][col]
        prices.append(stock_close_data.values)

    prices = np.stack(prices)
    return prices

class Algo:
    def __init__(self, data_path, slowSMA=40, fastSMA=5):
        self.df = readData(data_path)
        self.open_prices = pricesToNumpyArray(self.df, col="Open")
        self.trades = np.zeros(self.open_prices.shape)
        
        self.slowSMA = slowSMA
        self.fastSMA = fastSMA

    def runSMA(self):
        # calculate trades based off of SMA momentum strategy

        for stock in range(len(self.open_prices)): 
            fast_sma = ta.SMA(self.open_prices[stock], timeperiod=self.fastSMA)
            slow_sma = ta.SMA(self.open_prices[stock], timeperiod=self.slowSMA)

            for day in range(1, len(self.open_prices[0])-1):
                
                # Buy: fast SMA crosses above slow SMA
                if fast_sma[day] > slow_sma[day] and fast_sma[day-1] <= slow_sma[day-1]:
                    # we are trading the next day's open price
                    self.trades[stock][day+1] = 1
                
                # Sell/short: fast SMA crosses below slow SMA
                elif fast_sma[day] < slow_sma[day] and fast_sma[day-1] >= slow_sma[day-1]:
                    # we are trading the next day's open price
                    self.trades[stock][day+1] = -1
                # else do nothing
                else:
                    self.trades[stock][day+1] = 0


    def saveTrades(self, path):
        # for convention please name the file "trades.npy"
        np.save(path, self.trades)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Demo Algorithm")

    # note that this could be the training dataset or the test dataset
    # so we can't hardcode things!
    parser.add_argument(
        "-p",
        "--prices",
        help="path to stock prices csv file",
    )

    prices_path = parser.parse_args().prices

    algo = Algo(prices_path)

    algo.runSMA()

    # we can run the evaluation for ourselves here to see how our trades did
    # you very likely will want to make your own system to keep track of your trades, cash, portfolio value etc, inside the 
    # runSMA function (or whatever equivalent function you have)
    print(eval_actions(algo.trades,algo.open_prices, cash=25000,verbose=True))

    algo.saveTrades("trades.npy")
