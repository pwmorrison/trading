# Strategies
## Mean reversion
Identifies or computes a time series that is stationary, or from which we can easily calculate the mean, and assumes that any deviation from the mean is temporary, and will eventually return to the mean.
### Pair trading
A type of mean reversion strategy, where two cointegrated stocks are used to form the time series. For example, we can subtract a gold index from the gold price. The resulting series may be stationary. We can then long or short the two stocks when they deviate, with the assumption that they will eventually revert to the mean. 


## Trend following


## Opening gaps
See Scott Andrews' talk on "Better System Trader" podcast. The basic idea is that a stock that gaps up or down at the start of day, vs the end of the previous day, it will then revert back to the price at the end of the previous day. There seem to be conditions etc. on when this behaviour becomes tradeable. Apparently he has a book.


## Machine learning
These "data mining" techniques are generally frowned upon, because they tend to overfit the training set, and consequently don't give good returns on the validation set or in trading. The problem is that the sample size is small (particularly when fitting to a single stock's data), and also highly correlated. So, most people use simple hand-crafted strategies (see above), where they might do some limited hand-optimisation of the parameters on the training set. They say less than 5 parameters is best.

I wonder if the following will work:
* Use data from all stocks in a universe (e.g. S&P500), and try to find general patterns.
* Have a well-defined goal, such as "buy at the start of the day, and sell at the end of the day, or at a trailing stop". Then, I can run over the training set and use this rule to specify if that trade would be profitable on that day, for that stock. This becomes the (binary) target of the ML.
    * I'd need intra-day data to work out the trailing stops.
* Use a number of inputs that make sense, such as the past days' OHLCV, sentiment, bollinger bands, moving average values, maybe other indicators such as bull/bear market, sector, etc. 