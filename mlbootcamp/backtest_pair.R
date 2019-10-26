args = commandArgs(trailingOnly=TRUE)
ticker_1 <- args[1]
ticker_2 <- args[2]
start_date <- args[3]
end_date <- args[4]
sd <- as.numeric(args[5])
out_returns <- args[6]
out_plot <- args[7]

# "C:\Program Files\R\R-3.6.1\bin\Rscript.exe" r_test.R CATY TRMK 2018-01-01 2018-04-01 2 backtest_return.csv rplot.jpg

DATA_FOLDER <- 'D:/ResilioSync/MLBootcamp'
source(paste0(DATA_FOLDER,'/Code/unsupervised-learning/simple-pair-backtest.R'))
if(!exists('prices_df')) load(paste0(DATA_FOLDER, '/Code/unsupervised-learning/raw-data.RData'))

annualised_returns <- auto_backtest(ticker_1, ticker_2, startDate = start_date, endDate = end_date, SD = sd)
print(annualised_returns)
write.csv(annualised_returns, out_returns)

jpeg(out_plot, width = 1000, height = 1000)
plot_spread_positions()
dev.off()



