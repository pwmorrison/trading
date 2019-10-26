args = commandArgs(trailingOnly=TRUE)
pairs_file <- args[1]
start_date <- args[2]
end_date <- args[3]
sd <- as.numeric(args[4])
#out_dir <- args[5]
#out_plot <- args[7]

# "C:\Program Files\R\R-3.6.1\bin\Rscript.exe" r_test.R CATY TRMK 2018-01-01 2018-04-01 2 backtest_return.csv rplot.jpg

DATA_FOLDER <- 'D:/ResilioSync/MLBootcamp'
source(paste0(DATA_FOLDER,'/Code/unsupervised-learning/simple-pair-backtest.R'))
if(!exists('prices_df')) load(paste0(DATA_FOLDER, '/Code/unsupervised-learning/raw-data.RData'))

run_backtest <- function(pair) {
  print(pair)
  
  ticker_1 <- pair[1]
  ticker_2 <- pair[2]
  returns_filename <- pair[3]
  plot_filename <- pair[4]
  
  annualised_returns <- auto_backtest(ticker_1, ticker_2, startDate = start_date, endDate = end_date, SD = sd)
  print(annualised_returns)
  #write.csv(annualised_returns, paste0(out_dir, '/', 'backtest_returns_', ticker_1, '-', ticker_2, '.csv'))
  write.csv(annualised_returns, returns_filename)
  
  #jpeg(paste0(out_dir, '/', 'backtest_plot_', ticker_1, '-', ticker_2, '.jpeg'), width = 1000, height = 1000)
  jpeg(plot_filename)
  plot_spread_positions()
  dev.off()
}

pairs_df <- read.csv(pairs_file, header = FALSE)
apply(pairs_df, 1, run_backtest)


