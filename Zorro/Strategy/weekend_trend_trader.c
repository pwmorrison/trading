function run()
{
	set(PLOTNOW);
	StartDate = 20170601;
	EndDate = 20180201;
	// Daily bars.
	BarPeriod = 1440;
	// Needed to align the frames with weeks.
	FrameOffset = 5;
	LookBack = 100;
	
	//
	// Index filter.
	//
	asset("SPX");
	// Determine the close price of the previous week.
	TimeFrame = frameSync(7);
	vars index_filter_weekclose = series(priceClose());
	bool is_week_end = frame(FrameOffset);
	// A 10 week moving average.
	vars index_filter_10weekSMA = series(SMA(index_filter_weekclose, 10));
	// Whether the close is above the 10 week SMA.
	bool index_filter_above = index_filter_weekclose[0] > index_filter_10weekSMA[0];
	
	//
	// Manage trades of each asset.
	//
	TimeFrame = 1;
	//while(asset(loop("AAPL", "MSFT")))
	asset("AAPL");
	vars stock_dailyclose = series(priceClose());
	
	TimeFrame = frameSync(7);
	vars stock_weekclose = series(priceClose());
	// The maximum close over the previous 20 weeks.
	vars stock_weekclose_20weekhigh = series(MaxVal(stock_weekclose, 20));
	// Whether we exceeded the previous 20 week high.
	bool is_high_exceeded = false;
	if (is_week_end)
		is_high_exceeded = stock_weekclose[0] > stock_weekclose_20weekhigh[1];
	// Rate of change. The percent increase over the past 20 weeks.
	vars stock_weekclose_20weekroc = series(ROC(stock_weekclose, 20));
	bool is_roc_exceeded = false;
	if (is_week_end)
		is_roc_exceeded = stock_weekclose_20weekroc[0] > 30;
	//float stock_roc = (stock_weekclose[0] - stock_weekclose[20]) / stock_weekclose[20];
	TimeFrame = 1;
	
	// Trailing stop at 40% below the entry price, or the highest weekly close.
	// When the index filter shows a downtrend (close below the SMA), adjust trailing stop
	// to 10% below the recent week's close. This gets put back to 40% when the index filter
	// goes back to an uptrend. We never move the stops down though.
	
	
	printf("\n%s close: %.2f", datetime(), index_filter_weekclose[0]);
	if (is_week_end)
	{
		//printf("\n%s.", datetime());
		
		// Index filter.
		if (index_filter_weekclose[0] > index_filter_10weekSMA[0])
		{
			// We've crossed the 10 week MA. We can now enter trades.
			
		}
		
		printf("\nEnd week");
	}
	
	if (is(FIRSTRUN))
	{
		// Plot without the if statements, to establish the correct order.
		
		plot("SPX wk close", index_filter_weekclose, MAIN|DOT, RED);
		plot("SPX 20wk SMA", index_filter_10weekSMA, MAIN|LINE, BLUE);
		plot("SPX above SMA", index_filter_weekclose, MAIN|DOT, BLUE);
		
		// Plot AAPL, with the 10 week high and whether it was exceeded.
		plot("AAPL day close", stock_dailyclose, NEW|LINE, BLACK);
		plot("Week close", stock_weekclose, DOT, RED);
		plot("20wk high", stock_weekclose_20weekhigh, LINE, GREEN);
		plotGraph("20wk high exceeded", 0, stock_weekclose[0], DOT, GREEN);

		// Plot stock rate of change.
		plot("AAPL ROC", stock_weekclose_20weekroc, NEW|LINE, RED);
		
		//return;
	}
	
	//printf("\n%s, M: %d, D: %d, H: %d, Close: %.4f, WeekClose: %.4f, Prev WeekClose: %.4f, %d", 
	//	Asset, month(), day(), hour(), Close[0], WeekClose[0], WeekClose[1], is_week_end);
	plot("SPX wk close", index_filter_weekclose, MAIN|DOT, RED);
	plot("SPX 20wk SMA", index_filter_10weekSMA, MAIN|LINE, BLUE);
	if (index_filter_above)
		//plotGraph("SPX above SMA", 0, index_filter_weekclose[0], MAIN|DOT, BLUE);
		plot("SPX above SMA", index_filter_weekclose, MAIN|DOT, BLUE);
	
	// Plot AAPL, with the 10 week high and whether it was exceeded.
	plot("AAPL day close", stock_dailyclose, NEW|LINE, BLACK);
	plot("Week close", stock_weekclose, DOT, RED);
	plot("20wk high", stock_weekclose_20weekhigh, LINE, GREEN);
	if (is_high_exceeded)
		plotGraph("20wk high exceeded", 0, stock_weekclose[0], DOT, GREEN);
		//plot("20wk high exceeded", stock_weekclose, DOT, GREEN);
	
	// Plot stock rate of change.
	plot("AAPL ROC", stock_weekclose_20weekroc, NEW|LINE, RED);
	if (is_roc_exceeded)
		plotGraph("20wk ROC exceeded", 0, stock_weekclose_20weekroc[0], DOT, RED);
}

