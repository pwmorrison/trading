
#define CurrentTrailingStop TradeVar[0]

function run()
{
	//set(PLOTNOW);
	StartDate = 20120101;
	EndDate = 20180201;
	BarPeriod = 1440;
	FrameOffset = 5; // Needed to align the frames with weeks.
	LookBack = 100; 
	
	TimeFrame = frameSync(7);
	bool is_wk_end = frame(FrameOffset);
	
	//
	// Stock filters.
	//
	asset("AAPL");
	TimeFrame = 1;
	vars stock_close = series(priceClose());
	TimeFrame = frameSync(7);
	vars stock_wkclose = series(priceClose());
	vars stock_wkhigh = series(priceHigh());
	vars stock_20wkhigh = series(MaxVal(stock_wkclose, 20));
	vars stock_20wkROC = series(ROC(stock_wkclose, 20));

	//
	// Index filter.
	//
	asset("SPX");
	vars index_wkclose = series(priceClose());
	vars index_10wkSMA = series(SMA(index_wkclose, 10));
	bool is_index_uptrending = index_wkclose[0] > index_10wkSMA[0];
	
	//
	// Exit trades.
	//
	// Do this before entering, so we have capital. 
	// We won't know the exact exit price (and therefore current account) though.
	// If we use TMFs, these should be executed prior to calling this.
	// PAUL: I'm putting this code here, rather than inside a TMF, so that it can use the current indicator series.
	// This loops through all pending and open trades.
	bool is_stock_exit = false;
	for(open_trades)
	{
		// Only exit at the week end.
		if (is_wk_end)
		{
			var trailing_percent;
			if (is_index_uptrending)
				trailing_percent = 0.4;
			else
				trailing_percent = 0.1;
			
			// Increase the trailing loss if it's more than the required % less than the highest price during the week.
			var trailing_thresh = stock_wkhigh[0] * (1. - trailing_percent);
			
			//CurrentTrailingStop = -1.3;
			
			// Adjust the trailing stop if required.
			if (ThisTrade->Skill[0] == -1)
			{
				// This is the end of the first week of the trade.
				ThisTrade->Skill[0] = trailing_thresh;
			}
			else if (trailing_thresh > ThisTrade->Skill[0])
			{
				// The trailing stop needs to be moved up.
				ThisTrade->Skill[0] = trailing_thresh;
			}
			
			// Exit the trade if required.
			if (stock_wkclose[0] < ThisTrade->Skill[0])
			{
				exitTrade(ThisTrade);
				is_stock_exit = true;
			}
			
			// ThisTrade - points to the current trade.
			// We have acccess to all trade variables for the current trade - https://www.zorro-trader.com/manual/en/trade.htm
			//printf("\n%s Pending/open trade of asset %s. Open price: %.2f. Current trailing stop: %.2f", 
			//	datetime(), Asset, TradePriceOpen, ThisTrade->Skill[0]);
			
			// Exit at market.
			//exitTrade(ThisTrade);
		}
	}
	
	//
	// Enter trades.
	//
	bool is_stock_entry = false;
	bool is_stock_at_high = stock_wkclose[0] > stock_20wkhigh[1];
	bool is_stock_high_roc = stock_20wkROC[0] > 30;
	if (is_wk_end && is_index_uptrending)
	{
		// Loop over stocks here.
		
		if (is_stock_at_high && is_stock_high_roc && NumLongTotal == 0)
		{
			// Enter the trade.
			asset("AAPL");
			Entry = 0;
			//TakeProfit = 20 * PIP;
			
			//enterLong(tmf);
			TRADE *current_trade = enterLong();
			
			// Initialise current trailing loss, to indicate we don't currently have one.
			current_trade->Skill[0] = -1;
			
			printf("\n%s Enter trade: %.2f", datetime(), stock_wkclose[0]);
			
			is_stock_entry = true;
		}
	}
		
	//if (NumLongTotal > 0)
	//	printf("\n%s run, %.2f", datetime(), stock_wkclose[0]);
	
	
	//printf("\n%s close: %.2f", datetime(), stock_wkclose[0]);
	
	
	//
	// Plotting.
	//
	if (is(FIRSTRUN))
	{
		// Plot without the if statements, to establish the correct order.
		//plot("Index Wk close", index_wkclose, MAIN|DOT, RED);
		plot("Index 10wkSMA", index_10wkSMA, LINE, BLUE);
		plotGraph("Index uptrending", 0, index_wkclose[0], CROSS, BLUE);
		
		plot("AAPL", stock_close, NEW|LINE, BLACK);
		plot("AAPL 20wkhigh", stock_20wkhigh, LINE, BLUE);
		plotGraph("AAPL 20wkhigh exceeded", 0, stock_close[0], CROSS, BLUE);
		plotGraph("AAPL entry", 0, stock_wkclose[0], TRIANGLE, GREEN);
		plotGraph("AAPL exit", 0, stock_wkclose[0], TRIANGLE, RED);
		
		plot("AAPL 20wkROC", stock_20wkROC, NEW|LINE, BLACK);
		plotGraph("AAPL 20wkROC 30", 0, stock_20wkROC[0], CROSS, BLUE);
	}
	else
	{
		PlotScale = 10;
		//plot("Index Wk close", index_wkclose, MAIN|DOT, RED);
		plot("Index 10wkSMA", index_10wkSMA, LINE, BLUE);
		if (is_wk_end && is_index_uptrending)
			plotGraph("Index uptrending", 0, index_wkclose[0], CROSS, BLUE);
		
		plot("AAPL", stock_close, NEW|LINE, BLACK);
		//plot("AAPL Wk close", stock_wkclose, DOT, RED);
		plot("AAPL 20wkhigh", stock_20wkhigh, LINE, BLUE);
		if (is_wk_end && is_stock_at_high)
			plotGraph("AAPL 20wkhigh exceeded", 0, stock_close[0], CROSS, BLUE);
		if (is_wk_end && is_stock_entry)
			plotGraph("AAPL entry", 0, stock_wkclose[0], TRIANGLE, GREEN);
		if (is_wk_end && is_stock_exit)
			plotGraph("AAPL exit", 0, stock_wkclose[0], TRIANGLE, RED);
		
		plot("AAPL 20wkROC", stock_20wkROC, NEW|LINE, BLACK);
		if (is_wk_end && is_stock_high_roc)
			plotGraph("AAPL 20wkROC 30", 0, stock_20wkROC[0], CROSS, BLUE);
	}
}