# region imports
from AlgorithmImports import *
from collections import deque
import time
from sklearn.cluster import MeanShift, estimate_bandwidth
# endregion

class OilFuture(QCAlgorithm):
    def Initialize(self):
        # year = self.get_parameter('year',2007)
        self.SetStartDate(2020, 1, 4)  # Set Start Date
        self.SetEndDate(2024,12,27)
        self.SetCash(800000)  # Set Strategy Cash
        self.SetTimeZone(TimeZones.NEW_YORK)
        self.live_mode_sim = False
        self.lim_gap = 0.05

        self.timeframe = Resolution.MINUTE
        self.order_size = 5
        self.time_interval = 60 #set timeframe interval
        self.long_en = True
        self.short_en = True
        self.martingale_en = True

        self.continuous_contract = self.add_future(
            Futures.Energies.CrudeOilWTI,
            # Futures.Energies.MICRO_CRUDE_OIL_WTI,
            self.timeframe,
            data_normalization_mode= DataNormalizationMode.RAW,
            data_mapping_mode = DataMappingMode.LAST_TRADING_DAY,
            contract_depth_offset = 0,
            extended_market_hours = True,
            fill_forward = False,
            )
        # self.continuous_contract.SetFilter(0,90)
        self.continuous_contract.set_filter(lambda future_filter_universe: future_filter_universe.standards_only())
        self.symbol = self.continuous_contract.Symbol
        self.mapped_sym = None

        # self.consol_interval = self.get_parameter("timeframe", 60)
        self._consolidator = QuoteBarConsolidator(timedelta(minutes=self.time_interval))
        self._consolidator.DataConsolidated += self.consolidation_handler
        self.subscription_manager.add_consolidator(self.symbol, self._consolidator)

        # Indicators=========
        # long (10,100,1.8), (70,100,2)
        sma_long_param = self.get_parameter("sma_long_param", 10) #10
        std_long_param = self.get_parameter("std_long_param", 100) #50
        self.k_upper_long = self.get_parameter("k_upper_long", 1.8) #1.75, 0.75
        self.k_lower_long = self.k_upper_long*self.get_parameter("long_lower_mul", 0.9)

        # short (10,100,1.4)
        sma_short_param = self.get_parameter("sma_short_param", sma_long_param) #10
        std_short_param = self.get_parameter("std_short_param", std_long_param) #100
        self.k_lower_short = self.get_parameter("k_lower_short", 1.9) #1.9, 0.8
        # self.k_upper_short = 0.8
        self.k_upper_short = self.k_lower_short*self.get_parameter("short_lower_mul", 0.8)

        # macd_param = self.get_parameter("macd_param", 9)

        self._macd = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.Exponential)
        # macd_k = self.get_parameter("macd_k", 1.0)
        # fast, slow, sig = (int(sma_long_param*macd_k), int(sma_long_param*macd_k*2), int(sma_long_param*macd_k))
        # self._macd = MovingAverageConvergenceDivergence(fast, slow, sig, MovingAverageType.Exponential)
        self.register_indicator(self.symbol, self._macd, Resolution.HOUR)

        self._macd_long = MovingAverageConvergenceDivergence(120, 260, 90, MovingAverageType.Exponential)
        self.register_indicator(self.symbol, self._macd_long, Resolution.HOUR)

        self.sma_long= SimpleMovingAverage(sma_long_param) #create indicator
        self.register_indicator(self.symbol, self.sma_long, Resolution.HOUR) #register custom timeframe
        self.sma_long_window = RollingWindow[IndicatorDataPoint](5) #window to store historical data of the indicator
        self.sma_long.Updated += lambda sender,updated: self.sma_long_window.Add(updated)

        self.sma_short= SimpleMovingAverage(sma_short_param)
        self.register_indicator(self.symbol, self.sma_short, Resolution.HOUR)
        # self.sma_short.Updated += self.sma_shortUpdated
        self.sma_short_window = RollingWindow[IndicatorDataPoint](5)
        self.sma_short.Updated += lambda sender,updated: self.sma_short_window.Add(updated)

        self.std_long = StandardDeviation(std_long_param) #std for long
        self.register_indicator(self.symbol, self.std_long, Resolution.HOUR)
        self.std_long_window = RollingWindow[IndicatorDataPoint](2)
        self.std_long.Updated += lambda sender,updated: self.std_long_window.Add(updated)

        self.std_short = StandardDeviation(std_short_param) #std for short
        self.register_indicator(self.symbol, self.std_short, Resolution.HOUR)
        self.std_short_window = RollingWindow[IndicatorDataPoint](2)
        self.std_short.Updated += lambda sender,updated: self.std_short_window.Add(updated)
        
        # consolidator for price history window
        self.window = RollingWindow[QuoteBar](20)

        self.SetWarmup(5000)

        # state parameters (need object store)
        self.short_hold = 0
        self.long_hold = 0
        self.long_entryprice = None
        self.long_exitprice = None
        self.short_entryprice = None
        self.short_exitprice = None

        if self.live_mode:
            self.log("Init: Object store read")
            self.long_entryprice = float(self.object_store.read(f"{self.project_id}/long_entryprice"))
            self.long_exitprice = float(self.object_store.read(f"{self.project_id}/long_exitprice"))
            self.short_entryprice = float(self.object_store.read(f"{self.project_id}/short_entryprice"))
            self.short_exitprice = float(self.object_store.read(f"{self.project_id}/short_exitprice"))

        ## === Schedule for the custom rollover function
        self.Schedule.On(
            self.date_rules.every_day(self.symbol),
            self.time_rules.after_market_open(self.symbol, 5), #9.35 NY Time
            self.check_rollover)

        #Plot
        chart = Chart("Strategy")
        self.add_chart(chart)
        chart.add_series(Series("Price", SeriesType.LINE, "$", Color.Black))
        chart.add_series(Series("upperband_long", SeriesType.LINE, "$", Color.Blue))
        chart.add_series(Series("lowerband_long", SeriesType.LINE, "$", Color.Blue))
        chart.add_series(Series("Long", SeriesType.SCATTER, "$", Color.Green, ScatterMarkerSymbol.TRIANGLE))
        chart.add_series(Series("Short", SeriesType.SCATTER, "$", Color.Red, ScatterMarkerSymbol.TRIANGLE_DOWN))
       
        #plot2
        chart = Chart("Profit")
        self.add_chart(chart)
        chart.add_series(Series("Price", SeriesType.LINE, "$", Color.Black))
        chart.add_series(Series("Long", SeriesType.SCATTER, "$", Color.Green, ScatterMarkerSymbol.TRIANGLE))
        
        #plot2
        chart = Chart("long_strategy")
        self.add_chart(chart)
        chart.add_series(Series("Price", SeriesType.LINE, "$", Color.Black))
        chart.add_series(Series("long", SeriesType.SCATTER, "$", Color.Orange, ScatterMarkerSymbol.TRIANGLE))
        chart.add_series(Series("TP", SeriesType.SCATTER, "$", Color.Green, ScatterMarkerSymbol.TRIANGLE_DOWN))
        chart.add_series(Series("SL", SeriesType.SCATTER, "$", Color.Red, ScatterMarkerSymbol.TRIANGLE_DOWN))
        
        #plot3
        chart = Chart("short_strategy")
        self.add_chart(chart)
        chart.add_series(Series("Price", SeriesType.LINE, "$", Color.Black))
        chart.add_series(Series("short", SeriesType.SCATTER, "$", Color.Orange, ScatterMarkerSymbol.TRIANGLE))
        chart.add_series(Series("TP", SeriesType.SCATTER, "$", Color.Green, ScatterMarkerSymbol.TRIANGLE_DOWN))
        chart.add_series(Series("SL", SeriesType.SCATTER, "$", Color.Red, ScatterMarkerSymbol.TRIANGLE_DOWN))

    def is_active_time(self, log=False):
        is_time = (self.time.hour >= 4) and (self.time.hour <= 13)
        is_spread = abs(self.securities[self.mapped_sym].bid_price - self.securities[self.mapped_sym].ask_price) < 0.1
        # return True
        if is_time and is_spread:
            if log==True:
                self.log(f'Trade Enable at {self.time.hour}')
            return True
        else:
            return False

    def ceil_2digit(self, number):
        return math.ceil(number * 100) / 100

    # ==== Custom rollover logic (more customizable than the OnData logic) ====
    def check_rollover(self, day=1):
        if self.mapped_sym: #if mapped_sym exist and has more than 5 days expiry, do not roll
            if (self.mapped_sym.ID.Date.date() - self.time.date()).days >= day:
                return

        symbols = self.FutureChainProvider.GetFutureContractList(self.symbol, self.Time)
        if len(symbols) < 2:
                return #skip if symbols < 2
        
        # rollover until expiry more than 20 days
        old_sym = self.mapped_sym
        self.mapped_sym = sorted(symbols, key=lambda s: s.ID.Date)[0]
        if (self.mapped_sym.ID.Date.date() - self.time.date()).days <= 20:
            self.mapped_sym = sorted(symbols, key=lambda s: s.ID.Date)[1]
        if (self.mapped_sym.ID.Date.date() - self.time.date()).days <= 20:
            self.mapped_sym = sorted(symbols, key=lambda s: s.ID.Date)[2]

        if old_sym: #if old_sym exist, adjust gap and rollover position (if qty>0) 
            gap = (self.securities[self.mapped_sym].price - self.securities[old_sym].price)
            self.plot("symbol_gap", "gap", gap)

            # clear all open orders (if any doesn't filled)
            openOrders = self.Transactions.GetOpenOrders() 
            if len(openOrders)> 0: 
                for order in openOrders: 
                    self.Transactions.CancelOrder(order.Id)

            remaining_qty = self.portfolio[old_sym].quantity
            if remaining_qty != 0:
                if self.live_mode or (self.live_mode_sim==True):
                    ask_old = self.securities[old_sym].ask_price #for buy +
                    bid_old = self.securities[old_sym].bid_price #for sell -
                    ask_new = self.securities[self.mapped_sym].ask_price
                    bid_new = self.securities[self.mapped_sym].bid_price
                    if remaining_qty > 0: #roll_long
                        self.limit_order(old_sym, -remaining_qty, self.ceil_2digit(bid_old-self.lim_gap*2),tag=f'Rollover:NEW_SYM : Gap({gap})') #sell
                        self.limit_order(self.mapped_sym, remaining_qty, self.ceil_2digit(ask_new+self.lim_gap*2), tag=f'Rollover:NEW_SYM : Gap({gap})') #buy
                    else: #roll_short
                        self.limit_order(old_sym, -remaining_qty, self.ceil_2digit(ask_old+self.lim_gap*2),tag=f'Rollover:NEW_SYM : Gap({gap})') #buy
                        self.limit_order(self.mapped_sym, remaining_qty, self.ceil_2digit(bid_new-self.lim_gap*2), tag=f'Rollover:NEW_SYM : Gap({gap})') #sell
                else: #backtest
                    self.liquidate(old_sym, tag='Rollover:OLD_SYM')
                    self.market_order(self.mapped_sym, remaining_qty, tag=f'Rollover:NEW_SYM : Gap({gap})')

    # OnData function will feed in minutely data (due to Futures)
    def OnData(self, slice):
        if self.mapped_sym == None:
            symbols = self.FutureChainProvider.GetFutureContractList(self.symbol, self.Time)
            self.mapped_sym = sorted(symbols, key=lambda s: s.ID.Date)[0]
            if (self.mapped_sym.ID.Date.date() - self.time.date()).days <= 20:
                self.mapped_sym = sorted(symbols, key=lambda s: s.ID.Date)[1]
            if (self.mapped_sym.ID.Date.date() - self.time.date()).days <= 20:
                self.mapped_sym = sorted(symbols, key=lambda s: s.ID.Date)[2]
            return

    # Consolidated function that works like OnData for Cfd symbols (Put all the logic here)
    def consolidation_handler(self, sender: object, consolidated_bar: QuoteBar) -> None:
        self.window.add(consolidated_bar)

        # Accessing data        
        if not (self.mapped_sym and self.window.IsReady and self._macd.IsReady and self.sma_long.IsReady and self.std_long.IsReady) or self.IsWarmingUp:
            return

        # mapped_symbol = self.wti.Symbol
        sma_long_curr = self.sma_long[0].Value
        sma_short_curr = self.sma_short[0].Value

        macd_value = self._macd.Current.Value
        signal_value = self._macd.Signal.Current.Value

        histogram_long = self._macd_long.histogram[0].value

        curr = self.window[0].close
        prev = self.window[1].close

        curr_mapped = self.securities[self.mapped_sym].price

        # Calculate upper and lower bands for long strategy
        upperband_long = sma_long_curr + self.std_long[0].value * self.k_upper_long
        lowerband_long = sma_long_curr - self.std_long[0].value * self.k_lower_long

        # Calculate upper and lower bands for short strategy
        upperband_short = sma_short_curr + self.std_short[0].value * self.k_upper_short
        lowerband_short = sma_short_curr - self.std_short[0].value * self.k_lower_short

        # Short strategy =============
        short_qty = 0
        if self.short_en==True:
            if (curr < lowerband_short) and (macd_value < signal_value) and self.is_active_time(): #entry
                if self.short_hold < 0:
                    short_qty = 0
                elif self.short_hold == 0:
                    short_qty = -self.order_size
                    if self.martingale_en and ((self.long_hold + self.short_hold) > 0):
                        short_qty = -self.order_size*2 #double lot
                    self.Plot("short_strategy", "short", curr)
                    self.short_entryprice = curr
                    if self.live_mode:
                        self.object_store.save(f"{self.project_id}/short_entryprice", str(self.short_entryprice))
            elif (curr > upperband_short) and (macd_value > signal_value) and self.short_hold < 0: #exit
                short_qty = -self.short_hold
                self.short_exitprice = curr
                if self.live_mode:
                    self.object_store.save(f"{self.project_id}/short_exitprice", str(self.short_exitprice))
                if self.short_entryprice - self.short_exitprice > 0:
                    self.Plot("short_strategy", "TP", curr)
                else:
                    self.Plot("short_strategy", "SL", curr)

        # Long strategy ================
        long_qty = 0
        if self.long_en==True:
            if (curr > upperband_long) and (macd_value > signal_value) and self.is_active_time(): #entry
                if self.long_hold > 0:
                    long_qty = 0
                elif self.long_hold == 0:
                    long_qty = self.order_size
                    if self.martingale_en and ((self.short_hold + self.long_hold) < 0):
                        long_qty = self.order_size*2 #double lot
                    self.Plot("long_strategy", "long", curr)
                    self.long_entryprice = curr
                    if self.live_mode:
                        self.object_store.save(f"{self.project_id}/long_entryprice", str(self.long_entryprice))
            elif (curr < lowerband_long) and (macd_value < signal_value) and self.long_hold > 0: #exit
                long_qty = -self.long_hold
                self.long_exitprice = curr
                if self.live_mode:
                    self.object_store.save(f"{self.project_id}/long_exitprice", str(self.long_exitprice))
                if self.long_entryprice - self.long_exitprice >= 0:
                    self.Plot("long_strategy", "SL", curr)
                else:
                    self.Plot("long_strategy", "TP", curr)
      
        #convert the position
        order_qty = long_qty + short_qty
        self.short_hold = self.short_hold + short_qty
        self.long_hold = self.long_hold + long_qty
        if order_qty != 0:
            self.log(f'Order: QC_SENT ({order_qty})')
            if self.live_mode or (self.live_mode_sim==True):
                tag=f'hold[{self.long_hold + self.short_hold}] long[{long_qty}]: short[{short_qty}]'
                price = self.securities[self.mapped_sym].price
                ask = self.securities[self.mapped_sym].ask_price #buy +
                bid = self.securities[self.mapped_sym].bid_price #sell -
                if order_qty > 0: #buy
                    self.limit_order(self.mapped_sym, order_qty, self.ceil_2digit(ask+self.lim_gap), tag=tag+f' buy_lim@{self.ceil_2digit(curr_mapped+self.lim_gap)} p({price}):{bid}|{ask}') #positive adjust (buy)
                else: #sell
                    self.limit_order(self.mapped_sym, order_qty, self.ceil_2digit(bid-self.lim_gap), tag=tag+f' sell_lim@{self.ceil_2digit(curr_mapped-self.lim_gap)} p({price}):{bid}|{ask}') #negative adjust (sell)
                if self.live_mode:
                    self.object_store.save(f"{self.project_id}/short_hold", str(self.short_hold))
                    self.object_store.save(f"{self.project_id}/long_hold", str(self.long_hold))
            else:
                self.MarketOrder(self.mapped_sym, order_qty, tag=f'hold[{self.long_hold + self.short_hold}] long[{long_qty}]: short[{short_qty}]')
            
        #Plot=========================
        self.Plot("Strategy", "Price", curr)
        self.plot("MACD", "MACD Line", macd_value)
        self.Plot("MACD", "Signal Line", signal_value)
        if self.time.hour == 9:
            self.plot("long_strategy", "Price", curr)
            # self.Plot("long_strategy", "upperband_long", upperband_long)
            # self.Plot("long_strategy", "lowerband_long", lowerband_long)

            self.plot("short_strategy", "Price", curr)
            # self.Plot("short_strategy", "upperband_short", upperband_short)
            # self.Plot("short_strategy", "lowerband_short", lowerband_short)

            self.plot("Hold", "Total", self.short_hold + self.long_hold)
            # self.plot("Hold", "long", self.long_hold)
            # self.plot("Hold", "short", self.short_hold)

    # This function will be called for each Order events (e.g. Filled, Invalid, Canceled, etc.)
    def on_order_event(self, order_event: OrderEvent) -> None:
        # if INVALID order occurs (e.g. exceed margin limit), liquidate and stop algo
        if order_event.status == OrderStatus.INVALID:
            self.liquidate(tag="LIQ: INVALID order occurs")
            self.quit("Order: INVALID order occurs!!")
        elif order_event.status == OrderStatus.SUBMITTED:
            self.log(f'Order : SUBMITTED({order_event.fill_quantity})')
        elif order_event.status == OrderStatus.FILLED:
            self.log(f'Order : FILLED({order_event.fill_quantity})')
