import os
# import pandas_ta as ta # dependency removed as unused in main pipeline
# Math
import numpy as np

# DataFrames
import pandas as pd

# Time
import datetime as dt

# Utility
from collections import deque
import warnings

# Visualization
import matplotlib.pyplot as plt
import scipy as sp
try:
    import tpqoa
except ImportError:
    # Mock tpqoa if not available (for local data mode)
    class tpqoa:
        class tpqoa:
            def __init__(self, conf_file):
                pass
            def get_history(self, *args, **kwargs):
                pass
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="If the \"CreateHorizontalLine\" wants to be used")


def ProcessData(df, symbol, columns):
    raw = df.rename(columns=columns)
    raw['Time'] = pd.to_datetime(raw['Time'])
    raw['EndTime'] = raw['Time']
    raw = raw.set_index('Time',drop=False)
    raw = raw.sort_index()
    raw['instrument'] = instrument
    return raw


class PatternIndicator():
    def __init__(self,name,instrument,resolution = None,research=True):
        """ CustomHeikinAshi indicator
        This indicator creates a TradeBar custom object, that is to say,
        It has the four characteristics prices of a Candlestick computed with the
        rules of the Heikin-Ashi formula: https://www.investopedia.com/trading/heikin-ashi-better-candlestick/.

        Inputs:
        -- Name (String): A characteristic name for the definition of the indicator.

        -- resolution (int): Set the space between points for the "CreateHorizontalLine" method.
                             Only 4 preselected options available. 2 for minute, 3 for hour and 4 for day (same as the object Resolution of QC).
                             The last option is pass a timedelta object.  

        Initialization Parameters:
        -- Time (datetime.datetime object): Inherits the time of the creation and,
                                             on the update method, the moment of update.
        -- Value (float non-NonTypeObject): Actual value of the indicator. Where the arrows will be created.

        -- Date (datetime.datetime object): It records the last time the the indicator was active, i.e. a pattern was detected.

        -- Flag (Int non-NonTypeObject): cero represents no change, one bearish fractal, and two bullish fractals.
        """
        if resolution is None:
            warnings.warn('If the "CreateHorizontalLine" wants to be used, a resolution should be passed.\
                     Either a timedelta object or an "int" between 2-4')
        self.Name = name
        self.instrument = instrument
        self.Time = dt.datetime.min
        self.Flag = 0
        
        # The following parametters are necessary for the creation of the figures
        self.Value = np.nan
        self.Date = dt.datetime.min
        self.OnResearch = research
        if research:
            self.research = {}

    def InitResearchKey(self, key, datapoint):
        self.research[key] = [datapoint]

    def AddDataResearch(self,**kwargs):
        for k in kwargs.keys():
            try:
                self.research[k].append(kwargs[k])
            except KeyError:
                self.InitResearchKey(k, kwargs[k])

    def set_resolution(self,resolution):
        self.Delta = {'days':0,'hours':0,'minutes':0}
        if resolution == 2: # Resolution.Minute:
            self.Delta = dt.timedelta(minutes=1)
        elif resolution == 3: # Resolution.Hour:
            self.Delta = dt.timedelta(hours=1)
        elif resolution == 4: # Resolution.Daily:
            self.Delta = dt.timedelta(days=1)
        else:
            assert isinstance(resolution, dt.timedelta),f'The passed is {type(resolution)}, only int or timedelta is acepted.'
            self.Delta = resolution

#     def ArrowCreation(self,percentage=0.1,x_tail=5):
#         """
#         Function to create the positions of the X and Y axis for the arrows.
        
#         Input Parameters:
#         -- Point (float): Starting point value to create the arrow.
        
#         -- Percentage (float): Percentage to compute the final point of the arrow. 
        
#         -- x_tail (int): Number of points to compose the ScatterArrow.

#         Return:
#         -- (zip iterable object): Interable object with tuples of dates and values. (X,Y)
#         """
#         if self.Flag == 1:
#             y_values = np.linspace(self.Value,self.Value*(1+percentage),x_tail)
#         elif self.Flag == 2:
#             y_values = np.linspace(self.Value,self.Value*(1-percentage),x_tail)
#         else:
#             y_values = np.linspace(self.Value,self.Value,x_tail)
#             x_tail = 1
#         dates = pd.date_range(start=self.Date, end=self.Date + dt.timedelta(seconds=x_tail), periods=x_tail)

#         return zip(dates.values,y_values)
    
#     def CreateHorizontalLine(self,x_tail=3):
#         assert self.Delta is not None, 'If the "CreateHorizontalLine" wants to be used, a resolution should be passed.\
#                      Either a timedelta object or an "int" between 2-4. In object {0}'.format(self.Name)
#         """
#         Function to create the positions of the X and Y axis for the fractals.
        
#         Input Parameters:
#         -- Point (float): Starting point value to create the arrow.
        
#         -- x_tail (int): Number of points to compose the ScatterArrow.

#         Return:
#         -- (zip iterable object): Interable object with tuples of dates and values. (X,Y)
#         """
#         dates = pd.date_range(start=self.Date + dt.timedelta(seconds=1) 
#                             ,end=self.Date + self.Delta*x_tail
#                             ,periods=x_tail)
#         values = np.array([self.Value for i in range(x_tail)])
#         return zip(dates,values)
    
    
# import tpqoa # already handled above
import pandas as pd
import datetime as dt
from collections import deque
# Make sure PatternIndicator is imported or defined somewhere in your project.
# from your_module import PatternIndicator  

# class HeikinAshiBar(PatternIndicator, tpqoa.tpqoa): # removed tpqoa dependency
class HeikinAshiBar(PatternIndicator):

    """

    Custom Heikin-Ashi indicator.

    

    This indicator computes Heikin-Ashi bars based on the rules:

    https://www.investopedia.com/trading/heikin-ashi-better-candlestick/

    

    Parameters:

        conf_file (str): Path to the configuration file (e.g., "oanda.cfg").

        instrument (str): Financial instrument (e.g., "EUR_USD").

        start (str): Start date (YYYY-MM-DD) for fetching data.

        end (str): End date (YYYY-MM-DD) for fetching data.

        granularity (str): Data granularity (e.g., "D" for daily).

        price (str): Price type ("A" for ask or "B" for bid).

    """

    def __init__(self, conf_file, instrument, start, end, granularity, price):

        # Initialize PatternIndicator with a chosen name and the instrument

        PatternIndicator.__init__(self, "Heikin-Ashi", instrument, resolution=None, research=True)

        # Initialize the OANDA API connection

        # tpqoa.tpqoa.__init__(self, conf_file) # Removed for local use

        

        # Store parameters

        self.instrument = instrument

        self.start = start

        self.end = end

        self.granularity = granularity

        self.price = price

        

        # Optionally, you can fetch and prepare data here if needed:

        # self.get_data()

        # self.prepare_data()

        

        # Initialize Heikin-Ashi specific variables

        self.EndTime = dt.datetime.min

        self.Open = 0

        self.Close = 0

        self.High = 0

        self.Low = 0

        self.bar = deque(maxlen=2)

        self.direction = deque(maxlen=2)

        self.history = deque(maxlen=5)

        self.HAwarning_bear = True

        self.HAwarning_bull = False



    def get_ObjectDictionary(self):
        objDict = {}
        objDict['HI_Value'] = self.Value
        objDict['Time'] = self.Time
        objDict['Flag'] = self.Flag
        objDict['EndTime'] = self.EndTime
        objDict['Open'] = self.Open
        objDict['Close'] = self.Close
        objDict['High'] = self.High
        objDict['Low'] = self.Low
        objDict['BearWarning'] = self.HAwarning_bear
        objDict['BullWarning'] = self.HAwarning_bull
        return objDict

    @property
    def IsReady(self):
        return (len(self.bar) == self.bar.maxlen) and (len(self.direction) == self.direction.maxlen)

    def MyHeikenAshiBar(self, Time, instrument, Open, High, Low, Close):
        bar = {
            'Time': Time,
            'instrument': instrument,
            'Open': Open,
            'High': High,
            'Low': Low,
            'Close': Close
        }
        return pd.Series(bar)
   
    def Update(self, input):
        """
        Updates the Heikin-Ashi bar using an input TradeBar object.
        """
        self.Time = input.EndTime
        self.EndTime = input.EndTime
        
        self.Close = (input.Open + input.Close + input.High + input.Low) / 4.0
        self.High = max(input.High, input.Open, input.Close)
        self.Low = min(input.Low, input.Open, input.Close)

        if len(self.bar) < self.bar.maxlen:
            self.bar.append(input.Close)
            self.bar.append(input.Open)
            self.Open = sum(self.bar) / len(self.bar)
            self.direction.append(self.Close > self.Open)
            return False

        self.history.appendleft(self.MyHeikenAshiBar(self.Time, self.instrument, self.Open, self.High, self.Low, self.Close))
        
        # Compute the new Open value based on previous values.
        new_open = sum(self.bar) / len(self.bar)
        self.bar.append(input.Close)
        self.bar.append(input.Open)
        self.Open = new_open
        
        self.direction.append(self.Close > self.Open)

        self.HAwarning_bear = (self.Close > self.Open) and (input.Close < input.Open)
        self.HAwarning_bull = (self.Close < self.Open) and (input.Close > input.Open)

        self.Change_Trend_Indicator()

        if self.OnResearch:
            self.AddDataResearch(**self.get_ObjectDictionary())

        return (len(self.bar) == self.bar.maxlen) and (len(self.direction) == self.direction.maxlen)
    
    def Change_Trend_Indicator(self):
        if self.direction[0] != self.direction[-1]:
            self.Date, self.Value = self.Time, self.High
            self.Flag = 1 if self.direction[0] else 2
        else:
            self.Flag = 0


class FractalBar(PatternIndicator):
    def __init__(self,name, periods,store_fracts=4,**kwargs):
        """
        Indicator of a Bullish or Bearish tendency based on the Williams Fractals: https://www.investopedia.com/terms/f/fractal.asp.

        Inputs:
        -- Name (String): A characteristic name for the definition of the indicator.
        -- Periods (int): How many periods to the left and right compute the fractals. 
                          e.g. Periods=3, then the fractal will need seven periods to be computed,
                          three to the left, three to the right, and the middle point. [(2*periods + 1)]
                                    Further, the keys will correspondingly create a delta time to separate the points on the X-axis.
        -- store_fracts (int)= Number of dates and values of fractal generations to store.

        -- resolution (int): Set the space between points for the "CreateHorizontalLine" method from the "PatternIndicator" class.
                             Only 4 preselected options available. 2 for minute, 3 for hour and 4 for day (same as the object Resolution of QC).
                             The last option is pass a timedelta object. 

        Initialization Parameters:
        -- EndTime (datetime.datetime object): Same functionality than the Time parameters.
        -- dates (deque list): Store the dates of each point.
        -- Bearish (deque list): Store the highs of the last [(2*periods) + 1] TradeBars.
        -- Bullish (deque list): Store the lows of the last [(2*periods) + 1] TradeBars.
        -- Fracts: (zip iterable object): If there was a change in the trend, the function to compute the 
                                          positions on the X and Y axis is activated. Otherwise, NoneType Object.

           The append to the following two is on the left, so the most recent value will be in the cero position.
        -- RAM_Bullish (deque list): List to store the last <store_fracts> of bullish fractals. List of Tuples: (date,value)
        -- RAM_Bearish (deque list): List to store the last <store_fracts> of bearish fractals. List of Tuples: (date,value)
        """
        super().__init__(name,**kwargs)
        
        self.EndTime = dt.datetime.min

        if isinstance(periods, int):
            self.periods = periods
            self.Bearish = deque(maxlen=(periods * 2)+1)
            self.Bullish = deque(maxlen=(periods * 2)+1)
            self.dates = deque(maxlen=(periods * 2)+1)
        else:
            assert isinstance(periods, list), f'Parameter periods should be an\
                                                 integer or list of integers, {type(periods)} passed.'
            self.periods = periods[0] + 1
            self.Bearish = deque(maxlen=int(np.sum(periods)+1))
            self.Bullish = deque(maxlen=int(np.sum(periods)+1))
            self.dates = deque(maxlen=int(np.sum(periods)+1))

        self.fracts = None

        self.RAM_Bullish = deque(maxlen=store_fracts)
        self.RAM_Bearish = deque(maxlen=store_fracts)
        self.is_ready = False
    
    def get_ObjectDictionary(self):
        objDict = {}
        # Pattern Indicator
        objDict['Frac_Value'] = self.Value
        objDict['Time'] = self.Time
        objDict['Flag'] = self.Flag
        # Fractal
        objDict['EndTime'] = self.EndTime

        return objDict
    
    @property
    def IsReady(self):
        return (len(self.Bullish)==self.Bullish.maxlen)

    def Update(self,input):
        """ Update*

        Necessary function to automatically update the parameters based on the resolution specified 
        when the ticker associated was created or the indicator was registered.

        Input Parameters:
        -- Input (TradeBar Object): Object from which update the data.

        Returns:
        -- Boolean Object: Indicate if the necessary time to set up the indicator has finished.

        """
        self.Time = input.EndTime
        self.EndTime = input.EndTime
        
        self.dates.append(input.Time)
        self.Bearish.append(input.High)
        self.Bullish.append(input.Low)

        if len(self.Bullish) == self.Bullish.maxlen:
            self.Change_Trend_Indicator()
            if self.OnResearch:
                self.AddDataResearch(**self.get_ObjectDictionary())
                
        return (len(self.Bullish)==self.Bullish.maxlen)

    def Change_Trend_Indicator(self):
    
        # Identify if the middle data is the biggest or lower to create a bearish or bullish fractal, respectively.
        if self.Bearish[self.periods] == np.max(self.Bearish):
            self.Flag = 1
            self.Date, self.Value = self.dates[self.periods],self.Bearish[self.periods]
            # self.fracts = self.CreateHorizontalLine(self.dates[self.periods],self.Bearish[self.periods])
            self.RAM_Bearish.appendleft((self.dates[self.periods],self.Bearish[self.periods]))
        elif self.Bullish[self.periods] == np.min(self.Bullish):
            self.Flag = 2
            self.Date, self.Value = self.dates[self.periods],self.Bullish[self.periods]
            # self.fracts = self.CreateHorizontalLine(self.dates[self.periods],self.Bullish[self.periods])
            self.RAM_Bullish.appendleft((self.dates[self.periods],self.Bullish[self.periods]))
        else:
            # self.Value = 0
            self.Flag = 0

class RenkoIndicator(PatternIndicator):
    def __init__(self, name, block_points, as_percent = True, store=4,**kwargs):
        """
        Indicator of a Bullish or Bearish tendency based on the Williams Fractals: https://www.investopedia.com/terms/f/fractal.asp.

        Inputs:
        -- Name (String): A characteristic name for the definition of the indicator.
        -- resolution (int): Set the space between points for the "CreateHorizontalLine" method from the "PatternIndicator" class.
                             Only 4 preselected options available. 2 for minute, 3 for hour and 4 for day (same as the object Resolution of QC).
                             The last option is pass a timedelta object. 

        -- block_points (int, float, callable): Three possible options:
                                              Integer: Add the value to the current close of the renko to check for changes. If the 
                                              float: If as_percent [default] is True, the float numbes is taken as a percent of the price.
                                                    otherwise the float is added or substracted from the price.
                                              function: This is created for the ATR if used as indicator.
                                                        It should be callable.
        """
        super().__init__(name,**kwargs)
        
        self.EndTime = dt.datetime.min


        if callable(block_points):
            # This function enable the use of other functions or varaible objects
            # Created to be able to use an ATR object
            self.Height = block_points
            if as_percent:
                self.BlockPoints = lambda past_price, price, block: price*(1-block) < past_price < price*(1+block)
            else:
                self.BlockPoints = lambda past_price, price, block: price - block < past_price < price + block
        
        else:
            assert block_points > 0 or block_points < 0, 'block_points: zero value not valid.'
            self.Height = abs(block_points)

            if isinstance(block_points, int) or (isinstance(block_points, float) and not(as_percent)):
                self.BlockPoints = lambda past_price, price: price - block_points < past_price < price + block_points
            elif isinstance(block_points, float):
                self.BlockPoints = lambda past_price, price: price*(1-block_points) < past_price < price*(1+block_points)
        
        assert hasattr(self, 'BlockPoints'),f'The variable block_points admit an integer, decimal or callable object. {type(block_points)} passed and callable = {callable(block_points)}'
        
        self.history = deque(maxlen=store)
        self.direction = deque(maxlen=2)
        self.Yellow = False

    @property
    def IsReady(self):
        return len(self.history) == self.history.maxlen
    
    def get_ObjectDictionary(self):
        objDict = {}
        # Pattern Indicator
        objDict['High'] = self.High
        objDict['Low'] = self.Low
        objDict['Open'] = self.Open
        objDict['Close'] = self.Close
        objDict['Ren_Value'] = self.Value
        objDict['Time'] = self.Time
        objDict['Flag'] = self.Flag
        objDict['Blue'] = self.Increase
        objDict['Red'] = not(self.Increase)
        objDict['Increase'] = not(self.Increase)
        objDict['Yellow'] = self.Yellow
        objDict['Blue/Yellow'] = self.Increase and self.Yellow
        objDict['Red/Yellow'] = not(self.Increase) and self.Yellow
        # Fractal
        objDict['EndTime'] = self.EndTime

        return objDict

    def Update(self,input):
        """ Update*

        Necessary function to automatically update the parameters based on the resolution specified 
        when the ticker associated was created or the indicator was registered.

        Input Parameters:
        -- Input (TradeBar Object): Object from which update the data.

        Returns:
        -- Boolean Object: Indicate if the necessary time to set up the indicator has finished.

        """
        self.Time = input.EndTime
        
        if callable(self.Height):
            change = self.BlockPoints(self.Value,input.Close, self.Height())
        else:
            change = self.BlockPoints(self.Value,input.Close)

        if self.Value == 0:
            self.Open = self.Value
            self.Value = input.Close
            self.High = input.High
            self.Low = input.Low
            self.Close = input.Close
            return False
        elif not(change):
            print()
            self.Open = self.Value
            self.Value = input.Close
            self.High = input.High
            self.Low = input.Low
            self.Close = input.Close
            self.EndTime = input.EndTime

        self.history.append([self.Open,self.Close])

        self.Increase = self.Close > self.Open

        if len(self.history) >= 2:
            self.Yellow = self.history[-2][1] == self.history[-1][0]

        self.direction.append(self.Increase)

        if self.OnResearch:
            self.AddDataResearch(**self.get_ObjectDictionary())

        self.Change_Trend_Indicator()
        
        return True

    def Change_Trend_Indicator(self):

        if self.direction[0] != self.direction[-1]:
            # If there was a change in the trend create the arrow Objects.
            self.Date, self.Value = self.Time, self.High
            if self.direction[0] == True: # In this case the change was from Green to Red
                self.Flag = 1
            else: # Otherwise
                self.Flag = 2
        else:
            # self.Value = 0
            self.Flag = 0

            
class AcumIndicatorFlag(PatternIndicator):
    def __init__(self,PatternList, name, **kwargs):
        
        '''
        This indicator receives a dictionary of candle patterns that will be updated automatically.
        If an indicator within the dictionary is active, the "AcumIndicatorFlag" will signal with the parameter "movement" [boolean].

        Input: 

        -- PatternList [dictionary]: A dictionary 

        -- Name (String): A characteristic name for the definition of the indicator.

        -- resolution (int): Set the space between points for the "CreateHorizontalLine" method from the "PatternIndicator" class.
                             Only 4 preselected options available. 2 for minute, 3 for hour and 4 for day (same as the object Resolution of QC).
                             The last option is pass a timedelta object.

        

        Parameters:
        
        -- direction (deque list): Store boolean objects. 
                                   True if the trend is bullish and False contrary. Let us know if there was a change.

        -- movement  [boolean]: A pattern was identified.

        -- who [str]: Key of the last active indicator.
        '''
        super().__init__(name, **kwargs)

        self.myPatterns = PatternList
        self.direction = deque(maxlen=2)
        self.movement = False
        self.who = ''
    
    def get_ObjectDictionary(self):
        objDict = {}
        # Pattern Indicator
        objDict['Value'] = self.Value
        objDict['Time'] = self.Time
        objDict['Flag'] = self.Flag
        # AcumIndicator
        objDict['EndTime'] = self.EndTime
        objDict['Movement'] = self.movement
        objDict['who'] = self.who

        return objDict
    
    def Update(self,input):
        """ Update*

        Necessary function to automatically update the parameters based on the resolution specified 
        when the ticker associated was created or the indicator was registered.

        Input Parameters:
        -- Input (TradeBar Object): Object from which update the data.

        Returns:
        -- Boolean Object: Indicate if the necessary time to set up the indicator has finished.

        """
        
        self.Time = input.EndTime
        self.EndTime = input.EndTime

        for pattern in list(self.myPatterns.values()):
            pattern.Update(input)
        
        self.direction.append(input.Close > input.Open)

        is_ready = [pattern.IsReady for pattern in list(self.myPatterns.values())]
        if not(True in is_ready):
            return False
        for p in list(self.myPatterns.keys()):
            d,v = self.myPatterns[p].Current.EndTime, self.myPatterns[p].Current.Value
            trend_breack = self.direction[0] != self.direction[-1]
            if v > 0 :
                self.movement = True
                self.who = p
                self.Flag = int(trend_breack) * (int(self.direction[0]) + int(not(self.direction[0]))*2)
                if self.Flag == 1:
                    self.Value = input.High
                else:
                    self.Value = input.Close
                self.Date = d
                return True
            else:
                self.movement = False
                self.Flag = 0
            #     self.Date, self.Value = d, 0
        if self.OnResearch:
                self.AddDataResearch(**self.get_ObjectDictionary())
        return True


# class CandlePatterns(tpqoa.tpqoa): # Commented out due to pandas_ta dependency
class CandlePatterns:

    ''' Candlestick pattern indicator using pandas-ta. '''

    def __init__(self, conf_file, instrument, start, end, granularity, price):
        super().__init__(conf_file)  # Initialize OANDA API connection

        # Store parameters
        self.instrument = instrument
        self.start = start
        self.end = end
        self.granularity = granularity
        self.price = price

        # Fetch and prepare data
        self.get_data()
        self.prepare_data()
        self.add_candlestick_patterns()

    def get_data(self):
        ''' Retrieves (from OANDA) and prepares the data '''
        bid = self.get_history(
            instrument=self.instrument, 
            start=self.start, 
            end=self.end, 
            granularity=self.granularity, 
            price=self.price
        )
        self.data = bid

    def prepare_data(self):
        ''' Prepares the data for strategy backtesting '''
        self.data.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}, inplace=True)

    def add_candlestick_patterns(self):
        ''' Applies candlestick pattern recognition using pandas-ta '''
        pass
        # self.data['bearish_engulfing'] = ta.cdl_engulfing(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        # ... (commented out rest)
        self.data['bullish_engulfing'] = ta.cdl_engulfing(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['doji'] = ta.cdl_doji(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['inverted_hammer'] = ta.cdl_inverted_hammer(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['bearish_harami'] = ta.cdl_harami(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['bullish_harami'] = ta.cdl_harami(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['dark_cloud_cover'] = ta.cdl_darkcloudcover(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['dragonfly_doji'] = ta.cdl_dragonflydoji(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['gravestone_doji'] = ta.cdl_gravestonedoji(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['hammer'] = ta.cdl_hammer(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['hanging_man'] = ta.cdl_hangingman(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['morning_star'] = ta.cdl_morningstar(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['morning_star_doji'] = ta.cdl_morningdojistar(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['piercing_pattern'] = ta.cdl_piercing(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['shooting_star'] = ta.cdl_shootingstar(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        self.data['star'] = ta.cdl_star(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])


        # self.data = self.data.drop(self.data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8]], axis=1)    
class VolumeIndicators(tpqoa.tpqoa):
    ''' Volume Indicators Class - Retrieves and Computes Volume-Based Indicators.
    '''
    def __init__(self, conf_file, instrument, start, end, granularity, price):
        super().__init__(conf_file)  # Initialize tpqoa with the config file
        
        # Store parameters
        self.instrument = instrument
        self.start = start
        self.end = end
        self.granularity = granularity
        self.price = price  # "A" or "B"
        
        # Fetch & Prepare Data
        self.get_data()
        self.prepare_data()
        self.add_obv()
        self.add_vwap()
        self.dropper()

    def get_data(self):
        ''' Retrieves historical price data from OANDA '''
        bid = self.get_history(instrument=self.instrument, start=self.start, 
                               end=self.end, granularity=self.granularity, price=self.price)
        self.data = bid

    def prepare_data(self):
        ''' Prepares the data for strategy backtesting '''
        data = self.data.copy()
        data.rename(columns={'o':'Open', 'h':'High', 'l':'Low', 'c':'Close'}, inplace=True)
        self.data = data
    
    def add_obv(self):
        ''' Computes On-Balance Volume (OBV) '''
        self.data["OBV"] = (np.sign(self.data["Close"].diff()) * self.data["volume"]).fillna(0).cumsum()
        
    def add_vwap(self):
        ''' Computes the Volume-Weighted Average Price (VWAP) '''
        volume = self.data['volume']
        price = self.data['Close']
        self.data['VWAP'] = ((volume * price).cumsum() / volume.cumsum()).ffill()
        
    def dropper(self):
        ''' Drops unnecessary c'''
        self.data = self.data.drop(self.data.columns[[0, 1, 2, 3, 4, 5]], axis=1)

    
