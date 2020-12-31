import numpy as np
import pandas as pd


def interpolate_missing(data):
    df1 = data.interpolate()
    return df1


def drop_unwanted_cols(data):
    """
    Drop all the unwanted columns that are read in from the CoT spreadsheet.
    """
    df2 = data.drop(
        ['Contract_Units', 'As_of_Date_In_Form_YYMMDD',
         'CFTC_Contract_Market_Code', 'CFTC_Region_Code',
         'CFTC_Commodity_Code', 'CFTC_Market_Code', 'Open_Interest_Other',
         'NonComm_Positions_Long_Other', 'NonComm_Positions_Short_Other',
         'NonComm_Positions_Spread_Other', 'Comm_Positions_Long_Other',
         'Comm_Positions_Short_Other',
         'Tot_Rept_Positions_Long_Other', 'Tot_Rept_Positions_Short_Other',
         'NonRept_Positions_Long_Other', 'NonRept_Positions_Short_Other',
         'Pct_of_OI_NonComm_Long_Other', 'Pct_of_OI_NonComm_Short_Other',
         'Pct_of_OI_NonComm_Spread_Other', 'Pct_of_OI_Comm_Long_Other',
         'Pct_of_OI_Comm_Short_Other', 'Pct_of_OI_Tot_Rept_Long_Other',
         'Pct_of_OI_Tot_Rept_Short_Other', 'Pct_of_OI_NonRept_Long_Other',
         'Pct_of_OI_NonRept_Short_Other', 'Traders_Tot_Other',
         'Traders_NonComm_Long_Other', 'Traders_NonComm_Short_Other',
         'Traders_NonComm_Spread_Other', 'Traders_Comm_Long_Other',
         'Traders_Comm_Short_Other', 'Traders_Tot_Rept_Long_Other',
         'Traders_Tot_Rept_Short_Other', 'Conc_Gross_LE_4_TDR_Long_Other',
         'Conc_Gross_LE_4_TDR_Short_Other', 'Conc_Gross_LE_8_TDR_Long_Other',
         'Conc_Gross_LE_8_TDR_Short_Other', 'Conc_Net_LE_4_TDR_Long_Other',
         'Conc_Net_LE_4_TDR_Short_Other', 'Conc_Net_LE_8_TDR_Long_Other',
         'Conc_Net_LE_8_TDR_Short_Other'],
        axis=1)
    return df2


def smoothSeries(var, spanLength):
    """
    Exponential smoothing ("smoothSeries()")
    Params: spanLength: smoothing period
    """
    fwd = var.ewm(span=10).mean()
    bwd = var[::-1].ewm(span=10).mean()
    c = np.vstack((fwd, bwd[::-1]))
    c = np.mean(c, axis=0)
    return c


def generate_cot_data(data):
    """
    Read in the .csv, apply interpolation, dropping unwanted
    columns, apply exponential smoothing.
    """
    df1 = interpolate_missing(data)
    df2 = drop_unwanted_cols(df1)

    # create a df of just floats
    df_floats = df2.select_dtypes(include=['float64'])

    column_names = df_floats.select_dtypes(
        include=['float64']).columns
    index = np.arange(len(df_floats))

    # generate a dataframe in which to send smoothed values
    df_ = pd.DataFrame(index=index, columns=column_names)

    for i in range(df_floats.shape[1]):
        df_.iloc[:, i] = smoothSeries(df_floats.iloc[:, i], 25)
    return df_


def refined_cot_data(data):
    """
    Perform feature generation on the CoT data.
    Input: CoT dataframe generated from generate_cot_data().

    Net positioning: long positions minus short positions
    Net_positioning_1w_change: 1-week change in net positioning
    """
    net_positioning = {
        'net_comm':
        (data.Comm_Positions_Long_All - data.Comm_Positions_Short_All),
        'net_noncomm':
        (data.NonComm_Positions_Long_All - data.NonComm_Positions_Short_All),
        'net_nonrep':
        (data.NonRept_Positions_Long_All - data.NonRept_Positions_Short_All)
    }

    net_positioning_1w_change = {
        'net_comm_chg_1w':
        net_positioning['net_comm'] - net_positioning['net_comm'].shift(1),
        'net_noncomm_chg_1w':
        net_positioning['net_noncomm'] - net_positioning['net_noncomm'].shift(1),
        'net_nonrep_chg_1w':
        net_positioning['net_nonrep'] - net_positioning['net_nonrep'].shift(1)
    }

    net_positioning_1m_change = {
        'net_comm_chg_1m' : net_positioning['net_comm'] - net_positioning['net_comm'].shift(4),
        'net_noncomm_chg_1m' : net_positioning['net_noncomm'] - net_positioning['net_noncomm'].shift(4),
        'net_nonrep_chg_1m' : net_positioning['net_nonrep']  - net_positioning['net_nonrep'].shift(4)
    }
    
    net_positioning_1q_change = {
        'net_comm_chg_1q' : net_positioning['net_comm'] - net_positioning['net_comm'].shift(12),
        'net_noncomm_chg_1q' : net_positioning['net_noncomm'] - net_positioning['net_noncomm'].shift(12),
        'net_nonrep_chg_1q' : net_positioning['net_nonrep']  - net_positioning['net_nonrep'].shift(12)
    }
    
    pct_long = {
        'net_comm_long' : data.Comm_Positions_Long_All / \
            (data.Comm_Positions_Long_All + data.Comm_Positions_Short_All),
        'net_noncomm_long' : data.NonComm_Positions_Long_All / \
            (data.NonComm_Positions_Long_All + data.NonComm_Positions_Short_All),
        'net_nonrep_long' : data.NonRept_Positions_Long_All / \
            (data.NonRept_Positions_Long_All + data.NonRept_Positions_Short_All)}
    
    net_positioning = pd.DataFrame(net_positioning)
    net_positioning_1w_change = pd.DataFrame(net_positioning_1w_change)
    net_positioning_1m_change = pd.DataFrame(net_positioning_1m_change)
    net_positioning_1q_change = pd.DataFrame(net_positioning_1q_change)
    pct_long = pd.DataFrame(pct_long)

    df = pd.concat(
        [net_positioning, net_positioning_1w_change, net_positioning_1m_change,
         net_positioning_1q_change, pct_long],
        join='outer', axis=1
    )

    return df


def create_returns_variables(df, cols, period_weeks, string_append):
    """
    Keyword arguments
    :cols_to_transform: A list of columns on which to calculate
        the changes.
    :period_weeks: Number of weeks over which to calculate the change
    :string_append: A string to append to the column name.
        e.g. "_chg_1w" to imply it's a 1-week change var.
        
    Outputs
    :df: The original dataframe supplied, with additional columns
        showing the n-period percentage change of the provided
        variables.
    """
    for col in cols_to_transform:
        df[col+str(string_append)] = df[col].diff(period_weeks)
        
    return df


def create_binary_variables(df, cols_to_binarise, ma_period=50):
    """
    Creates binary variables that indicate whether a
    features' values are above their moving averages.
    
    Keyword arguments
    :df: Dataframe containing features which will be used
        to create a binary variables
    :cols_to_binarise: A list of features which will be
        binarised
    :ma_period: Input to pandas exponential weighted moving
        average; the minimum number of observations in window
        required to have a value.
        
    Outputs
    :df: The original dataframe supplied, with additional columns
        showing the whether the specified features are above
        their moving average on a given week.
    
    """
    num_rows = df.shape[0]
    
    for col in cols_to_binarise:
        for i in np.arange(num_rows):
            if df.loc[i, col] > df[col].ewm(ignore_na=False, min_periods=ma_period, com=50, adjust=True).mean()[i]:
                df.loc[i, col+str("_bin")] = 1
            else:
                df.loc[i, col+str("_bin")] = 0
    return df


# technical indicator functions
# code authored by https://github.com/voice32/stock_market_indicators/blob/master/indicators.py


def ema(data, period=10, column='Close'): # change period to alter "smoothness"
    """
    Exponential moving average
    Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    Params: 
        data: pandas DataFrame
        period: smoothing period
        column: the name of the column with values for calculating EMA in the 'data' DataFrame
    Returns:
        copy of 'data' DataFrame with 'ema[period]' column added
    """
    data['ema' + str(period)] = data[column].ewm(ignore_na=False,
                                                 min_periods=period,
                                                 com=period,
                                                 adjust=True).mean()
    
    return data


def macd(data, period_long=26, period_short=12, period_signal=9, column='Close'): 
    """
    Moving Average Convergence/Divergence Oscillator (MACD)
    Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    Params: 
        data: pandas DataFrame
        period_long: the longer period EMA (26 days recommended)
        period_short: the shorter period EMA (12 days recommended)
        period_signal: signal line EMA (9 days recommended)
        column: the name of the column with values for calculating MACD in the 'data' DataFrame

    Returns:
        copy of 'data' DataFrame with 'macd_val' and 'macd_signal_line' columns added
    """
    remove_cols = []
    if not 'ema' + str(period_long) in data.columns:
        data = ema(data, period_long)
        remove_cols.append('ema' + str(period_long))

    if not 'ema' + str(period_short) in data.columns:
        data = ema(data, period_short)
        remove_cols.append('ema' + str(period_short))

    data['macd_val'] = data['ema' + str(period_short)] - data['ema' + str(period_long)]
    data['macd_signal_line'] = data['macd_val']\
        .ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()

    data = data.drop(remove_cols, axis=1)
        
    return data


def rsi(data, periods=14, close_col='Close'):
    """
    Relative Strength Index (Note: this is a stochastic version)
    Source: https://en.wikipedia.org/wiki/Relative_strength_index
    Params: 
        data: pandas DataFrame
        periods: period for calculating momentum
        close_col: the name of the CLOSE values column

    Returns:
        copy of 'data' DataFrame with 'rsi' column added
    """
    data['rsi_u'] = 0.
    data['rsi_d'] = 0.
    data['rsi'] = 0.
    
    for index, row in data.iterrows():
        if index >= periods:
            
            prev_close = data.loc[index-periods, close_col]
            if prev_close < row[close_col]:
                data.loc[index, 'rsi_u'] = row[close_col] - prev_close
            elif prev_close > row[close_col]:
                data.loc[index, 'rsi_d'] = prev_close - row[close_col]
            
    data['rsi'] = data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() \
        / (data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() \
        + data['rsi_d'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean())
    
    data = data.drop(['rsi_u', 'rsi_d'], axis=1)
        
    return data


def williams_ad(data, high_col='High', low_col='Low', close_col='Close'):
    """
    William's Accumulation/Distribution
    Source: https://www.metastock.com/customer/resources/taaz/?p=125
    Params: 
        data: pandas DataFrame
        high_col: the name of the HIGH values column
        low_col: the name of the LOW values column
        close_col: the name of the CLOSE values column

    Returns:
        copy of 'data' DataFrame with 'williams_ad' column added
    """
    data['williams_ad'] = 0.
    
    for index, row in data.iterrows():
        if index > 0:
            prev_value = data.loc[index-1, 'williams_ad']
            prev_close = data.loc[index-1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.
                                                                                                        
            data.loc[index, 'williams_ad'] = (ad+prev_value)
        
    return data


def williams_r(data, periods=14, high_col='High', low_col='Low', close_col='Close'):
    """
    William's % R
    Source: https://www.metastock.com/customer/resources/taaz/?p=126
    Params: 
        data: pandas DataFrame
        periods: the period over which to calculate the indicator value
        high_col: the name of the HIGH values column
        low_col: the name of the LOW values column
        close_col: the name of the CLOSE values column

    Returns:
        copy of 'data' DataFrame with 'williams_r' column added
    """
    data['williams_r'] = 0.
    
    for index,row in data.iterrows():
        if index > periods:
            data.loc[index, 'williams_r'] = ((max(data[high_col][index-periods:index]) - row[close_col]) \
                                             / (max(data[high_col][index-periods:index]) \
                                                - min(data[low_col][index-periods:index])))
        
    return data


def average_true_range(data, trend_periods=14, open_col='Open', high_col='High',
                       low_col='Low', close_col='Close', drop_tr = True):
    """
    Average true range (ATR)
    Source: https://en.wikipedia.org/wiki/Average_true_range
    Params: 
        data: pandas DataFrame
        trend_periods: the over which to calculate ATR
        open_col: the name of the OPEN values column
        high_col: the name of the HIGH values column
        low_col: the name of the LOW values column
        close_col: the name of the CLOSE values column
        vol_col: the name of the VOL values column
        drop_tr: whether to drop the True Range values column from the resulting DataFrame

    Returns:
        copy of 'data' DataFrame with 'atr' (and 'true_range' if 'drop_tr' == True) column(s) added
    """
    for index, row in data.iterrows():
        prices = [row[high_col], row[low_col], row[close_col], row[open_col]]
        if index > 0:
            val1 = np.amax(prices) - np.amin(prices)
            val2 = abs(np.amax(prices) - data.loc[index - 1, close_col])
            val3 = abs(np.amin(prices) - data.loc[index - 1, close_col])
            true_range = np.amax([val1, val2, val3])

        else:
            true_range = np.amax(prices) - np.amin(prices)

        data.loc[index, 'true_range'] = true_range
    data['atr'] = data['true_range'].ewm(ignore_na=False,
                                         min_periods=0,
                                         com=trend_periods,
                                         adjust=True).mean()
    
    if drop_tr:
        data = data.drop(['true_range'], axis=1)
        
    return data


def bollinger_bands(data, trend_periods=20, close_col='Close'):
    """
    Bollinger Bands
    Source: https://en.wikipedia.org/wiki/Bollinger_Bands
    Params: 
        data: pandas DataFrame
        trend_periods: the over which to calculate BB
        close_col: the name of the CLOSE values column

    Returns:
        copy of 'data' DataFrame with 'bol_bands_middle',
        'bol_bands_upper' and 'bol_bands_lower' columns added
    """

    data['bol_bands_middle'] = data[close_col].ewm(ignore_na=False,
                                                   min_periods=0,
                                                   com=trend_periods,
                                                   adjust=True).mean()
    
    for index, row in data.iterrows():
        s = data[close_col].iloc[index - trend_periods: index]
        sums = 0
        middle_band = data.loc[index, 'bol_bands_middle']
        for e in s:
            sums += np.square(e - middle_band)

        std = np.sqrt(sums / trend_periods)
        d = 2
        upper_band = middle_band + (d * std)
        lower_band = middle_band - (d * std)

        data.loc[index, 'bol_bands_upper'] = upper_band
        data.loc[index, 'bol_bands_lower'] = lower_band

    return data


def momentum(data, period=9, close_col = 'Close'):
    """Absolute change in price over a defined period of time"""
    data['momentum'] = np.nan
    for i in np.arange(period, data.shape[0]):
        data['momentum'][i] = data[close_col][i] - data[close_col][i-period]
    return data


def pct_chg(data, period):
    """Percentage change in price over a defined period of time"""
    change = {'change_'+str(period) : data['Close'].diff(period)}
    return pd.DataFrame(list(change.values())[0]).rename(columns = {'Close': str(list(change.keys())[0])})


def create_indicators(data):
    """Return a dataframe containing all technical indicators"""
    df1 = ema(data, period=10, column='Close')
    df2 = macd(df1, period_long=26, period_short=12, period_signal=9, column='Close')
    df3 = rsi(df2, periods=14, close_col='Close')
    df4 = williams_ad(df3, high_col='High', low_col='Low', close_col='Close')
    df5 = williams_r(df4, periods=14, high_col='High', low_col='Low', close_col='Close')
    df6 = average_true_range(df5, trend_periods=14, open_col='Open', high_col='High',
                             low_col='Low', close_col='Close', drop_tr = True)
    df7 = bollinger_bands(df6, trend_periods=20, close_col='Close')
    df8 = momentum(df7, period = 9, close_col= 'Close')
    df9 = pd.concat(
        [df8, pct_chg(df8, 1), pct_chg(df8, 2), pct_chg(df8, 3),
         pct_chg(df8, 4), pct_chg(df8, 8)], join='outer', axis = 1) # add the weekly price change (%)
    return df9


def create_technical_indicators(path):
    """
    Creates a dataframe containing technical indicators for
    the asset provided. Technical indicators are functions
    of price.
    
    Technical indicators include:
        ema
        macd
        rsi
        williams_ad
        williams_r
        average_true_range
        bollinger_bands
        momentum
        pct_chg
    
    Keyword arguments:
    :path: A path to the .csv containing the futures price data
    
    Outputs:
    :technical_indicators_df: A dataframe of technical indicators.
    """
    df = pd.read_csv(path)
    df = df.dropna(how='any').reset_index(drop=True)
    
    df_floats = df.iloc[:,1:] # exclude date column
    
    # smooth series
    column_names = df_floats.select_dtypes(include=['float64']).columns
    index = np.arange(len(df_floats))
    df_smoothed = pd.DataFrame(index=index, columns=column_names)
    
    # smooth the data (open, high, low, close)
    for i in range(df_floats.shape[1]):
        # don't smooth first column because it's a date column
        df_smoothed.iloc[:,i] = smoothSeries(df_floats.iloc[:,i], 15)
        
    # create technical indicators from the smoothed series    
    technical_indicators_df = create_indicators(df_smoothed)
    
    # rejoin the date column for future merging
    technical_indicators_df = pd.concat(
        [technical_indicators_df, df.iloc[:,0]], join='outer', axis=1)
    
    # dayfirst = True, because original data is in DD/MM/YYYY format
    technical_indicators_df['Date'] = pd.to_datetime(
        technical_indicators_df['Date'], dayfirst=True)
    
    return technical_indicators_df


def ema_binary(data, column='Close', ema_column='ema_bin'):
    """
    Creates a binary variable that indicates whether EMA is
    higher or lower than the previous reading.
    """
    data[ema_column] = np.nan
    for i in np.arange(data.shape[0]):
        if data[column][i] > data['ema10'][i]:
            data.loc[i, ema_column] = 1
        else:
            data.loc[i, ema_column] = -1
    return data


def rsi_binary(data):
    """
    Creates a binary variable that indicates whether RSI is higher
    or lower than the previous reading.
    """
    data['rsi_bin'] = np.nan
    for i in np.arange(1, data.shape[0]): # goes from index 1 because [0-1] index doesn't exist
        if data['rsi'][i] > data['rsi'][i-1]:
            data.loc[i, 'rsi_bin'] = 1
        else:
            data.loc[i, 'rsi_bin'] = -1
    return data


def macd_binary(data):
    """
    Creates a binary variable that indicates whether MACD is
    higher or lower than the previous reading.
    """
    data['macd_bin'] = np.nan
    for i in np.arange(1, data.shape[0]): # goes from index 1 because [0-1] index doesn't exist
        if data.loc[i, 'macd_val'] > data.loc[i-1, 'macd_val']:
            data.loc[i, 'macd_bin'] = 1
        else:
            data.loc[i, 'macd_bin'] = -1
    return data


def momentum_binary(data, weeks):
    """
    Creates a binary variable that indicates whether price is
    higher than price i weeks ago.
    """
    data['momentum_bin'] = np.nan
    for i in np.arange(weeks, data.shape[0]): # goes from index 1 because [0-1] index doesn't exist
        if data.loc[i, 'momentum'] > data.loc[i-weeks, 'momentum']:
            data.loc[i, 'momentum_bin'] = 1
        else:
            data.loc[i,'momentum_bin'] = -1
    return data


def ema_crossover(data, period1=4, period2=12, column='Close'): # change period to alter "smoothness"
    """
    Creates a binary variable that indicates whether the
    shorter-period EMA is above the long-period EMA.
    """
    data['ema' + str(period1)] = data[column].ewm(
        ignore_na=False,
        min_periods=period1,
        com=period1, adjust=True).mean()

    data['ema' + str(period2)] = data[column].ewm(
        ignore_na=False,
        min_periods=period2,
        com=period2, adjust=True).mean()

    data['ema_cross_bin'] = np.nan
    
    for i in np.arange(period2, data.shape[0]): # goes from index 1 because [0-1] index doesn't exist
        if data.loc[i, 'ema' + str(period1)] > data.loc[i, 'ema' + str(period2)]:
            data.loc[i, 'ema_cross_bin'] = 1
        else:
            data.loc[i, 'ema_cross_bin'] = -1
    return data


def create_binary_indicators(data):
    """Merges the binary indicators together."""
    df1 = ema_binary(data)
    df2 = rsi_binary(df1)
    df3 = macd_binary(df2)
    df4 = momentum_binary(df3, weeks=8)
    df5 = ema_crossover(df4)
    
    df5['Date'] = pd.to_datetime(
        df5['Date'], dayfirst=True)
    
    # filter relevent columns
    df5[['Date','ema_bin', 'macd_bin', 'ema_cross_bin',
         'momentum_bin', 'change_1', 'change_2', 'change_3',
         'change_8']]
    
    return df5


def create_lagged_features(df, cols_to_lag):
    """
    Keyword arguments:
    :df: Dataframe containing the features to be lagged.
    :cols_to_lag: A list of the features which are
        to be lagged.
        
    Outputs:
    :df: Original dataframe with additional lagged features
        appended.
    """
    for i, column in enumerate(cols_to_lag):
        df[column+'_t-1'] = df[column].shift()
        df[column+'_t-2'] = df[column].shift(2)
        df[column+'_t-3'] = df[column].shift(3)
        df[column+'_t-3'] = df[column].shift(4)    
    return df


def create_target_variable(path, weeks=4):
    """
    Keyword arguments:
    :path: The path to the price data of variable in use.
    :weeks: Time period over which to calculate whether
        price has risen or fallen.
        
    Outputs:
        
    """
    
    df = pd.read_csv(path)
    df = df.dropna(how='any').reset_index()
    df = df[['Date','Close']]
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    if "esa" in path:
        print("found esa in path")
        df = df.iloc[:-3, :] # cut the last 3 rows to match up with the inputs
    elif "fva" in path:
        print("found fva in path")
        df= df.iloc[:-5, :] # cut the last 5 rows to match up with the inputs
    else:
        print("found dxa in path")
        df = df.iloc[:-3, :] # cut the last 3 rows to match up with the inputs
        
    # generate binary target variable 'movement'
    weeks = 4

    for i in np.arange(weeks, df.Close.shape[0]):
        if df.loc[i, 'Close'] > df.loc[i-weeks, 'Close']:
            df.loc[i-weeks, 'Movement'] = 1
        else:
            df.loc[i-weeks, 'Movement'] = 0

    df = df.drop(['Close'], axis=1)
    return df


def create_x_y(df):
    """
    Keyword arguments:
    :df: The inputs dataframe containing X values and y target.
    
    Outputs:
    :X, y: Input matrix and target vector.
    :Class frequency bar chart: shows class balance
    """
    # exclude the movement/target column
    X = df.iloc[:, 1:-1] # i.e. exclude the date and 'movement' column.
    y = df.iloc[:, -1] # drop the date
    return X, y