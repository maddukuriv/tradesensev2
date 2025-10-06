

import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime, timedelta
from utils.constants import SUPABASE_URL, SUPABASE_KEY, Largecap, Midcap, Smallcap, Indices, crypto_largecap, crypto_midcap
import talib
import numpy as np
import pandas as pd
from ta.trend import AroonIndicator
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.linear_model import Ridge
import ta



# Connect to Supabase
from supabase import create_client, Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Fetch stock data
def get_stock_data(ticker):
    """Fetch stock data from Supabase (limited to last 1 year)"""
    try:
        all_data = []
        page = 1
        while True:
            response = (
                supabase.table("stock_data")
                .select("*")
                .filter("ticker", "eq", ticker)
                .range((page - 1) * 1000, page * 1000 - 1)
                .execute()
            )
            if response.data:
                all_data.extend(response.data)
                page += 1
            else:
                break

        if all_data:
            df = pd.DataFrame(all_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.drop_duplicates(subset=['date'], keep='first')
                df = df.sort_values(by='date', ascending=True)
                df.set_index('date', inplace=True)
                
                # ✅ Filter for the past 1 year
                one_year_ago = datetime.now() - timedelta(days=365)
                df = df[df.index >= one_year_ago]

                df.columns = list(map(lambda x: x.capitalize(), df.columns))
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
    
def get_stock_data_all(ticker):
    """Fetch stock data from Supabase (limited to last 1 year)"""
    try:
        all_data = []
        page = 1
        while True:
            response = (
                supabase.table("stock_data")
                .select("*")
                .filter("ticker", "eq", ticker)
                .range((page - 1) * 1000, page * 1000 - 1)
                .execute()
            )
            if response.data:
                all_data.extend(response.data)
                page += 1
            else:
                break

        if all_data:
            df = pd.DataFrame(all_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.drop_duplicates(subset=['date'], keep='first')
                df = df.sort_values(by='date', ascending=True)
                df.set_index('date', inplace=True)
                

                df.columns = list(map(lambda x: x.capitalize(), df.columns))
                df = df.drop(columns=["Id", "Adj_close", "Ticker"])
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# ---------------------------
# 2. Candlestick patterns & weights
# ---------------------------
all_patterns = [
 'CDL2CROWS','CDL3BLACKCROWS','CDL3INSIDE','CDL3LINESTRIKE','CDL3OUTSIDE',
 'CDL3STARSINSOUTH','CDL3WHITESOLDIERS','CDLABANDONEDBABY','CDLADVANCEBLOCK',
 'CDLBELTHOLD','CDLBREAKAWAY','CDLCLOSINGMARUBOZU','CDLCONCEALBABYSWALL',
 'CDLCOUNTERATTACK','CDLDARKCLOUDCOVER','CDLDOJI','CDLDOJISTAR',
 'CDLDRAGONFLYDOJI','CDLENGULFING','CDLEVENINGDOJISTAR','CDLEVENINGSTAR',
 'CDLGAPSIDESIDEWHITE','CDLGRAVESTONEDOJI','CDLHAMMER','CDLHANGINGMAN',
 'CDLHARAMI','CDLHARAMICROSS','CDLHIGHWAVE','CDLHIKKAKE','CDLHIKKAKEMOD',
 'CDLHOMINGPIGEON','CDLIDENTICAL3CROWS','CDLINNECK','CDLINVERTEDHAMMER',
 'CDLKICKING','CDLKICKINGBYLENGTH','CDLLADDERBOTTOM','CDLLONGLEGGEDDOJI',
 'CDLLONGLINE','CDLMARUBOZU','CDLMATCHINGLOW','CDLMATHOLD',
 'CDLMORNINGDOJISTAR','CDLMORNINGSTAR','CDLONNECK','CDLPIERCING',
 'CDLRICKSHAWMAN','CDLRISEFALL3METHODS','CDLSEPARATINGLINES','CDLSHOOTINGSTAR',
 'CDLSHORTLINE','CDLSPINNINGTOP','CDLSTALLEDPATTERN','CDLSTICKSANDWICH',
 'CDLTAKURI','CDLTASUKIGAP','CDLTHRUSTING','CDLTRISTAR','CDLUNIQUE3RIVER',
 'CDLUPSIDEGAP2CROWS','CDLXSIDEGAP3METHODS'
]

pattern_weights = {
    "CDLMORNINGSTAR": 2, "CDLMORNINGDOJISTAR": 2, "CDL3WHITESOLDIERS": 2,
    "CDLENGULFING_BULL": 2, "CDLHAMMER": 2, "CDLINVERTEDHAMMER": 2,
    "CDLPIERCING": 2, "CDLMATHOLD": 2, "CDLLADDERBOTTOM": 2,
    "CDLABANDONEDBABY_BULL": 2, "CDLTAKURI": 2, "CDLUNIQUE3RIVER": 2,
    "CDLMATCHINGLOW": 2,

    "CDL3BLACKCROWS": -2, "CDLIDENTICAL3CROWS": -2, "CDLEVENINGSTAR": -2,
    "CDLEVENINGDOJISTAR": -2, "CDLDARKCLOUDCOVER": -2, "CDLENGULFING_BEAR": -2,
    "CDLSHOOTINGSTAR": -2, "CDLHANGINGMAN": -2, "CDL2CROWS": -2,
    "CDLUPSIDEGAP2CROWS": -2,

    "CDL3INSIDE_BULL": 1, "CDL3INSIDE_BEAR": -1, "CDL3OUTSIDE_BULL": 1,
    "CDL3OUTSIDE_BEAR": -1, "CDLADVANCEBLOCK": -1, "CDLSTALLEDPATTERN": -1,
    "CDLCOUNTERATTACK_BULL": 1, "CDLCOUNTERATTACK_BEAR": -1,
    "CDLSEPARATINGLINES_BULL": 1, "CDLSEPARATINGLINES_BEAR": -1,
    "CDLABANDONEDBABY_BEAR": -1, "CDLSTICKSANDWICH": 1, "CDLBREAKAWAY": 1,

    "CDLBELTHOLD": 0.5
}



# ---------------------------
# 4. Screener function
# ---------------------------
def analyze_ticker(ticker):
    df = get_stock_data(ticker)
    if df.empty:
        return None

    try:
        # Ensure Date column exists (even if df has index = date)
        if "Date" not in df.columns:
            if "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"])
            elif df.index.name in ["date", "Date"]:
                df = df.reset_index().rename(columns={df.index.name: "Date"})
            else:
                df = df.reset_index().rename(columns={df.index.name: "Date"})

        # Apply TA-Lib patterns
        for p in all_patterns:
            func = getattr(talib, p)
            df[p] = func(df["Open"], df["High"], df["Low"], df["Close"])

        # Weighted scoring
        def get_day_score(row):
            score = 0
            for p in all_patterns:
                val = row[p]
                if val != 0:
                    if p == "CDLENGULFING":
                        score += 2 if val > 0 else -2
                    elif p in ["CDL3INSIDE","CDL3OUTSIDE","CDLCOUNTERATTACK",
                               "CDLABANDONEDBABY","CDLSEPARATINGLINES"]:
                        score += 1 if val > 0 else -1
                    else:
                        score += pattern_weights.get(p, 0)
            return score

        df["PatternScore"] = df.apply(get_day_score, axis=1)
        df["Signal"] = df["PatternScore"].apply(lambda s: "BUY" if s > 0 else ("SELL" if s < 0 else "HOLD"))

        # Occurred patterns
        def get_patterns(row):
            occurred = []
            for p in all_patterns:
                if row[p] != 0:
                    if p == "CDLENGULFING":
                        occurred.append("CDLENGULFING_BULL" if row[p] > 0 else "CDLENGULFING_BEAR")
                    elif p in ["CDL3INSIDE","CDL3OUTSIDE","CDLCOUNTERATTACK",
                               "CDLABANDONEDBABY","CDLSEPARATINGLINES"]:
                        occurred.append(p + ("_BULL" if row[p] > 0 else "_BEAR"))
                    else:
                        occurred.append(p)
            return ",".join(occurred)

        df["OccurredPatterns"] = df.apply(get_patterns, axis=1)

        # Last 5 days BUY/SELL
        signals = df.tail(5).reset_index(drop=True)
        signals = signals[signals["Signal"].isin(["BUY", "SELL"])]

        if not signals.empty:
            signals = signals[["Date", "Close", "OccurredPatterns", "PatternScore", "Signal"]]
            signals.insert(0, "Ticker", ticker)
            return signals

    except Exception as e:
        st.error(f"Error with {ticker}: {e}")
        return None

    return None



# RSI calculation
def rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Add RSI column
def calculate_rsi(df):
    df['RSI'] = rsi(df['Close'])
    return df

# Function to calculate Momentum
def calculate_momentum(df, period=10):
    df['Momentum'] = df['Close'] - df['Close'].shift(period)
    return df

# Function to calculate Rate of Change (ROC)
def calculate_roc(df, period=12):
    df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
    return df

# Function to calculate TRIX
def calculate_trix(df, trix_length=15, signal_length=9):
    ema1 = df['Close'].ewm(span=trix_length, adjust=False).mean()
    ema2 = ema1.ewm(span=trix_length, adjust=False).mean()
    ema3 = ema2.ewm(span=trix_length, adjust=False).mean()
    df['TRIX'] = ema3.pct_change() * 100
    df['TRIX_Signal'] = df['TRIX'].ewm(span=signal_length, adjust=False).mean()
    return df

# Function to calculate TSI
def calculate_tsi(df, long=25, short=13):
    delta = df['Close'].diff()
    abs_delta = delta.abs()
    ema1 = delta.ewm(span=long, adjust=False).mean()
    ema2 = ema1.ewm(span=short, adjust=False).mean()
    abs_ema1 = abs_delta.ewm(span=long, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=short, adjust=False).mean()
    df['TSI'] = 100 * (ema2 / abs_ema2)
    return df

# Function to calculate Stochastic Oscillator
def calculate_stochastic(df, k_period=14, d_period=3):
    df['Low_Min'] = df['Low'].rolling(window=k_period).min()
    df['High_Max'] = df['High'].rolling(window=k_period).max()
    df['%K'] = ((df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min'])) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df


# Function to calculate Connors RSI
def calculate_connors_rsi(df, rsi_period=3, streak_rsi_period=2, pr_period=100):
    # RSI(3)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Streak calculation
    streak = []
    current_streak = 0
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            current_streak = current_streak + 1 if current_streak >= 0 else 1
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            current_streak = current_streak - 1 if current_streak <= 0 else -1
        else:
            current_streak = 0
        streak.append(current_streak)
    streak = [0] + streak
    df['Streak'] = streak

    # RSI of the streak
    delta_streak = df['Streak'].diff()
    gain_streak = delta_streak.clip(lower=0)
    loss_streak = -delta_streak.clip(upper=0)
    avg_gain_streak = gain_streak.rolling(streak_rsi_period).mean()
    avg_loss_streak = loss_streak.rolling(streak_rsi_period).mean()
    rs_streak = avg_gain_streak / avg_loss_streak
    df['Streak_RSI'] = 100 - (100 / (1 + rs_streak))

    # Percent Rank of 1-day ROC
    roc_1 = df['Close'].pct_change() * 100
    df['PercentRank'] = roc_1.rolling(pr_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Connors RSI
    df['ConnorsRSI'] = (df['RSI'] + df['Streak_RSI'] + df['PercentRank']) / 3
    return df

# VWAP
def calculate_vwap(df):
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    return df

# MFI
def calculate_mfi(df, window=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    # Positive and negative money flow
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)
    
    pos_mf_sum = positive_flow.rolling(window=window).sum()
    neg_mf_sum = negative_flow.rolling(window=window).sum()
    
    # Avoid division by zero
    mfr = pos_mf_sum / neg_mf_sum.replace(0, pd.NA)
    mfi = 100 - (100 / (1 + mfr))
    
    df['MFI'] = mfi
    return df

# Volume Indicators
def calculate_obv(df):
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

def calculate_ad_line(df):
    ad = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) /
          (df['High'] - df['Low']).replace(0, np.nan)) * df['Volume']
    df['AD'] = ad.fillna(0).cumsum()
    return df

def calculate_cmf(df, window=20):
    # Money Flow Multiplier
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)  # handle division by zero

    # Money Flow Volume
    mf_volume = mf_multiplier * df['Volume']

    # Rolling sums
    rolling_mf = mf_volume.rolling(window=window).sum()
    rolling_vol = df['Volume'].rolling(window=window).sum()

    # CMF
    df['CMF'] = (rolling_mf / rolling_vol).fillna(0)
    return df

def calculate_vwma(df, window=20):
    df['VWMA'] = (df['Close'] * df['Volume']).rolling(window).sum() / df['Volume'].rolling(window).sum()
    return df


# Function to calculate ADX, +DI, -DI
def calculate_adx(df, window=14):
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
    df['ADX'] = adx.adx()
    df['+DI'] = adx.adx_pos()
    df['-DI'] = adx.adx_neg()
    return df

# Function to calculate Hull MA
def calculate_hma(df, period=20):
    df = df.copy()
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))

    wma_half = df['Close'].rolling(half_period).mean()
    wma_full = df['Close'].rolling(period).mean()

    hma = (2 * wma_half - wma_full).rolling(sqrt_period).mean()
    df['HMA'] = hma
    return df

# Function to calculate SuperTrend
def calculate_supertrend(df, period=10, multiplier=3):
    df = df.copy()
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=period)
    df['ATR'] = atr.average_true_range()
    
    hl2 = (df['High'] + df['Low']) / 2
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])
    
    df['SuperTrend'] = np.nan
    in_uptrend = True
    
    for i in range(period, len(df)):
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i-1]:
            in_uptrend = True
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i-1]:
            in_uptrend = False
        
        df.loc[df.index[i], 'SuperTrend'] = (
            df['LowerBand'].iloc[i] if in_uptrend else df['UpperBand'].iloc[i]
        )
    
    return df

# Function to calculate Ichimoku
def calculate_ichimoku(df):
    df = df.copy()
    ichimoku = IchimokuIndicator(
        high=df['High'],
        low=df['Low'],
        window1=9,
        window2=26,
        window3=52
    )
    df['Tenkan'] = ichimoku.ichimoku_conversion_line()
    df['Kijun'] = ichimoku.ichimoku_base_line()
    df['SenkouA'] = ichimoku.ichimoku_a()
    df['SenkouB'] = ichimoku.ichimoku_b()
    df['Chikou'] = df['Close'].shift(-26)
    
    # Cloud boundaries
    df['Cloud_Upper'] = df[['SenkouA', 'SenkouB']].max(axis=1)
    df['Cloud_Lower'] = df[['SenkouA', 'SenkouB']].min(axis=1)
    return df

# Function to calculate GMMA
def calculate_gmma(df, short_windows=[3,5,8,10,12,15], long_windows=[30,35,40,45,50,60]):
    df = df.copy()
    for w in short_windows:
        df[f'SMA_short_{w}'] = df['Close'].rolling(w).mean()
    for w in long_windows:
        df[f'SMA_long_{w}'] = df['Close'].rolling(w).mean()
        
    # Aggregate short and long
    df['GMMA_Short'] = df[[f'SMA_short_{w}' for w in short_windows]].mean(axis=1)
    df['GMMA_Long'] = df[[f'SMA_long_{w}' for w in long_windows]].mean(axis=1)
    
    return df

# Function to calculate MACD
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal_Line']
    return df


# Function to calculate Aroon Up and Aroon Down
def calculate_aroon(df, window=14):
    aroon = AroonIndicator(high=df['High'], low=df['Low'], window=window)
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()
    return df


# Filter momentum trend stocks
def filter_momentum_trend(tickers, min_score=3):
    filtered_stocks = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker)
            if df.empty or 'Close' not in df.columns or len(df) < 20:
                continue

            df = calculate_rsi(df)
            df = calculate_momentum(df)
            df = calculate_roc(df)
            df = calculate_trix(df)
            df = calculate_tsi(df)
            df.dropna(subset=['RSI', 'Momentum', 'ROC', 'TRIX', 'TRIX_Signal', 'TSI'], inplace=True)

            # Look at last 5 days
            last_5 = df.tail(5)
            for i in range(1, len(last_5)):
                prev = last_5.iloc[i-1]
                curr = last_5.iloc[i]

                score = 0
                # RSI: crosses 50 from below
                if prev['RSI'] < 50 and curr['RSI'] > 50:
                    score += 1
                # Momentum positive and rising
                if curr['Momentum'] > 0 and curr['Momentum'] > prev['Momentum']:
                    score += 1
                # ROC crosses 0
                if prev['ROC'] < 0 and curr['ROC'] > 0:
                    score += 1
                # TRIX crosses above signal
                if prev['TRIX'] < prev['TRIX_Signal'] and curr['TRIX'] > curr['TRIX_Signal']:
                    score += 1
                # TSI positive and rising
                if curr['TSI'] > 0 and curr['TSI'] > prev['TSI']:
                    score += 1

                if score >= min_score:
                    filtered_stocks.append({
                        'Ticker': ticker,
                        'Date': curr.name.strftime('%Y-%m-%d') if isinstance(curr.name, pd.Timestamp) else curr.name,
                        'RSI': round(curr['RSI'], 2),
                        'Momentum': round(curr['Momentum'], 2),
                        'ROC': round(curr['ROC'], 2),
                        'TRIX': round(curr['TRIX'], 4),
                        'TRIX_Signal': round(curr['TRIX_Signal'], 4),
                        'TSI': round(curr['TSI'], 2),
                        'Close': round(curr['Close'], 2),
                        'Score': score
                    })
                    break  # take first matching day in last 5 days

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(filtered_stocks)



# Momentum + Mean Reversion Filter
def filter_momentum_mean_reversion(tickers, min_score=1):
    result = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker)

            if df.empty or 'Close' not in df.columns or len(df) < 120:
                continue

            df = calculate_stochastic(df).dropna(subset=['%K', '%D'])
            df = calculate_connors_rsi(df).dropna(subset=['ConnorsRSI'])

            # Check last 5 days
            for i in range(-6, -1):
                prev = df.iloc[i - 1]
                curr = df.iloc[i]

                score = 0
                # Condition 1: Stochastic crossover in oversold region
                if (prev['%K'] < prev['%D']) and (curr['%K'] > curr['%D']) and (curr['%K'] < 20):
                    score += 1
                # Condition 2: ConnorsRSI < 20 and rising
                if (curr['ConnorsRSI'] < 20) and (curr['ConnorsRSI'] > prev['ConnorsRSI']):
                    score += 1

                if score >= min_score:
                    result.append({
                        'Ticker': ticker,
                        'Date': curr.name.strftime('%Y-%m-%d') if isinstance(curr.name, pd.Timestamp) else curr.name,
                        '%K': round(curr['%K'], 2),
                        '%D': round(curr['%D'], 2),
                        'ConnorsRSI': round(curr['ConnorsRSI'], 2),
                        'Close': round(curr['Close'], 2),
                        'Score': score
                    })
                    break  # log one valid instance per ticker

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(result)

# Filter function with 5-day rising confirmation
def filter_volume_confirmation(tickers, window=5, min_score=2):
    filtered_stocks = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker)
            if df.empty or 'Close' not in df.columns or len(df) < window + 1:
                continue

            df = calculate_obv(df)
            df = calculate_ad_line(df)
            df = calculate_cmf(df)
            df = calculate_vwma(df)

            # Look at last 'window' days
            last_n = df.tail(window)
            for i in range(1, len(last_n)):
                score = 0

                # OBV rising trend
                obv_diff = last_n['OBV'].diff().dropna()
                obv_rising_days = (obv_diff > 0).sum()
                price_pct_change = (last_n['Close'].iloc[-1] - last_n['Close'].iloc[0]) / last_n['Close'].iloc[0] * 100
                if obv_rising_days >= 3 and price_pct_change >= -1:  # stable or slightly rising
                    score += 1

                # A/D rising trend
                ad_diff = last_n['AD'].diff().dropna()
                ad_rising_days = (ad_diff > 0).sum()
                if ad_rising_days >= 3 and price_pct_change > 0:
                    score += 1

                # CMF positive and rising
                cmf_diff = last_n['CMF'].diff().dropna()
                cmf_rising_days = (cmf_diff > 0).sum()
                if last_n['CMF'].iloc[-1] > 0 and cmf_rising_days >= 3:
                    score += 1

                # VWMA rising and price above VWMA
                vwma_diff = last_n['VWMA'].diff().dropna()
                if (last_n['Close'].iloc[-1] > last_n['VWMA'].iloc[-1]) and (vwma_diff[-3:] > 0).all():
                    score += 1

                if score >= min_score:
                    filtered_stocks.append({
                        'Ticker': ticker,
                        'Date': last_n.index[-1].strftime('%Y-%m-%d') if isinstance(last_n.index[-1], pd.Timestamp) else last_n.index[-1],
                        'OBV': round(last_n['OBV'].iloc[-1], 2),
                        'AD': round(last_n['AD'].iloc[-1], 2),
                        'CMF': round(last_n['CMF'].iloc[-1], 4),
                        'VWMA': round(last_n['VWMA'].iloc[-1], 4),
                        'Close': round(last_n['Close'].iloc[-1], 2),
                        'Score': score
                    })
                    break  # take first qualifying match in last 'window' days

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(filtered_stocks)

# Filter function
def filter_volume_mean_reversion(tickers, window=5, min_score=1):
    filtered_stocks = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker)
            if df.empty or 'Close' not in df.columns or len(df) < max(window, 14):
                continue

            df = calculate_vwap(df)
            df = calculate_mfi(df)

            # Last N days for mean reversion checks
            last_n = df.tail(window)

            for i in range(1, len(last_n)):
                prev = last_n.iloc[i-1]
                curr = last_n.iloc[i]

                score = 0

                # VWAP: Price crossing above VWAP
                if prev['Close'] < prev['VWAP'] and curr['Close'] > curr['VWAP']:
                    score += 1

                # MFI: < 20 and rising in last 'window' days
                mfi_diff = last_n['MFI'].diff().dropna()
                rising_days = (mfi_diff > 0).sum()
                if last_n['MFI'].iloc[-1] < 20 and rising_days >= 3:
                    score += 1

                if score >= min_score:
                    filtered_stocks.append({
                        'Ticker': ticker,
                        'Date': curr.name.strftime('%Y-%m-%d') if isinstance(curr.name, pd.Timestamp) else curr.name,
                        'VWAP': round(curr['VWAP'], 2),
                        'MFI': round(curr['MFI'], 2),
                        'Close': round(curr['Close'], 2),
                        'Score': score
                    })
                    break  # take first qualifying day in last 'window' days

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(filtered_stocks)


def filter_trend_reversal(tickers, min_score=1):
    filtered_stocks = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker)
            if df.empty or 'Close' not in df.columns or len(df) < 20:
                continue

            df = calculate_macd(df)
            df = calculate_aroon(df)

            # Look at last 5 days
            last_5 = df.tail(5)
            for i in range(1, len(last_5)):
                prev = last_5.iloc[i - 1]
                curr = last_5.iloc[i]

                score = 0

                # MACD crossover
                if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
                    score += 1

                # Aroon confirmation
                if df['Aroon_Up'].iloc[-1] > 70 and df['Aroon_Down'].iloc[-1] < 30:
                    score += 1

                if score >= min_score:
                    filtered_stocks.append({
                        'Ticker': ticker,
                        'Date': curr.name.strftime('%Y-%m-%d') if isinstance(curr.name, pd.Timestamp) else curr.name,
                        'MACD': round(curr['MACD'], 2),
                        'Signal_Line': round(curr['Signal_Line'], 2),
                        'Aroon_Up': round(curr['Aroon_Up'], 2),
                        'Aroon_Down': round(curr['Aroon_Down'], 2),
                        'Close': round(curr['Close'], 2),
                        'Score': score
                    })
                    break  # take first matching day in last 5 days

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(filtered_stocks)

# Trend continuation filter
def filter_trend_continuation(tickers, min_score=3):
    filtered_stocks = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker)
            if df.empty or 'Close' not in df.columns or len(df) < 60:
                continue

            df = calculate_adx(df)
            df = calculate_hma(df)
            df = calculate_supertrend(df)
            df = calculate_ichimoku(df)
            df = calculate_gmma(df)
            
            last_5 = df.tail(5)
            for i in range(1, len(last_5)):
                prev = last_5.iloc[i - 1]
                curr = last_5.iloc[i]

                score = 0
                if curr['ADX'] > 25 and curr['+DI'] > curr['-DI']:
                    score += 1
                if curr['Close'] > curr['SuperTrend']:
                    score += 1
                if curr['Close'] > curr['HMA'] and curr['HMA'] > prev['HMA']:
                    score += 1
                if (curr['Close'] > curr['Cloud_Upper'] and 
                    curr['Tenkan'] > curr['Kijun'] and 
                    curr['Chikou'] > df['Close'].iloc[i-26] if i-26 >= 0 else False):
                    score += 1
                if curr['GMMA_Short'] > curr['GMMA_Long']:
                    score += 1

                if score >= min_score:
                    filtered_stocks.append({
                        'Ticker': ticker,
                        'Date': curr.name.strftime('%Y-%m-%d') if isinstance(curr.name, pd.Timestamp) else curr.name,
                        'ADX': round(curr['ADX'], 2),
                        'SuperTrend': round(curr['SuperTrend'], 2) if not pd.isna(curr['SuperTrend']) else None,
                        'HMA': round(curr['HMA'], 2) if not pd.isna(curr['HMA']) else None,
                        'GMMA_Short': round(curr['GMMA_Short'], 2),
                        'GMMA_Long': round(curr['GMMA_Long'], 2),
                        'Close': round(curr['Close'], 2),
                        'Score': score
                    })
                    break  # stop after first qualifying match in last 5 days

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(filtered_stocks)




# ----------------------
# Helper: Metrics function
# ----------------------
def regression_metrics(y_true, y_pred, dataset_name=""):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-8, y_true))) * 100
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}



# ----------------------
# Function to run pipeline for one ticker
# ----------------------
def forecast_ticker(ticker):
    data = get_stock_data_all(ticker)
    if data.empty:
        return None  # Skip if no data
    
    # Technical indicators
    data['SMA_20'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = SMAIndicator(data['Close'], window=200).sma_indicator()

    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()

    data['RSI'] = RSIIndicator(data['Close']).rsi()

    bb = BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()

    adx = ADXIndicator(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx.adx()

    data['ATR'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()

    stoch = StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['Stoch'] = stoch.stoch()
    data['Stoch_Signal'] = stoch.stoch_signal()

    data['H-L'] = data['High'] - data['Low']
    data['O-C'] = data['Close'] - data['Open']
    data['7_DAYS_MA'] = data['Close'].rolling(window=7).mean()
    data['14_DAYS_MA'] = data['Close'].rolling(window=14).mean()
    data['7_DAYS_STD_DEV'] = data['Close'].rolling(window=7).std()

    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    data['VWAP'] = ta.volume.volume_weighted_average_price(
        data['High'], data['Low'], data['Close'], data['Volume'], window=14
    )
    data['A/D Line'] = ta.volume.acc_dist_index(data['High'], data['Low'], data['Close'], data['Volume'])

    # Lag features
    for lag in range(1, 6):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)

    data.dropna(inplace=True)

    # Features & Target
    features = [
        'Open','High','Low','Volume','SMA_20','SMA_50','SMA_200','MACD','MACD_Signal',
        'RSI','BB_High','BB_Low','ADX','ATR','Stoch','Stoch_Signal','H-L','O-C','OBV','VWAP','A/D Line',
        '7_DAYS_MA','14_DAYS_MA','7_DAYS_STD_DEV'
    ] + [f'Lag_{lag}' for lag in range(1, 6)]

    X = data[features]
    y = data['Close'].shift(-1)  # next-day prediction
    X, y = X[:-1], y[:-1]

    # Train-Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Training (GridSearch + CV)
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    ridge = Ridge()

    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    best_model.fit(X_train, y_train)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)
    

    # Forecasting next 30 business days (naive one-step, not recursive)
    last_30_days = data[features].tail(30)
    scaled_last_30_days = scaler.transform(last_30_days)
    next_30_days_forecast = best_model.predict(scaled_last_30_days)

    forecast_df = pd.DataFrame({'Forecasted_Price': next_30_days_forecast})

    # % change calculations
    one_week_change_pct = ((forecast_df["Forecasted_Price"].iloc[4] - forecast_df["Forecasted_Price"].iloc[0]) 
                        / forecast_df["Forecasted_Price"].iloc[0]) * 100
    two_week_change_pct = ((forecast_df["Forecasted_Price"].iloc[9] - forecast_df["Forecasted_Price"].iloc[0]) 
                        / forecast_df["Forecasted_Price"].iloc[0]) * 100
    three_week_change_pct = ((forecast_df["Forecasted_Price"].iloc[14] - forecast_df["Forecasted_Price"].iloc[0]) 
                        / forecast_df["Forecasted_Price"].iloc[0]) * 100

    return {
        "Ticker": ticker,
        "1-Week Change %": round(one_week_change_pct, 2),
        "2-Week Change %": round(two_week_change_pct, 2),
        "3-Week Change %": round(three_week_change_pct, 2),
        "train_r2": round(train_metrics["R2"], 4),
        "test_r2": round(test_metrics["R2"], 4),
        "train_mae": round(train_metrics["MAE"], 4),
        "test_mae": round(test_metrics["MAE"], 4),
        "train_mape": round(train_metrics["MAPE"], 2),
        "test_mape": round(test_metrics["MAPE"], 2),
    }


def stock_screener_app():
    st.title("Stock Screener")

    # ------------------------------
    # Sidebar: Dropdowns
    # ------------------------------
    category = st.sidebar.selectbox(
        "Select Ticker Category",
        ["Largecap", "Midcap", "Smallcap", "Indices", "Crypto Largecap", "Crypto Midcap"]
    )

    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["Candlestick Patterns","Momentum", "Volume Driven", "Trend Following",  
         "Volatility Reversion", "Breakout","Mean Reversion","Swing Trading", "Arbitrage", 
        "News-Based","Machine Learning-Based","Fundamental Analysis"]
    )

    # ------------------------------
    # Category → Ticker Mapping
    # ------------------------------
    ticker_dict = {
        "Largecap": Largecap,
        "Midcap": Midcap,
        "Smallcap": Smallcap,
        "Indices": Indices,
        "Crypto Largecap": crypto_largecap,
        "Crypto Midcap": crypto_midcap
    }

    selected_ticker = ticker_dict.get(category, [])



    # ------------------------------
    # Strategy Explanation
    # ------------------------------
    if strategy == "Candlestick Patterns":
        tickers = selected_ticker
        st.write(f"Analyzing {len(tickers)} tickers in the {category} category for Candlestick patterns...")

        if tickers:
            st.subheader("Stocks with Candlestick Patterns")
            with st.spinner("Analyzing patterns..."):     
                results = []
                for t in tickers:
                    res = analyze_ticker(t)
                    if res is not None:
                        results.append(res)

                final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
                if not final_df.empty:
                    final_df = final_df.sort_values(by=["Date", "PatternScore"], ascending=[False, False])

                    # Two-column layout
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("✅ BUY Signals")
                        st.dataframe(final_df[final_df["Signal"] == "BUY"])

                    with col2:
                        st.subheader("❌ SELL Signals")
                        st.dataframe(final_df[final_df["Signal"] == "SELL"])
                else:
                    st.warning("No candlestick signals found.")

    elif strategy == "Momentum":
        tickers = selected_ticker
        st.write(f"Analyzing {len(tickers)} tickers in the {category} category for Momentum strategy...")

        if tickers:
            st.subheader("Momentum Stocks with Trend")
            with st.spinner("Analyzing momentum trends..."):
                result = filter_momentum_trend([t.strip() for t in tickers])
                
                if not result.empty:
                    st.success(f"Found {len(result)} stocks with momentum trend")
                    st.dataframe(result.sort_values(["Date","Score"], ascending=False))
                else:
                    st.info("No stocks met the momentum criteria in the last 5 days.")

            st.subheader("Momentum Stocks with Mean Reversion")
            with st.spinner("Analyzing momentum mean reversion..."):
                filtered = filter_momentum_mean_reversion(tickers, min_score=1)
                
                if not result.empty:
                    st.success(f"Found {len(filtered)} stocks with momentum trend")
                    st.dataframe(filtered.sort_values(["Date","Score"], ascending=False))

                else:
                    st.info("No stocks met the momentum criteria in the last 5 days.")


        else:
            st.warning("No tickers available for the selected category.")
    
    elif strategy == "Volume Driven":
        tickers = selected_ticker
        st.write(f"Analyzing {len(tickers)} tickers in the {category} category for Volume strategy...")

        if tickers:
            st.subheader("Volume Stocks with Trend")
            with st.spinner("Analyzing volume trends..."):
                result = filter_volume_confirmation([t.strip() for t in tickers])
                
                if not result.empty:
                    st.success(f"Found {len(result)} stocks with volume trend")
                    st.dataframe(result.sort_values(["Date","Score"], ascending=False))
                else:
                    st.info("No stocks met the volume criteria in the last 5 days.")

            st.subheader("Volume Stocks with Mean Reversion")
            with st.spinner("Analyzing volume mean reversion..."):
                filtered = filter_volume_mean_reversion(tickers, min_score=1)
                
                if not result.empty:
                    st.success(f"Found {len(filtered)} stocks with volume trend")
                    st.dataframe(filtered.sort_values(["Date","Score"], ascending=False))

                else:
                    st.info("No stocks met the volume criteria in the last 5 days.")


        else:
            st.warning("No tickers available for the selected category.")

    elif strategy == "Trend Following":                 
        tickers = selected_ticker
        st.write(f"Analyzing {len(tickers)} tickers in the {category} category for Trend strategy...")

        if tickers:
            st.subheader("Stocks with Trend Continuation")
            with st.spinner("Analyzing trend continuation..."):
                result = filter_trend_continuation([t.strip() for t in tickers])
                
                if not result.empty:
                    st.success(f"Found {len(result)} stocks with trend continuation")
                    st.dataframe(result.sort_values(["Date","Score"], ascending=False))
                else:
                    st.info("No stocks met the trend criteria in the last 5 days.")

            st.subheader("Stocks with Trend Reversal")
            with st.spinner("Analyzing trend mean reversion..."):
                filtered = filter_trend_reversal(tickers, min_score=1)
                
                if not result.empty:
                    st.success(f"Found {len(filtered)} stocks with trend reversal")
                    st.dataframe(filtered.sort_values(["Date","Score"], ascending=False))

                else:
                    st.info("No stocks met the trend criteria in the last 5 days.")


        else:
            st.warning("No tickers available for the selected category.")

    elif strategy == "Mean Reversion":
        st.write("📉 Mean reversion assumes that stock prices will revert to their historical average over time.")
         
    elif strategy == "Breakout":
        st.write("🚀 Breakout strategy involves buying stocks that break above resistance or selling those that fall below support.")
    elif strategy == "Volatility Reversion":
        st.write("📈 Volatility reversion assumes that periods of high volatility will be followed by low volatility, and vice versa.")
    elif strategy == "Swing Trading":
        st.write("🔄 Swing trading involves holding stocks for days to weeks to capture medium-term moves.")
    elif strategy == "Arbitrage":           
        st.write("🔀 Arbitrage exploits price differences of the same asset across markets, buying low and selling high.")
    elif strategy == "News-Based":
        st.write("📰 News-based trading makes decisions based on news events, earnings, or economic releases...")

    elif strategy == "Machine Learning-Based":

        tickers = selected_ticker
        st.write(f"Analyzing {len(tickers)} tickers in the {category} category for Candlestick patterns...")

        if tickers:
            st.subheader("Forecasted Weekly Stock Returns")
            with st.spinner("Forecasting patterns..."):     

                results = []

                for t in tickers:
                    try:
                        res = forecast_ticker(t)
                        if res:
                            results.append(res)
                            print(f"✅ Done: {t}")
                        else:
                            print(f"⚠️ Skipped: {t} (No Data)")
                    except Exception as e:
                        print(f"❌ Error with {t}: {e}")

                # Final result table
                forecast_df = pd.DataFrame(results)
                st.write("\n📊 Final Forecast Summary:")
                st.dataframe(forecast_df.sort_values("1-Week Change %",ascending=False))