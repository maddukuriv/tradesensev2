import streamlit as st

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client

from utils.constants import SUPABASE_URL, SUPABASE_KEY, Largecap, Midcap, Smallcap, Indices, crypto_largecap, crypto_midcap



import warnings
warnings.filterwarnings("ignore")


tickers = Largecap + Midcap + Smallcap + Indices + crypto_largecap + crypto_midcap


# Supabase Connection
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_stock_data(ticker):
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
                one_year_ago = datetime.now() - timedelta(days=365)
                df = df[df.index >= one_year_ago]

            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    

def atr(high, low, close, window=14):
        tr = pd.concat([high.diff(), low.diff().abs(), (high - low).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

def aroon_up_down(high, low, window=14):
    aroon_up = 100 * (window - high.rolling(window=window).apply(np.argmax)) / window
    aroon_down = 100 * (window - low.rolling(window=window).apply(np.argmin)) / window
    return aroon_up, aroon_down

def rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def hull_moving_average(series, window=9):
    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    wma_half = series.rolling(window=half_length).mean()
    wma_full = series.rolling(window=window).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=sqrt_length).mean()
    return hma


def parabolic_sar(high, low, close, af=0.02, max_af=0.2):
    psar = close.copy()
    psar.fillna(0, inplace=True)
    bull = True
    af = af
    max_af = max_af
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(2, len(close)):
        psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
        if bull:
            if low[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = low[i]
                af = 0.02
                ep = low[i]
            if high[i] > ep:
                ep = high[i]
                af = min(af + 0.02, max_af)
            if low[i - 1] < psar[i]:
                psar[i] = low[i - 1]
            if low[i - 2] < psar[i]:
                psar[i] = low[i - 2]
        else:
            if high[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = high[i]
                af = 0.02
                ep = high[i]
            if low[i] < ep:
                ep = low[i]
                af = min(af + 0.02, max_af)
            if high[i - 1] > psar[i]:
                psar[i] = high[i - 1]
            if high[i - 2] > psar[i]:
                psar[i] = high[i - 2]
    return psar

def calculate_adx(df):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([df['high'] - df['low'], 
                    (df['high'] - df['close'].shift()).abs(), 
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
    return adx, plus_di, minus_di


def historical_volatility(df, window=252, return_full_df=False):
    df = df.copy()  # Avoid modifying original dataframe
    
    # Calculate daily returns
    df['Returns'] = df['close'].pct_change()
    
    # Rolling standard deviation of returns
    df['Hist_Vol'] = df['Returns'].rolling(window=window).std()
    
    # Annualized historical volatility
    df['Hist_Vol_Annualized'] = df['Hist_Vol'] * np.sqrt(window)

    return df if return_full_df else df['Hist_Vol_Annualized']   


def williams_fractal(df):
    def fractal_high(df, n):
        return df['high'][(df['high'] == df['high'].rolling(window=n, center=True).max()) &
                        (df['high'] > df['high'].shift(1)) &
                        (df['high'] > df['high'].shift(2)) &
                        (df['high'] > df['high'].shift(-1)) &
                        (df['high'] > df['high'].shift(-2))]

    def fractal_low(df, n):
        return df['low'][(df['low'] == df['low'].rolling(window=n, center=True).min()) &
                        (df['low'] < df['low'].shift(1)) &
                        (df['low'] < df['low'].shift(2)) &
                        (df['low'] < df['low'].shift(-1)) &
                        (df['low'] < df['low'].shift(-2))]

    n = 5  # Number of periods, typical value for Williams Fractal
    df['Fractal_Up'] = fractal_high(df, n)
    df['Fractal_Down'] = fractal_low(df, n)

    # Replace NaN with 0, indicating no fractal at these points
    df['Fractal_Up'] = df['Fractal_Up'].fillna(0)
    df['Fractal_Down'] = df['Fractal_Down'].fillna(0)

    return df[['Fractal_Up', 'Fractal_Down']]

# Chaikin Money Flow (CMF)
def cmf(high, low, close, volume, window=20):
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return cmf

# Money Flow Index (MFI)
def mfi(high, low, close, volume, window=14):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_mf = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_mf = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf_sum = positive_mf.rolling(window=window).sum()
    negative_mf_sum = negative_mf.rolling(window=window).sum()

    money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, 1)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi


# Calculate Chande Kroll Stop
def chande_kroll_stop(high, low, close, n=10, m=1):
    atr = (high - low).rolling(n).mean()
    long_stop = close - (m * atr)
    short_stop = close + (m * atr)
    return long_stop, short_stop

# Calculate Fisher Transform
def fisher_transform(price, n=10):
    median_price = price.rolling(window=n).median()
    min_low = price.rolling(window=n).min()
    max_high = price.rolling(window=n).max()
    value = 2 * ((median_price - min_low) / (max_high - min_low) - 0.5)
    fish = 0.5 * np.log((1 + value) / (1 - value))
    fish_signal = fish.shift(1)
    return fish, fish_signal



# Calculate Williams %R
def williams_r(high, low, close, n=14):
    highest_high = high.rolling(n).max()
    lowest_low = low.rolling(n).min()
    r = (highest_high - close) / (highest_high - lowest_low) * -100
    return r




def calculate_indicators(df):

    # Trend Indicators

    # Exponential Moving Average (EMA)
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ADX calculation
    #df['ADX'] = calculate_adx(df)
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df) 

    # SuperTrend
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3.0)
    df['SuperTrend'] = supertrend['SUPERT_7_3.0']

    # MACD
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Hull Moving Average (HMA)
    df['HMA'] = hull_moving_average(df['close'])

    # Ichimoku Cloud
    df['Ichimoku_Tenkan'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['Ichimoku_Kijun'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['Ichimoku_Senkou_Span_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
    df['Ichimoku_Senkou_Span_B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)

    # Aroon Indicator
    df['Aroon_Up'], df['Aroon_Down'] = aroon_up_down(df['high'], df['low'])

    # Guppy Multiple Moving Average (GMMA)
    df['GMMA_Short'] = df['close'].ewm(span=3, adjust=False).mean()
    df['GMMA_Long'] = df['close'].ewm(span=30, adjust=False).mean()

    # Keltner Channels
    #df['KC_Middle'] = df['close'].rolling(window=20).mean()
    #df['ATR_10'] = atr(df['high'], df['low'], df['close'], window=10)
    #df['KC_high'] = df['KC_Middle'] + (df['ATR_10'] * 2)
    #df['KC_low'] = df['KC_Middle'] - (df['ATR_10'] * 2)

    # Parabolic SAR
    df['Parabolic_SAR'] = parabolic_sar(df['high'], df['low'], df['close'])


    # Momentum Indicators

    df['RSI'] = rsi(df['close'])
    df['Momentum'] = df['close'] - df['close'].shift(10)

    df['ROC'] = df['close'].pct_change(12)
    df['Stochastic_%K'] = (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * 100
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    df['Stochastic_RSI'] = (rsi(df['close'], window=14) - rsi(df['close'], window=14).rolling(window=14).min()) / (rsi(df['close'], window=14).rolling(window=14).max() - rsi(df['close'], window=14).rolling(window=14).min())
    df['TRIX'] = df['close'].ewm(span=15, adjust=False).mean().pct_change(1)
    trix = ta.trix(df['close'])
    df['TRIX'] = trix['TRIX_30_9']
    df['TRIX_Signal'] = trix['TRIXs_30_9']
    
    
    df['TSI'] = df['close'].diff(1).ewm(span=25, adjust=False).mean() / df['close'].diff(1).abs().ewm(span=13, adjust=False).mean()
    df['TSI_Signal'] = df['TSI'].ewm(span=9, adjust=False).mean()
    df['CRSI'] = (rsi(df['close'], window=3) + rsi(df['close'], window=2) + rsi(df['close'], window=5)) / 3
    df['Fisher_Transform'], df['Fisher_Transform_Signal'] = fisher_transform(df['close'])

    #df['KST'] = df['close'].rolling(window=10).mean() + df['close'].rolling(window=15).mean() + df['close'].rolling(window=20).mean() + df['close'].rolling(window=30).mean()
    #df['KST_Signal'] = df['KST'].rolling(window=9).mean()

    # volume Indicators
    df['10_volume_MA'] = df['volume'].rolling(window=10).mean()
    df['30_volume_MA'] = df['volume'].rolling(window=30).mean()
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['AD'] = (df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['MFI'] = mfi(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['CMF'] = cmf(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['VWMA'] = ta.vwma(df['close'], df['volume'], length=20)


    #df['Net_volume'] = df['volume'] * (df['close'].diff() / df['close'].shift(1))
    #df['CO'] = df['close'].diff(3).ewm(span=10, adjust=False).mean()
    #df['EFI'] = df['close'].diff(1) * df['volume']
    #df['KVO'] = (df['high'] - df['low']).ewm(span=34, adjust=False).mean() - (df['high'] - df['low']).ewm(span=55, adjust=False).mean()
    #df['KVO_Signal'] = df['KVO'].ewm(span=13, adjust=False).mean()
    #df['PVT'] = (df['close'].pct_change(1) * df['volume']).cumsum()
    #df['Vortex_Pos'] = df['high'].diff(1).abs().rolling(window=14).sum() / atr(df['high'], df['low'], df['close'])
    #df['Vortex_Neg'] = df['low'].diff(1).abs().rolling(window=14).sum() / atr(df['high'], df['low'], df['close'])


    # Volatility Indicators
    df['ATR'] = atr(df['high'], df['low'], df['close'])
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_high'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_low'] = df['BB_Middle'] - (df['BB_Std'] * 2)

    df['BB_%B'] = (df['close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'])
    df['BB_Width'] = (df['BB_high'] - df['BB_low']) / df['close']
    df['Choppiness_Index'] = np.log10((df['high'] - df['low']).rolling(window=14).sum() / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100
    df['RVI'] = df['close'].rolling(window=10).mean() / df['close'].rolling(window=10).std() 
    df['Hist_Vol_Annualized'] = historical_volatility(df)
    df['Standard_Deviation'] = df['close'].rolling(window=20).std()



    #df['Chaikin_Volatility'] = (df['high'] - df['low']).ewm(span=10, adjust=False).mean()
    #df['Mass_Index'] = (df['high'] - df['low']).rolling(window=25).sum() / (df['high'] - df['low']).rolling(window=9).sum()
    #df['Chande_Kroll_Stop_Long'], df['Chande_Kroll_Stop_Short'] = chande_kroll_stop(df['high'], df['low'], df['close'])


    # Support and Resistance Indicators

    def pivot_points(high, low, close):
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        return pp, r1, s1, r2, s2, r3, s3

    df['Pivot_Point'], df['Resistance_1'], df['Support_1'], df['Resistance_2'], df['Support_2'], df['Resistance_3'], df['Support_3'] = pivot_points(df['high'], df['low'], df['close'])

    # Williams Fractal
    fractals = williams_fractal(df)
    df['Fractal_Up'] = fractals['Fractal_Up']
    df['Fractal_Down'] = fractals['Fractal_Down']
    
    
    # Fibonacci Levels
    max_price = df['high'].max()
    min_price = df['low'].min()
    diff = max_price - min_price
    df['Fibo_23_6'] = max_price - 0.236 * diff
    df['Fibo_38_2'] = max_price - 0.382 * diff
    df['Fibo_50'] = max_price - 0.5 * diff
    df['Fibo_61_8'] = max_price - 0.618 * diff
    
    # Donchian Channels
    donchian = ta.donchian(df['high'], df['low'])
    df['Donchian_High'] = donchian['DCU_20_20']
    df['Donchian_Low'] = donchian['DCL_20_20']
    
    # Darvas Box Theory
    df['Darvas_High'] = df['high'].rolling(window=20).max()
    df['Darvas_Low'] = df['low'].rolling(window=20).min()
    
    
    return df



def stock_analysis_app():

    # Streamlit UI
    st.title("Stock Analysis")
    ticker = st.sidebar.selectbox("Select Stock Ticker", tickers, index=tickers.index("^BSESN") if "^BSESN" in tickers else 0)

    if ticker:
        df = get_stock_data(ticker)

        if df.empty:
            print(f"No data for {ticker}")
        else:
            df.fillna(method="bfill", inplace=True)

            # Ensure lowercase columns for TA-Lib
            df.rename(columns=lambda x: x.lower(), inplace=True)
            #data = get_stock_data(ticker)
            df = calculate_indicators(df)

            df.index = pd.to_datetime(df.index)
            df = df.drop(columns = ['id','ticker','EMA_12', 'EMA_26'],axis=1)

            # ---------------------------
            # 3. Candlestick patterns
            # ---------------------------
            all_patterns = talib.get_function_groups()["Pattern Recognition"]

            pattern_weights = {
                "CDLMORNINGSTAR": 2, "CDLMORNINGDOJISTAR": 2, "CDL3WHITESOLDIERS": 2,
                "CDLENGULFING": 0,  # handled separately
                "CDLHAMMER": 2, "CDLINVERTEDHAMMER": 2, "CDLPIERCING": 2,
                "CDLMATHOLD": 2, "CDLLADDERBOTTOM": 2, "CDLABANDONEDBABY": 0,
                "CDLTAKURI": 2, "CDLUNIQUE3RIVER": 2, "CDLMATCHINGLOW": 2,
                "CDL3BLACKCROWS": -2, "CDLIDENTICAL3CROWS": -2, "CDLEVENINGSTAR": -2,
                "CDLEVENINGDOJISTAR": -2, "CDLDARKCLOUDCOVER": -2,
                "CDLSHOOTINGSTAR": -2, "CDLHANGINGMAN": -2, "CDL2CROWS": -2,
                "CDLUPSIDEGAP2CROWS": -2,
                "CDLBELTHOLD": 0.5
            }

            for p in all_patterns:
                func = getattr(talib, p)
                df[p] = func(df["open"], df["high"], df["low"], df["close"])

            # ---------------------------
            # 4. Weighted scoring
            # ---------------------------
            def get_day_score(row):
                score = 0
                for p in all_patterns:
                    val = row[p]
                    if val != 0:
                        if p == "CDLENGULFING":
                            score += 2 if val > 0 else -2
                        elif p in ["CDL3INSIDE", "CDL3OUTSIDE", "CDLCOUNTERATTACK", "CDLABANDONEDBABY", "CDLSEPARATINGLINES"]:
                            score += 1 if val > 0 else -1
                        else:
                            score += pattern_weights.get(p, 0)
                return score

            df["PatternScore"] = df.apply(get_day_score, axis=1)

            df["Signal"] = np.where(df["PatternScore"] > 0, "BUY",
                            np.where(df["PatternScore"] < 0, "SELL", "HOLD"))

            # ---------------------------
            # 5. Trend & volatility filters
            # ---------------------------
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
            df["Trend"] = np.where(df["close"] > df["ema50"], "Uptrend", "Downtrend")
            df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

            def apply_filters(row):
                if row["Signal"] == "BUY" and row["Trend"] != "Uptrend":
                    return "HOLD"
                if row["Signal"] == "SELL" and row["Trend"] != "Downtrend":
                    return "HOLD"
                if pd.isna(row["ATR"]) or row["close"] == 0:
                    return "HOLD"
                if row["ATR"] / row["close"] > 0.08:
                    return "HOLD"
                return row["Signal"]

            df["FilteredSignal"] = df.apply(apply_filters, axis=1)

            # ---------------------------
            # 6. Backtest filtered signals
            # ---------------------------
            initial_capital = 100000
            capital = initial_capital
            position = 0
            entry_price = 0
            trades = []

            for i in range(len(df)):
                signal = df["FilteredSignal"].iloc[i]
                close = df["close"].iloc[i]

                if position == 0 and signal == "BUY":
                    position = 1
                    entry_price = close
                    trades.append({"Entry": close, "Exit": None, "P&L": None})

                if position == 1:
                    if (close - entry_price) / entry_price <= -0.04:
                        position = 0
                        pnl = (close - entry_price) / entry_price * capital
                        capital += pnl
                        trades[-1]["Exit"] = close
                        trades[-1]["P&L"] = pnl
                    elif signal == "SELL":
                        position = 0
                        pnl = (close - entry_price) / entry_price * capital
                        capital += pnl
                        trades[-1]["Exit"] = close
                        trades[-1]["P&L"] = pnl

            trades_df = pd.DataFrame(trades)
            total_trades = len(trades_df)
            win_trades = trades_df[trades_df["P&L"] > 0].shape[0]
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            cumulative_pnl_percent = (capital - initial_capital) / initial_capital * 100



            # ---------------------------
            # 7. Plot filtered signals
            # ---------------------------
            df["BuySignal"] = np.where(df["FilteredSignal"] == "BUY", df["close"], np.nan)
            df["SellSignal"] = np.where(df["FilteredSignal"] == "SELL", df["close"], np.nan)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                        low=df["low"], close=df["close"], name="Candlesticks"))
            fig.add_trace(go.Scatter(x=df.index, y=df["BuySignal"], mode="markers",
                                    marker=dict(symbol="triangle-up", size=12, color="green"), name="Filtered BUY"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SellSignal"], mode="markers",
                                    marker=dict(symbol="triangle-down", size=12, color="red"), name="Filtered SELL"))
            fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], mode="lines", name="EMA50"))
            fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], mode="lines", name="EMA200"))

            fig.update_layout(
                title=f"{ticker} - Candlestick Pattern Signals",
                xaxis_title="Date", yaxis_title="Price",
                template="plotly", width=1600, height=700,
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=[
                            dict(count=7, label="7d", step="day", stepmode="backward"),
                            dict(count=14, label="14d", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all")
                        ]
                    )
                )
            )
            


            tab1, tab2, tab3 , tab4, tab5, tab6, tab7 = st.tabs(["Candlestick", "Momentum", "Trend","Volume","Volatility","Support & Resistance", "Forecast"])
            with tab1:
                st.subheader("Candlestick Chart with Signals")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Backtest Results")
                st.write(f"Total Trades: {total_trades}")
                st.write(f"Winning Trades: {win_trades}")
                st.write(f"Win Rate: {win_rate:.2f}%")
                st.write(f"Cumulative P&L %: {cumulative_pnl_percent:.2f}%")
                st.write(f"Final Amount: {capital:.2f}")

            with tab2:
                st.subheader("Momentum Indicators")
    
                # 1. Create 5-row subplot with shared x-axis
                fig = make_subplots(
                    rows=8, cols=1,
                    shared_xaxes=True,  # Required for unified hover
                    vertical_spacing=0.05,
                    row_heights=[1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,1.2],
                    subplot_titles=[
                        "RSI", 
                        "Stocastic RSI", 
                        "Momentum", 
                        "RoC",
                        "TRIX",
                        "TSI",
                        "CRSI", 
                        "Fisher Transform"

                    ]
                )


                # Plot RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',line={'color': 'blue', 'width': 2}), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), name='RSI 30', line={'color':'green', 'width': 2}), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[50] * len(df), name='RSI 30', line=dict(color='orange', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), name='RSI 70', line={'color':'red', 'width': 2}), row=1, col=1)

                # Plot Stochastic Oscillator
                fig.add_trace(go.Scatter(x=df.index, y=df['Stochastic_%K'], name='Stochastic_%K', line=dict(color='green')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Stochastic_%D'], name='Stochastic_%D', line=dict(color='red')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[20] * len(df), name='Stochastic 20', line=dict(color='blue', dash='dash')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[80] * len(df), name='Stochastic 80', line=dict(color='blue', dash='dash')), row=2, col=1)

                # Momentum
                fig.add_trace(go.Scatter(x=df.index, y=df['Momentum'], name='Momentum', line=dict(color='blue')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[0] * len(df), name='Momentum 0', line=dict(color='red', dash='dash')), row=3, col=1)

                # ROC
                fig.add_trace(go.Scatter(x=df.index, y=df['ROC'], name='ROC', line=dict(color='blue')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[0] * len(df), name='ROC 0', line=dict(color='red', dash='dash')), row=4, col=1)

                # TRIX
                fig.add_trace(go.Scatter(x=df.index, y=df['TRIX'], name='TRIX', line=dict(color='green')), row=5, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['TRIX_Signal'], name='TRIX_Signal', line=dict(color='red')), row=5, col=1)

                # TSI
                fig.add_trace(go.Scatter(x=df.index, y=df['TSI'], name='TSI', line=dict(color='green')), row=6, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['TSI_Signal'], name='TSI_Signal', line=dict(color='red')), row=6, col=1)

                # Plot CRSI
                fig.add_trace(go.Scatter(x=df.index, y=df['CRSI'], name='CRSI',line={'color': 'blue', 'width': 2}), row=7, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[20] * len(df), name='CRSI 20', line={'color':'green', 'width': 2}), row=7, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[80] * len(df), name='CRSI 80', line={'color':'red', 'width': 2}), row=7, col=1)

                # Fisher Transform
                fig.add_trace(go.Scatter(x=df.index, y=df['Fisher_Transform'], name='Fisher_Transform', line=dict(color='green')), row=8, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Fisher_Transform_Signal'], name='Fisher_Transform_Signal', line=dict(color='red')), row=8, col=1)

                # 5. Unified hover + Range selector

                fig.update_layout(
                    height=2000,
                    width=1500,
                    title=f'Momentum - {ticker}',
                    showlegend=False,

                    hovermode='x unified',

                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=12, label="12m", step="month", stepmode="backward"),
                                dict(count=24, label="24m", step="month", stepmode="backward"),
                                dict(count=36, label="36m", step="month", stepmode="backward"),
                                dict(count=48, label="48m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ],
                            x=0,
                            y=1.1,
                            xanchor="left",
                            yanchor="top"
                        ),
                        type='date',
                        rangeslider=dict(visible=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Trend Indicators")

                # 1. Create 5-row subplot with shared x-axis
                fig = make_subplots(
                    rows=9, cols=1,
                    shared_xaxes=True,  # Required for unified hover
                    vertical_spacing=0.05,
                    row_heights=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    subplot_titles=[
                        "MACD", 
                        "ADX", 
                        "Super Trend", 
                        "HMA", 
                        "Ichimoku",
                        "Aroon",
                        "Bollinger",
                        "PSAR",
                        "GMMA"
                        
                        
                        
                    ]
                )


                # Plot MACD
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='green')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', marker_color='rgba(255,0,0,2)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[0] * len(df), name='MACD 0', line={'color': 'black', 'width': 0.5}), row=1, col=1)

                # Plot DMI
                fig.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], name='Plus DI', line=dict(color='green')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], name='Minus DI', line=dict(color='red')), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='blue')), row=2, col=1)


                # Plot SuperTrend
                fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], name='SuperTrend', line=dict(color='red')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=3, col=1)


                # HMA
                fig.add_trace(go.Scatter(x=df.index, y=df['HMA'], name='HMA', line={'color': 'red', 'width': 2}), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=4, col=1)

                                
                # Plot Ichimoku Cloud
                fig.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_Senkou_Span_A'], name='Ichimoku A', fill='tonexty', fillcolor='rgba(0,128,0,0.3)'), row=5, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Ichimoku_Senkou_Span_B'], name='Ichimoku B', fill='tonexty', fillcolor='rgba(255,0,0,0.8)'), row=5, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='close',line={'color': 'blue', 'width': 2}), row=5, col=1)
                                
                # Aroon
                fig.add_trace(go.Scatter(x=df.index, y=df['Aroon_Up'], name='Aroon_Up', line=dict(color='green')), row=6, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Aroon_Down'], name='Aroon_Down', line=dict(color='red')), row=6, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), name='Aroon 30', line={'color':'green', 'width': 2}), row=6, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), name='Aroon 70', line={'color':'red', 'width': 2}), row=6, col=1)

                # Plot Bollinger Bands
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_high'], name='BB High', line=dict(color='red')), row=7, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_low'], name='BB Low', line=dict(color='green')), row=7, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle', line={'dash': 'dot'}), row=7, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=7, col=1)

                # Plot Parabolic SAR
                fig.add_trace(go.Scatter(x=df.index, y=df['Parabolic_SAR'], mode='markers', name='PSAR', marker=dict(color='red', size=3)), row=8, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='close', line={'color': 'blue', 'width': 2}), row=8, col=1)

                # Plot GMMA
                fig.add_trace(go.Scatter(x=df.index, y=df['GMMA_Short'], name='GMMA_Short', line=dict(color='green')), row=9, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['GMMA_Long'], name='GMMA_Long', line=dict(color='red')), row=9, col=1)
                                
                                

                # 5. Unified hover + Range selector
                fig.update_layout(
                    height=2000,
                    width=1500,
                    title=f'Trend - {ticker}',
                    showlegend=False,

                    hovermode='x unified',

                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=12, label="12m", step="month", stepmode="backward"),
                                dict(count=24, label="24m", step="month", stepmode="backward"),
                                dict(count=36, label="36m", step="month", stepmode="backward"),
                                dict(count=48, label="48m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ],
                            x=0,
                            y=1.1,
                            xanchor="left",
                            yanchor="top"
                        ),
                        type='date',
                        rangeslider=dict(visible=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)                         


            with tab4:
                st.subheader("Volume Indicators")

                # 1. Create 5-row subplot with shared x-axis
                fig = make_subplots(
                    rows=6, cols=1,
                    shared_xaxes=True,  # Required for unified hover
                    vertical_spacing=0.05,
                    row_heights=[1.5, 1.5, 1.5, 1.5, 1.5,1.5],
                    subplot_titles=[
                        "OBV (On-Balance Volume)", 
                        "A/D (Accumulation/Distribution)",
                        "MFI (Money Flow Index)", 
                        "CMF (Chaikin Money Flow)",       
                        "Price with VWAP", 
                        "Price with VWMA"
                    ]
                )


                # Row 1: OBV
                fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', line=dict(color='purple')), row=1, col=1)

                # Row 2: A/D
                fig.add_trace(go.Scatter(x=df.index, y=df['AD'], name='Accumulation/Distribution', line=dict(color='orange')), row=2, col=1)

                # Row 3: MFI with reference lines
                fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], name='MFI', line=dict(color='blue')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[20] * len(df), name='MFI 20', line={'color': 'green', 'width': 2, 'dash': 'dash'}), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[80] * len(df), name='MFI 80', line={'color': 'red', 'width': 2, 'dash': 'dash'}), row=3, col=1)

                # Row 4: CMF with reference line at 0
                fig.add_trace(go.Scatter(x=df.index, y=df['CMF'], name='CMF', line={'color': 'blue', 'width': 2}), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[0] * len(df), name='CMF 0', line={'color': 'red', 'width': 2, 'dash': 'dash'}), row=4, col=1)

                # Row 5: VWAP and Close
                fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line={'color': 'red', 'width': 2}), row=5, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=5, col=1)

                # Row 6: VWMA and Close
                fig.add_trace(go.Scatter(x=df.index, y=df['VWMA'], name='VWMA', line={'color': 'red', 'width': 2}), row=6, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=6, col=1)

                # 5. Unified hover + Range selector
                fig.update_layout(
                    height=2000,
                    width=1500,
                    title=f'Volume Analysis - {ticker}',
                    showlegend=False,

                    hovermode='x unified',

                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=12, label="12m", step="month", stepmode="backward"),
                                dict(count=24, label="24m", step="month", stepmode="backward"),
                                dict(count=36, label="36m", step="month", stepmode="backward"),
                                dict(count=48, label="48m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ],
                            x=0,
                            y=1.1,
                            xanchor="left",
                            yanchor="top"
                        ),
                        type='date',
                        rangeslider=dict(visible=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)      


            with tab5:
                st.subheader("Volatility Indicators")

                # 1. Create 5-row subplot with shared x-axis
                fig = make_subplots(
                    rows=7, cols=1,
                    shared_xaxes=True,  # Required for unified hover
                    vertical_spacing=0.05,
                    row_heights=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                    subplot_titles=[
                        "ATR",
                        "BB_Width",
                        "BB%", 
                        "Choppiness_Index", 
                        "RVI", 
                        "Hist_Vol_Annualized",
                        "Standard_Deviation"     
                        
                    ]
                )

                # Plot ATR
                fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR', line=dict(color='red')), row=1, col=1)

                # Plot BB_Width
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Width'], name='BB_Width', line=dict(color='blue')), row=2, col=1)

                # Plot BB%
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_%B'], name='BB%', line=dict(color='blue')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[1] * len(df), name='BB 1', line=dict(color='red', dash='dash')), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[0] * len(df), name='BB 0', line=dict(color='red', dash='dash')), row=3, col=1)


                # Plot Choppiness_Index
                fig.add_trace(go.Scatter(x=df.index, y=df['Choppiness_Index'], name='Choppiness_Index',line={'color': 'blue', 'width': 2}), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[38] * len(df), name='Choppiness_Index 38', line={'color':'green', 'width': 2}), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[61] * len(df), name='Choppiness_Index 61', line={'color':'red', 'width': 2}), row=4, col=1)

                # Plot RVI
                fig.add_trace(go.Scatter(x=df.index, y=df['RVI'], name='RVI', line=dict(color='blue')), row=5, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=[50] * len(df), name='RVI 50', line=dict(color='red', dash='dash')), row=5, col=1)


                # Hist_Vol_Annualized
                fig.add_trace(go.Scatter(x=df.index, y=df['Hist_Vol_Annualized'], name='Hist_Vol_Annualized', line=dict(color='red')), row=6, col=1)

                # Plot Standard_Deviation
                fig.add_trace(go.Scatter(x=df.index, y=df['Standard_Deviation'], name='Standard Deviation', line=dict(color='red')), row=7, col=1)



                # 5. Unified hover + Range selector
                fig.update_layout(
                    height=2000,
                    width=1500,
                    title=f'Volatality - {ticker}',
                    showlegend=False,

                    hovermode='x unified',

                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=12, label="12m", step="month", stepmode="backward"),
                                dict(count=24, label="24m", step="month", stepmode="backward"),
                                dict(count=36, label="36m", step="month", stepmode="backward"),
                                dict(count=48, label="48m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ],
                            x=0,
                            y=1.1,
                            xanchor="left",
                            yanchor="top"
                        ),
                        type='date',
                        rangeslider=dict(visible=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)    

            with tab6:
                st.subheader("Support & Resistance Indicators")

                # 1. Create 5-row subplot with shared x-axis
                fig = make_subplots(
                    rows=5, cols=1,
                    shared_xaxes=True,  # Required for unified hover
                    vertical_spacing=0.05,
                    row_heights=[1.5, 1.5, 1.5, 1.5, 1.5],
                    subplot_titles=[
                        "Donchian Channels",
                        "Pivot Points",
                        "Fibonacci Levels", 
                        "Darvas Box", 
                        "Williams Fractal"   
                        
                    ]
                )

                # Donchian Channels
                fig.add_trace(go.Scatter(x=df.index, y=df['Donchian_High'], name='Donchian_High', line=dict(color='green')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Donchian_Low'], name='Donchian_Low', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=1, col=1)



                # Plot Pivot Points
                fig.add_trace(go.Scatter(x=df.index, y=df['Pivot_Point'], name='Pivot'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_1'], name='R1', line={'dash': 'dot'}), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Support_1'], name='S1', line={'dash': 'dot'}), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_2'], name='R2', line={'dash': 'dot'}), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Support_2'], name='S2', line={'dash': 'dot'}), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_3'], name='R3', line={'dash': 'dot'}), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Support_3'], name='S3', line={'dash': 'dot'}), row=2, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=2, col=1)

                # Plot Fibonacci Levels
                fig.add_trace(go.Scatter(x=df.index, y=df['Fibo_23_6'], name='Fibo 23.6%', line={'dash': 'dot'}), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Fibo_38_2'], name='Fibo 38.2%', line={'dash': 'dot'}), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Fibo_50'], name='Fibo 50%', line={'dash': 'dot'}), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Fibo_61_8'], name='Fibo 61.8%', line={'dash': 'dot'}), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=3, col=1)

                # Darvas Box
                fig.add_trace(go.Scatter(x=df.index, y=df['Darvas_High'], name='Darvas_High', line=dict(color='green')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Darvas_Low'], name='Darvas_Low', line=dict(color='red')), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line={'color': 'blue', 'width': 2}), row=4, col=1)

                # Williams Fractal
                fig.add_trace(go.Scatter(x=df.index, y=df['Fractal_Up'], name='Fractal_Up', line=dict(color='green')), row=5, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Fractal_Down'], name='Fractal_Down', line=dict(color='red')), row=5, col=1)


                # 5. Unified hover + Range selector
                fig.update_layout(
                    height=2000,
                    width=1500,
                    title=f'Support & Resistance - {ticker}',
                    showlegend=False,

                    hovermode='x unified',

                    xaxis=dict(
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=12, label="12m", step="month", stepmode="backward"),
                                dict(count=24, label="24m", step="month", stepmode="backward"),
                                dict(count=36, label="36m", step="month", stepmode="backward"),
                                dict(count=48, label="48m", step="month", stepmode="backward"),
                                dict(step="all", label="All")
                            ],
                            x=0,
                            y=1.1,
                            xanchor="left",
                            yanchor="top"
                        ),
                        type='date',
                        rangeslider=dict(visible=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)  

            with tab7:
                st.subheader("Forecasting with ML")


