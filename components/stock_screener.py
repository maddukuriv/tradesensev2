

import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime, timedelta
from utils.constants import SUPABASE_URL, SUPABASE_KEY, Largecap, Midcap, Smallcap, Indices, crypto_largecap, crypto_midcap
import talib


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
        ["Candlestick Patterns","Momentum", "Mean Reversion", "Trend Following", "Volume Driven", 
        "Breakout", "Volatility Reversion", "Swing Trading", "Arbitrage", 
        "News-Based"]
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
                    st.success(f"Found {len(result)} stocks with momentum trend")
                    st.dataframe(filtered.sort_values(["Date","Score"], ascending=False))

                else:
                    st.info("No stocks met the momentum criteria in the last 5 days.")


        else:
            st.warning("No tickers available for the selected category.")


    elif strategy == "Mean Reversion":
        st.write("📉 Mean reversion assumes that stock prices will revert to their historical average over time.")
    elif strategy == "Trend Following":                 
        st.write("📊 Trend following involves buying stocks in an uptrend and selling those in a downtrend.")
    elif strategy == "Volume Driven":
        st.write("💹 Volume driven strategy focuses on stocks with high trading volumes, showing strong investor interest.")                
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




