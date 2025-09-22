import streamlit as st
import streamlit as st
from components.my_portfolio import get_user_by_username
from supabase import create_client
import pandas as pd
from utils.constants import ticker_to_company_dict,rsi,calculate_adx,calculate_bollinger_bands,SUPABASE_URL,SUPABASE_KEY

import plotly.graph_objects as go
import plotly.subplots as sp
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta

# Supabase client setup 
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Helper function to calculate indicators
def calculate_indicators(data):
    try:
        data['5_day_EMA'] = data['close'].ewm(span=5, adjust=False).mean()
        data['15_day_EMA'] = data['close'].ewm(span=15, adjust=False).mean()
        
        # MACD calculations
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # Manual RSI calculation
        data['RSI'] = rsi(data['close'])

        # Manual ADX calculation
        data['ADX'], data['+DI'], data['-DI'] = calculate_adx(data)

        # Manual Bollinger Bands calculation
        data['Bollinger_High'], data['Bollinger_Low'] = calculate_bollinger_bands(data)

        # Volume MA calculation
        data['20_day_vol_MA'] = data['volume'].rolling(window=20).mean()
        
        return data
    except Exception as e:
        raise ValueError(f"Error calculating indicators: {str(e)}")


# Helper function to fetch ticker data
def fetch_ticker_data(ticker):
    try:
        response = (
            supabase.table("stock_data")
            .select("*")
            .filter("ticker", "eq", ticker)
            .order("date", desc=True)  # Order by latest date
            .execute()
        )
        if response.data:
            df = pd.DataFrame(response.data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.drop_duplicates(subset=['date'], keep='first', inplace=True)
                df.set_index('date', inplace=True)
                df = df.sort_index()

                six_months_ago = datetime.today() - timedelta(days=180)
                df = df[df.index >= six_months_ago]

            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# Helper function to fetch company info 
def get_company_info(ticker):
    try:
        # Fetch data from the Supabase database
        response = supabase.table("stock_info").select("*").execute()

        # Check if data exists and convert it to a DataFrame
        if response.data:
            df = pd.DataFrame(response.data)
            
            # Filter the data based on the ticker symbol
            company_data = df[df['ticker'] == ticker]
            
            if not company_data.empty:
                long_name = company_data['longname'].iloc[0]  # Get the long name
                sector = company_data['sector'].iloc[0]  # Get the sector
                industry = company_data['industry'].iloc[0]  # Get the industry
                return long_name, sector, industry
            else:
                st.error(f"No data found for ticker '{ticker}' in the database.")
                return None, None, None
        else:
            st.error(f"No data found in the database.")
            return None, None, None

    except Exception as e:
        st.error(f"Error fetching company info for ticker '{ticker}': {e}")
        return None, None, None


# Convert the ticker_to_company_dict dictionary to a list of company names
company_names = list(ticker_to_company_dict.values())





# Watchlist feature
def display_watchlist():
    st.header(f"{st.session_state.username}'s Watchlist")
    user = get_user_by_username(st.session_state.username)
    user_id = user['id'] if user else None
    if user_id is None:
        st.warning("No user is currently logged in. Please log in to view your watchlist.")
        return
    # Fetch watchlist from Supabase
    response = supabase.table("watchlist").select("*").filter("user_id", "eq", user_id).execute()
    watchlist = getattr(response, 'data', [])

    st.sidebar.header("Watchlist Management")

    # selectbox for company name auto-suggestion
    selected_company = st.sidebar.selectbox('Select the Stock:', company_names)

    # Retrieve the corresponding ticker for the selected company
    ticker = [ticker for ticker, company in ticker_to_company_dict.items() if company == selected_company][0]

    # Add new ticker to watchlist
    if st.sidebar.button("Add Ticker"):
        try:
            fetch_ticker_data(ticker)
            # Check if ticker already exists in watchlist
            check_response = supabase.table("watchlist").select("id").filter("user_id", "eq", user_id).filter("ticker", "eq", ticker).execute()
            exists = bool(getattr(check_response, 'data', []))
            if not exists:
                insert_response = supabase.table("watchlist").insert({"user_id": user_id, "ticker": ticker}).execute()
                error = getattr(insert_response, 'error', None)
                if error is None:
                    st.success(f"{ticker} ({selected_company}) added to your watchlist!")
                    st.rerun()
                else:
                    st.error(f"Error adding to watchlist: {error}")
            else:
                st.warning(f"{ticker} ({selected_company}) is already in your watchlist.")
        except ValueError as ve:
            st.error(ve)

    # Display watchlist
    if watchlist:
        watchlist_data = {}
        ticker_to_name_map = {}
        for entry in watchlist:
            ticker = entry['ticker']
            try:
                data = fetch_ticker_data(ticker)
                data = calculate_indicators(data)
                latest_data = data.iloc[-1]
                company_name, sector, industry = get_company_info(ticker)
                watchlist_data[ticker] = {
                    'Company Name': company_name,
                    'Sector': sector,
                    'Industry': industry,
                    'Close': latest_data['close'],
                    '5_day_EMA': latest_data['5_day_EMA'],
                    '15_day_EMA': latest_data['15_day_EMA'],
                    'MACD': latest_data['MACD'],
                    'MACD_Signal': latest_data['MACD_Signal'],
                    'MACD_Hist': latest_data['MACD_Hist'],
                    'RSI': latest_data['RSI'],
                    'ADX': latest_data['ADX'],
                    'Bollinger_High': latest_data['Bollinger_High'],
                    'Bollinger_Low': latest_data['Bollinger_Low'],
                    'Volume': latest_data['volume'],
                    '20_day_vol_MA': latest_data['20_day_vol_MA']
                }
                ticker_to_name_map[ticker] = company_name
            except ValueError as ve:
                st.error(f"Error fetching data for {ticker}: {ve}")

        if watchlist_data:
            watchlist_df = pd.DataFrame.from_dict(watchlist_data, orient='index')
            watchlist_df.reset_index(inplace=True)
            watchlist_df.rename(columns={'index': 'Ticker'}, inplace=True)

            # Use Styler to format the DataFrame
            styled_df = watchlist_df.style.format(precision=2)

            st.write("Your Watchlist:")
            st.dataframe(styled_df.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))
            
            

            # Selectionbox for Inputs
            col1, col2 = st.columns(2)


            with col1:
                st.subheader("Technical Indicators Vs Price")
            with col2:
                stock_symbol = st.selectbox("Select Stock", watchlist_df['Ticker'].tolist())


            # Step 1: Download Stock Data
            data = fetch_ticker_data(stock_symbol)

            # Check if data is available
            if data.empty:
                st.warning("No data available for the given ticker and date range. Please try again.")
            else:
                # Step 2: Calculate Technical Indicators
                # VWAP (Volume Weighted Average Price)
                data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

                # MFI (Money Flow Index)
                data['MFI'] = ta.mfi(data['high'], data['low'], data['close'], data['volume'], length=14)

                # OBV (On-Balance Volume)
                data['OBV'] = ta.obv(data['close'], data['volume'])

                # CMF (Chaikin Money Flow)
                data['CMF'] = ta.cmf(data['high'], data['low'], data['close'], data['volume'], length=20)

                # A/D (Accumulation/Distribution)
                data['AD'] = ta.ad(data['high'], data['low'], data['close'], data['volume'])

                # Ichimoku Cloud
                data['Ichimoku_Tenkan'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
                data['Ichimoku_Kijun'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
                data['Ichimoku_Senkou_Span_A'] = ((data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2).shift(26)
                data['Ichimoku_Senkou_Span_B'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

                # MACD (Moving Average Convergence Divergence)
                data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_hist'] = data['MACD'] - data['MACD_signal']

                # SuperTrend
                supertrend = ta.supertrend(data['high'], data['low'], data['close'], length=7, multiplier=3.0)
                data['SuperTrend'] = supertrend['SUPERT_7_3.0']

                # Bollinger Bands
                data['BB_Middle'] = data['close'].rolling(window=20).mean()
                data['BB_Std'] = data['close'].rolling(window=20).std()
                data['BB_High'] = data['BB_Middle'] + (data['BB_Std'] * 2)
                data['BB_Low'] = data['BB_Middle'] - (data['BB_Std'] * 2)

                # Parabolic SAR
                def parabolic_sar(high, low, close, af=0.02, max_af=0.2):
                    psar = close.copy()
                    psar.fillna(0, inplace=True)
                    bull = True
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

                data['PSAR'] = parabolic_sar(data['high'], data['low'], data['close'])

                # GMMA (Guppy Multiple Moving Average)
                short_ema = ta.ema(data['close'], length=3)
                long_ema = ta.ema(data['close'], length=30)

                # RSI (Relative Strength Index)
                data['RSI'] = ta.rsi(data['close'], length=14)

                # Stochastic Oscillator
                data['Stochastic_%K'] = (data['close'] - data['low'].rolling(window=14).min()) / (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min()) * 100
                data['Stochastic_%D'] = data['Stochastic_%K'].rolling(window=3).mean()

                # DMI (Directional Movement Index)
                def calculate_adx(data):
                    # Calculate the Directional Movements
                    plus_dm = data['high'].diff()
                    minus_dm = data['low'].diff()
                    plus_dm[plus_dm < 0] = 0
                    minus_dm[minus_dm > 0] = 0
                    
                    # Calculate the True Range (TR) and Average True Range (ATR)
                    tr = pd.concat([data['high'] - data['low'], 
                                    (data['high'] - data['close'].shift()).abs(), 
                                    (data['low'] - data['close'].shift()).abs()], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    
                    # Calculate the Positive Directional Indicator (Plus DI) and Negative Directional Indicator (Minus DI)
                    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
                    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
                    
                    # Calculate the ADX
                    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
                    
                    return adx, plus_di, minus_di

                # Apply ADX to your data
                data['ADX'], data['Plus_DI'], data['Minus_DI'] = calculate_adx(data)

                # Awesome Oscillator
                data['AO'] = ta.ao(data['high'], data['low'])

                # Bollinger Bands %B
                data['BB%'] =  (data['close'] - data['BB_Low']) / (data['BB_High'] - data['BB_Low'])

                # Mass Index
                data['Mass_Index'] = (data['high'] - data['low']).rolling(window=25).sum() / (data['high'] - data['low']).rolling(window=9).sum()

                # Relative Volatility Index (RVI)
                data['RVI'] = ta.rvi(data['high'], data['low'], data['close'], length=14)

                # ZigZag
                def zigzag(close, percentage=5):
                    zz = [0]
                    for i in range(1, len(close)):
                        change = (close[i] - close[zz[-1]]) / close[zz[-1]] * 100
                        if abs(change) > percentage:
                            zz.append(i)
                    zigzag_series = pd.Series(index=close.index, data=np.nan)
                    zigzag_series.iloc[zz] = close.iloc[zz]
                    return zigzag_series.ffill()

                data['ZigZag'] = zigzag(data['close'])

                # Pivot Points Standard
                data['Pivot'] = (data['high'] + data['low'] + data['close']) / 3
                data['R1'] = 2 * data['Pivot'] - data['low']
                data['S1'] = 2 * data['Pivot'] - data['high']

                # Fibonacci Levels
                max_price = data['high'].max()
                min_price = data['low'].min()
                diff = max_price - min_price
                data['Fibo_23_6'] = max_price - 0.236 * diff
                data['Fibo_38_2'] = max_price - 0.382 * diff
                data['Fibo_50'] = max_price - 0.5 * diff
                data['Fibo_61_8'] = max_price - 0.618 * diff

                # Calculate volume moving averages
                data['Volume_MA10'] = data['volume'].rolling(window=10).mean()
                data['Volume_MA30'] = data['volume'].rolling(window=30).mean()

                # Moving Averages
                data['MA50'] = data['close'].rolling(window=50).mean()
                data['MA200'] = data['close'].rolling(window=200).mean()

                # Identify volume spikes (days where volume is 50% higher than 10-day MA)
                volume_spikes = data[data['volume'] > data['Volume_MA30'] * 1.5].index

                # Step 3: Create Subplots for all indicators
           

                fig = sp.make_subplots(rows=8, cols=3, subplot_titles=[
                    'VWAP', 'MFI', 'OBV', 'CMF', 
                    'A/D', 'Ichimoku Cloud', 'MACD', 'SuperTrend', 
                    'Bollinger Bands', 'Parabolic SAR', 'GMMA', 'RSI', 
                    'Stochastic Oscillator', 'DMI', 'Awesome Oscillator', 'BB%',
                    'Mass Index', 'RVI', 'ZigZag', 'Pivot Points', 
                    'Fibonacci Levels','Volume','Moving Averages'],
                    vertical_spacing=0.05, horizontal_spacing=0.05)

                # Plot VWAP and Close
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP',line={'color': 'red', 'width': 2}), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='close',line={'color': 'blue', 'width': 2}), row=1, col=1)


                # Plot MFI with red lines at 20 and 80
                fig.add_trace(go.Scatter(x=data.index, y=data['MFI'], name='MFI', line=dict(color='blue')), row=1, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=[20] * len(data), name='MFI 20', line={'color':'green', 'width': 2}), row=1, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=[80] * len(data), name='MFI 80', line={'color':'red', 'width': 2}), row=1, col=2)


                # Plot OBV
                fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], name='OBV'), row=1, col=3)

                # Plot CMF
                fig.add_trace(go.Scatter(x=data.index, y=data['CMF'], name='CMF',line={'color': 'blue', 'width': 2}), row=2, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='CMF 0', line={'color':'red', 'width': 2}), row=2, col=1)

                # Plot A/D
                fig.add_trace(go.Scatter(x=data.index, y=data['AD'], name='A/D'), row=2, col=2)

                # Plot Ichimoku Cloud
                fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Senkou_Span_A'], name='Ichimoku A', fill='tonexty', fillcolor='rgba(0,128,0,0.3)'), row=2, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Senkou_Span_B'], name='Ichimoku B', fill='tonexty', fillcolor='rgba(255,0,0,0.8)'), row=2, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='close',line={'color': 'blue', 'width': 2}), row=2, col=3)



                # Plot MACD
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='green')), row=3, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal', line=dict(color='red')), row=3, col=1)
                fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], name='MACD Histogram', marker_color='rgba(255,0,0,2)'), row=3, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='MACD 0', line={'color': 'black', 'width': 0.5}), row=3, col=1)

                # Plot SuperTrend
                fig.add_trace(go.Scatter(x=data.index, y=data['SuperTrend'], name='SuperTrend', line=dict(color='red')), row=3, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='close', line={'color': 'blue', 'width': 2}), row=3, col=2)

                # Plot Bollinger Bands
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='BB High', line=dict(color='red')), row=3, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name='BB Low', line=dict(color='green')), row=3, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], name='BB Middle', line={'dash': 'dot'}), row=3, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='close', line={'color': 'blue', 'width': 2}), row=3, col=3)

                # Plot Parabolic SAR
                fig.add_trace(go.Scatter(x=data.index, y=data['PSAR'], mode='markers', name='PSAR', marker=dict(color='red', size=3)), row=4, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='close', line={'color': 'blue', 'width': 2}), row=4, col=1)

                # Plot GMMA
                fig.add_trace(go.Scatter(x=data.index, y=short_ema, name='Short EMA', line=dict(color='green')), row=4, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=long_ema, name='Long EMA', line=dict(color='red')), row=4, col=2)

                # Plot RSI
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',line={'color': 'blue', 'width': 2}), row=4, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[30] * len(data), name='RSI 30', line={'color':'green', 'width': 2}), row=4, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[70] * len(data), name='RSI 70', line={'color':'red', 'width': 2}), row=4, col=3)


                # Plot Stochastic Oscillator
                fig.add_trace(go.Scatter(x=data.index, y=data['Stochastic_%K'], name='Stochastic_%K', line=dict(color='green')), row=5, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Stochastic_%D'], name='Stochastic_%D', line=dict(color='red')), row=5, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[20] * len(data), name='Stochastic 20', line=dict(color='blue', dash='dash')), row=5, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[80] * len(data), name='Stochastic 80', line=dict(color='blue', dash='dash')), row=5, col=1)

                # Plot DMI
                fig.add_trace(go.Scatter(x=data.index, y=data['Plus_DI'], name='Plus DI', line=dict(color='green')), row=5, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['Minus_DI'], name='Minus DI', line=dict(color='red')), row=5, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX', line=dict(color='blue')), row=5, col=2)

                # Plot Awesome Oscillator
                fig.add_trace(go.Scatter(x=data.index, y=data['AO'], name='Awesome Oscillator', line=dict(color='blue')), row=5, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='AO 0', line=dict(color='red', dash='dash')), row=5, col=3)

                # Plot BB%
                fig.add_trace(go.Scatter(x=data.index, y=data['BB%'], name='BB%', line=dict(color='blue')), row=6, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[1] * len(data), name='BB 1', line=dict(color='red', dash='dash')), row=6, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='BB 0', line=dict(color='red', dash='dash')), row=6, col=1)

                # Plot Mass Index
                fig.add_trace(go.Scatter(x=data.index, y=data['Mass_Index'], name='Mass Index'), row=6, col=2)

                # Plot RVI
                fig.add_trace(go.Scatter(x=data.index, y=data['RVI'], name='RVI', line=dict(color='blue')), row=6, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[50] * len(data), name='RVI 50', line=dict(color='red', dash='dash')), row=6, col=3)

                # Plot ZigZag
                fig.add_trace(go.Scatter(x=data.index, y=data['ZigZag'], name='ZigZag'), row=7, col=1)

                # Plot Pivot Points
                fig.add_trace(go.Scatter(x=data.index, y=data['Pivot'], name='Pivot'), row=7, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['R1'], name='R1', line={'dash': 'dot'}), row=7, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['S1'], name='S1', line={'dash': 'dot'}), row=7, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close', line={'color': 'blue', 'width': 2}), row=7, col=2)

                # Plot Fibonacci Levels
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_23_6'], name='Fibo 23.6%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_38_2'], name='Fibo 38.2%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_50'], name='Fibo 50%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_61_8'], name='Fibo 61.8%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close', line={'color': 'blue', 'width': 2}), row=7, col=3)


                # Plot Volume
                fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='blue'), row=8, col=1)
                fig.add_trace(go.Scatter(x=volume_spikes, y=data.loc[volume_spikes, 'volume'], mode='markers', name='Volume_Spike', marker=dict(color='red', size=8)), row=8, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Volume_MA10'], name='Vol MA10', line={'color': 'green', 'width': 2}), row=8, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Volume_MA30'], name='Vol MA30', line={'color': 'red', 'width': 2}), row=8, col=1)

                # Plot Moving Averages
                fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line={'color': 'green', 'width': 2}), row=8, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='MA200', line={'color': 'red', 'width': 2}), row=8, col=2)

                # Layout
                fig.update_layout(
                    height=3000,
                    width=2000,
                    title={
                        'text': f"Technical Indicators and Price Analysis of {stock_symbol}",
                        'x': 0.5,   # Center horizontally
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    showlegend=False
                )

                # Display the plot
                st.plotly_chart(fig)

            # Option to remove ticker from watchlist
            company_names_in_watchlist = [ticker_to_name_map[entry['ticker']] for entry in watchlist]
            company_to_remove = st.sidebar.selectbox("Select a company to remove", company_names_in_watchlist)
            ticker_to_remove = [ticker for ticker, name in ticker_to_name_map.items() if name == company_to_remove][0]
            if st.sidebar.button("Remove Ticker"):
                    delete_response = supabase.table("watchlist").delete().filter("user_id", "eq", user_id).filter("ticker", "eq", ticker_to_remove).execute()
                    error = getattr(delete_response, 'error', None)
                    if error is None:
                        st.success(f"{company_to_remove} removed from your watchlist.")
                        st.rerun()
                    else:
                        st.error(f"Error removing from watchlist: {error}")
        else:
            st.write("No valid data found for the tickers in your watchlist.")
    else:
        st.write("Your watchlist is empty.")




# Call the function to display the watchlist
if 'username' not in st.session_state:
    st.session_state.username = 'Guest'  # or handle the case where username is not set
if 'email' not in st.session_state:
    st.session_state.email = 'guest@example.com'  # or handle the case where email is not set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False  # or handle the case where logged_in is not set

if st.session_state.logged_in:
    display_watchlist()
else:
    st.write("Please log in to view your watchlist.")
