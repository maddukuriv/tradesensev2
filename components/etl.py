import streamlit as st

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime, timedelta
from utils.constants import SUPABASE_URL, SUPABASE_KEY, Largecap, Midcap, Smallcap, Indices, crypto_largecap, crypto_midcap


def etl_app():

    st.title("ETL Page")
    
    # ------------------------------
    # Sidebar: Dropdowns
    # ------------------------------
    category = st.sidebar.selectbox(
        "Select Ticker Category",
        ["Largecap", "Midcap", "Smallcap", "Indices", "Crypto Largecap", "Crypto Midcap"]
    )

    strategy = st.sidebar.selectbox(
        "Select Action",
        ["Insert Data","Remove Data"]
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