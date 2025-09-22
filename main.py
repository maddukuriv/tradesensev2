import streamlit as st
from dotenv import load_dotenv

from authentication import login, signup, forgot_password
from components import my_account, my_portfolio, my_watchlist, markets, stock_screener, stock_analysis, admin, home_page, etl



# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="TradeSense", page_icon="📈", initial_sidebar_state="expanded")

# Load environment variables from .env file
load_dotenv()

# Initialize session state for login status and user info
for key, default in {
    'logged_in': False,
    'username': "",
    'email': "",
    'user_id': None,
    'identity_verified': False,
    'is_admin': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Logout function
def logout():
    for key in ['logged_in', 'username', 'email', 'user_id', 'identity_verified', 'is_admin']:
        st.session_state[key] = False if key == 'logged_in' else "" if key in ['username', 'email'] else None
    st.rerun()

# Main menu function
def main_menu():
    menu_options = [
        f"{st.session_state.username}'s Portfolio",
        f"{st.session_state.username}'s Watchlist",
        "Markets",
        "Screener",
        "Analysis",
        "My Account"
    ]
    # Add admin pages if user is admin
    if st.session_state.is_admin:
        menu_options += ["Database Admin Page", "ETL Page"]
    return st.selectbox("Select an option", menu_options)

# Sidebar menu
with st.sidebar:
    st.title("TradeSense")
    if st.session_state.logged_in:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            logout()
        choice = main_menu()
    else:
        selected = st.selectbox("Choose an option", ["Login", "Sign Up", "Forgot Password"])
        if selected == "Login":
            login()
        elif selected == "Sign Up":
            signup()
        elif selected == "Forgot Password":
            forgot_password()
        choice = None

######################################################### Main content area ######################################################################

if not st.session_state.logged_in:
    home_page.home_page_app()
else:
    if choice:
        if choice == "My Account":
            my_account.my_account()
        elif choice == f"{st.session_state.username}'s Watchlist":
            my_watchlist.display_watchlist()
        elif choice == f"{st.session_state.username}'s Portfolio":
            my_portfolio.display_portfolio()
        elif choice == "Markets":
            markets.markets_app()

        elif choice == "Screener":
            stock_screener.stock_screener_app()
        elif choice == "Analysis":
            stock_analysis.stock_analysis_app()
        elif choice == "Database Admin Page" and st.session_state.is_admin:
            admin.display_tables()
        elif choice == "ETL Page" and st.session_state.is_admin:
            etl.etl_app()
        else:
            st.error("You don't have permission to access this page.")
