import streamlit as st
import pandas as pd
from supabase import create_client, Client
from utils.constants import SUPABASE_URL, SUPABASE_KEY
import os

def display_tables():
    st.title("Admin Dashboard")
    st.write("Manage users, roles, and app data.")

    # Supabase client setup
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # User Management
    st.header("User Management")
    user_response = supabase.table("users").select("id, username, email, is_admin, created_at").execute()
    users = user_response.data if hasattr(user_response, 'data') else []
    if users:
        user_df = pd.DataFrame(users)
        st.dataframe(user_df)
        selected_user = st.selectbox("Select user to manage", user_df["username"])
        if st.button("Delete User"):
            supabase.table("users").delete().eq("username", selected_user).execute()
            st.success(f"Deleted user: {selected_user}")
            st.rerun()
        if st.button("Promote/Demote Admin"):
            user_row = user_df[user_df["username"] == selected_user].iloc[0]
            new_status = not user_row["is_admin"]
            supabase.table("users").update({"is_admin": new_status}).eq("username", selected_user).execute()
            st.success(f"Updated admin status for: {selected_user}")
            st.rerun()
    else:
        st.info("No users found.")

    # App Data Tables
    st.header("App Data Tables")
    table_options = ["trades", "watchlist"]
    selected_table = st.selectbox("Select table to view", table_options)
    table_response = supabase.table(selected_table).select("*").execute()
    table_data = table_response.data if hasattr(table_response, 'data') else []
    if table_data:
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df)
    else:
        st.info(f"No data found in {selected_table} table.")

    # App Analytics (placeholder)
    st.header("App Analytics")
    st.write("User count:", len(users))
    st.write("Trades count:", len(supabase.table("trades").select("id").execute().data or []))
    st.write("Watchlist count:", len(supabase.table("watchlist").select("id").execute().data or []))

    # Activity Logs (placeholder)
    st.header("Activity Logs")
    st.info("Activity log feature coming soon.")
