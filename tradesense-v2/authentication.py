import streamlit as st
from supabase import create_client, Client
import os
from utils.constants import SUPABASE_URL, SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def login():
    st.subheader("Login")
    login_input = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_input and password:
            # Try to find user by username or email
            response = supabase.table("users").select("username, email, password, is_admin").or_(f"username.eq.{login_input},email.eq.{login_input}").execute()
            data = getattr(response, 'data', None)
            error = getattr(response, 'error', None)
            if error is None and data:
                user = data[0]
                # For demo: check password directly
                if user['password'] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = user['username']
                    st.session_state.is_admin = user.get('is_admin', False)
                    st.success(f"Welcome, {user['username']}!")
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            else:
                st.error("User not found.")
        else:
            st.error("Please enter both username/email and password.")

def signup():
    st.subheader("Sign Up")
    full_name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth")
    city = st.text_input("City")
    country = st.text_input("Country")
    phone = st.text_input("Phone Number")
    username = st.text_input("Choose a Username")
    email = st.text_input("Email")
    password = st.text_input("Choose a Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        if all([full_name, dob, city, country, phone, username, email, password, confirm]):
            if password == confirm:
                # Save to Supabase
                try:
                    response = supabase.table("users").insert({
                        "full_name": full_name,
                        "dob": str(dob),
                        "city": city,
                        "country": country,
                        "phone": phone,
                        "username": username,
                        "email": email,
                        "password": password  # For demo only! Hash in production.
                    }).execute()
                    if response.error is None:
                        st.success("Account created! Please login.")
                    else:
                        st.error(f"Signup failed: {response.error.message}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Passwords do not match.")
        else:
            st.error("Please fill all fields.")

def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email")
    if st.button("Reset Password"):
        if email:
            st.info(f"Password reset link sent to {email} (simulated)")
        else:
            st.error("Please enter your email.")
