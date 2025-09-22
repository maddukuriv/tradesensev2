import streamlit as st
from supabase import create_client, Client
from utils.constants import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def my_account():
    st.title("My Account")
    username = st.session_state.get("username", None)
    if not username:
        st.warning("No user is currently logged in.")
        return
    try:
        response = supabase.table("users").select(
            "full_name, dob, city, country, phone, username, email, password"
        ).filter("username", "eq", username).execute()
        data = getattr(response, 'data', None)
        error = getattr(response, 'error', None)
        if error is None and data:
            user = data[0]
            st.subheader(f"Welcome, {user['full_name']}")
            with st.form("edit_account_form"):
                full_name = st.text_input("Full Name", value=user['full_name'])
                dob = st.text_input("Date of Birth", value=user['dob'])
                city = st.text_input("City", value=user['city'])
                country = st.text_input("Country", value=user['country'])
                phone = st.text_input("Phone", value=user['phone'])
                email = st.text_input("Email", value=user['email'])
                password = st.text_input("Password", value=user['password'], type="password")
                submitted = st.form_submit_button("Save Changes")
                if submitted:
                    try:
                        update_response = supabase.table("users").update({
                            "full_name": full_name,
                            "dob": dob,
                            "city": city,
                            "country": country,
                            "phone": phone,
                            "email": email,
                            "password": password
                        }).filter("username", "eq", username).execute()
                        update_error = getattr(update_response, 'error', None)
                        if update_error is None:
                            st.success("Account details updated successfully!")
                        else:
                            st.error(f"Update failed: {update_error}")
                    except Exception as e:
                        st.error(f"Error updating account: {e}")
        else:
            st.error("Could not fetch account details.")
    except Exception as e:
        st.error(f"Error: {e}")

# To call the function in your main Streamlit app
if __name__ == "__main__":
    my_account()