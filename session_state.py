import streamlit as st

def init_session_state():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'init_done' not in st.session_state:
        st.session_state.init_done = True

def login(username: str):
    """Log in a user."""
    st.session_state.logged_in = True
    st.session_state.username = username

def logout():
    """Log out the current user."""
    st.session_state.logged_in = False
    st.session_state.username = None

def is_logged_in() -> bool:
    """Check if a user is logged in."""
    init_session_state()  # Ensure session state is initialized
    return bool(st.session_state.logged_in)

def get_username() -> str:
    """Get the current username."""
    init_session_state()  # Ensure session state is initialized
    return st.session_state.username or "Guest"
