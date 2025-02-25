import streamlit as st

class SessionState:
    """Class to manage session state."""
    
    @staticmethod
    def init_state():
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            
        if 'username' not in st.session_state:
            st.session_state.username = None
            
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
            
        if 'supabase_client' not in st.session_state:
            st.session_state.supabase_client = None
            
    @staticmethod
    def login(username: str):
        """Set login state."""
        st.session_state.authenticated = True
        st.session_state.username = username
        
    @staticmethod
    def logout():
        """Clear login state."""
        st.session_state.authenticated = False
        st.session_state.username = None
        
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated."""
        return st.session_state.get('authenticated', False)
        
    @staticmethod
    def get_username() -> str:
        """Get current username."""
        return st.session_state.get('username', None)
        
    @staticmethod
    def set_page(page: str):
        """Set current page."""
        st.session_state.current_page = page
        
    @staticmethod
    def get_page() -> str:
        """Get current page."""
        return st.session_state.get('current_page', 'home')
        
    @staticmethod
    def get(key):
        """Get a value from session state."""
        return st.session_state.get(key)
        
    @staticmethod
    def set(key, value):
        """Set a value in session state."""
        st.session_state[key] = value
        
    @staticmethod
    def clear():
        """Clear all session state variables."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
