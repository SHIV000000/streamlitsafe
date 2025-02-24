import streamlit as st
from typing import Any, Dict

class SessionState:
    """Manages session state across all pages"""
    
    @staticmethod
    def init_state():
        """Initialize all session state variables"""
        defaults = {
            'current_page': 'predictions',
            'supabase_client': None,
            'predictions': [],
            'last_refresh': None,
            'show_history': False,
            'filter_date': None,
            'selected_teams': [],
        }
        
        # Initialize each state variable if not already present
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a session state value"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any):
        """Set a session state value"""
        st.session_state[key] = value
    
    @staticmethod
    def clear():
        """Clear all session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        SessionState.init_state()
    
    @staticmethod
    def navigate_to(page: str):
        """Navigate to a different page"""
        st.session_state.current_page = page
    
    @staticmethod
    def get_all_state() -> Dict[str, Any]:
        """Get all session state as a dictionary"""
        return dict(st.session_state)
