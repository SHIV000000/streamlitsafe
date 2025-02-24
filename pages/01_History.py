import streamlit as st
from datetime import datetime, timezone
import pytz
from supabase import create_client, Client
from typing import Dict, List
from session_state import SessionState
import json
import os
from zoneinfo import ZoneInfo

# Initialize session state
SessionState.init_state()

# Get Supabase client from session state
supabase = SessionState.get('supabase_client')
if not supabase:
    # Initialize if not in session state
    SUPABASE_URL = "https://jdvxisvtktunywgdtxvz.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impkdnhpc3Z0a3R1bnl3Z2R0eHZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTE2MDAsImV4cCI6MjA1NTk2NzYwMH0.-Hdbq82ctFUCGjXkmzRDOUzlXkHjVZfp5ws4vpIFmi4"
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    SessionState.set('supabase_client', supabase)

def convert_to_et(utc_time: datetime) -> datetime:
    """Convert UTC time to Eastern Time"""
    et_zone = ZoneInfo("America/New_York")
    return utc_time.replace(tzinfo=timezone.utc).astimezone(et_zone)

def load_predictions():
    """Load predictions from Supabase database."""
    try:
        # Query all predictions from Supabase
        result = supabase.from_('predictions').select("*").order('scheduled_start', desc=True).execute()
        
        predictions = []
        seen_games = set()  # Track unique game combinations
        
        for row in result.data:
            # Create game_info structure
            game_info = {
                'home_team': {'name': row['home_team']},
                'away_team': {'name': row['away_team']},
                'scheduled_start': row['scheduled_start']
            }
            
            # Create prediction structure
            prediction = {
                'predicted_winner': {'name': row['predicted_winner']},
                'win_probability': row['win_probability'],
                'score_prediction': {
                    'home_low': row['home_score_min'],
                    'home_high': row['home_score_max'],
                    'away_low': row['away_score_min'],
                    'away_high': row['away_score_max']
                }
            }
            
            # Create the full prediction object
            prediction_obj = {
                'game_info': game_info,
                'prediction': prediction
            }
            
            # Create a unique key for the game
            game_key = f"{row['home_team']}_{row['away_team']}_{row['scheduled_start']}"
            
            # Only add if we haven't seen this game before
            if game_key not in seen_games:
                predictions.append(prediction_obj)
                seen_games.add(game_key)
        
        return predictions
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return []

def convert_to_et(dt):
    """Convert UTC datetime to ET."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    et = pytz.timezone('US/Eastern')
    return dt.astimezone(et)

def create_navigation():
    """Create navigation bar with buttons"""
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            SessionState.set('current_page', 'predictions')
            st.switch_page("home.py")
    
    with col2:
        if st.button("📊 History", use_container_width=True):
            SessionState.set('current_page', 'history')
            st.rerun()
    
    with col4:
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            SessionState.clear()
            st.switch_page("home.py")
    
    st.divider()

def apply_custom_styles():
    """Apply custom styles"""
    st.markdown("""
        <style>
        /* Hide sidebar and menu */
        [data-testid="stSidebar"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Button styling */
        .stButton > button {
            border: 1px solid #ddd;
            background-color: white;
            color: #333;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background-color: #f0f0f0;
            border-color: #ccc;
        }
        
        /* Primary button (logout) */
        .stButton > button[kind="primary"] {
            background-color: #ff4b4b;
            color: white;
            border: none;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #ff3333;
        }
        </style>
    """, unsafe_allow_html=True)

def display_history_dashboard():
    """Display the history dashboard with predictions and analytics."""
    
    # Load predictions from Supabase
    predictions = load_predictions()
    
    if not predictions:
        st.warning("No prediction history found.")
        return
    
    # Create a simple table view instead of DataFrame
    st.subheader("Prediction History")
    
    for prediction in predictions:
        game_info = prediction['game_info']
        pred_info = prediction['prediction']
        score_pred = pred_info['score_prediction']
        
        with st.expander(f"{game_info['home_team']['name']} vs {game_info['away_team']['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Home Team:**", game_info['home_team']['name'])
                st.write("**Score Range:**", f"{score_pred['home_low']}-{score_pred['home_high']}")
            
            with col2:
                st.write("**Away Team:**", game_info['away_team']['name'])
                st.write("**Score Range:**", f"{score_pred['away_low']}-{score_pred['away_high']}")
            
            st.write("**Predicted Winner:**", pred_info['predicted_winner']['name'])
            st.write("**Win Probability:**", f"{pred_info['win_probability']:.1%}")
            
            game_time = datetime.fromisoformat(game_info['scheduled_start'].replace('Z', '+00:00'))
            st.write("**Game Time:**", game_time.strftime('%Y-%m-%d %H:%M UTC'))
            
            st.divider()

def main():
    """Main function to run the history page"""
    st.set_page_config(
        page_title="NBA Game Predictions - History",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Check authentication
    if not SessionState.get('authenticated'):
        st.warning("Please log in first")
        st.switch_page("home.py")
        return
    
    # Apply custom styles
    apply_custom_styles()
    
    # Show navigation
    create_navigation()
    
    st.title("📊 Game Prediction History")
    
    # Load and display history
    display_history_dashboard()

if __name__ == "__main__":
    main()
