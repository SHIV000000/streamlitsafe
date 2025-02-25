import streamlit as st
from datetime import datetime, timedelta, timezone
import sys
import os
import logging
from typing import List, Dict

# Add parent directory to path to import from parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from session_state import init_session_state, is_logged_in, login, logout, get_username
from supabase import Client, create_client
from nba_api_client import NBAGameResultsFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)

# Supabase configuration
SUPABASE_URL = "https://jdvxisvtktunywgdtxvz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impkdnhpc3Z0a3R1bnl3Z2R0eHZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTE2MDAsImV4cCI6MjA1NTk2NzYwMH0.-Hdbq82ctFUCGjXkmzRDOUzlXkHjVZfp5ws4vpIFmi4"

def init_supabase():
    """Initialize Supabase client."""
    if 'supabase' not in st.session_state:
        st.session_state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return st.session_state.supabase

def format_datetime(dt_str: str) -> str:
    """Format datetime string."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return 'N/A'

def get_game_date(dt_str: str) -> str:
    """Get game date in YYYY-MM-DD format."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except:
        return None

def load_predictions(start_date=None, end_date=None):
    """Load predictions from Supabase."""
    try:
        supabase = init_supabase()
        query = supabase.table('predictions').select('*')
        
        if start_date:
            query = query.gte('created_at', start_date.isoformat())
        if end_date:
            query = query.lte('created_at', end_date.isoformat())
            
        response = query.execute()
        return response.data
        
    except Exception as e:
        logging.error(f"Error loading predictions: {str(e)}")
        return []

def show_navigation():
    """Show navigation bar."""
    col1, col2, col3 = st.columns([2,2,1])
    
    with col1:
        st.markdown(f"👤 Welcome, {get_username()}")
    with col2:
        if st.button("🏠 Home"):
            st.switch_page("home.py")
    with col3:
        if st.button("🚪 Logout"):
            logout()
            st.switch_page("home.py")

def get_prediction_accuracy(prediction: Dict, actual_result: Dict) -> str:
    """Calculate prediction accuracy and return formatted string."""
    if not actual_result:
        return "Pending"
        
    predicted_winner = prediction.get('predicted_winner')
    actual_winner = actual_result.get('winner')
    
    if predicted_winner == actual_winner:
        return "✅ Correct"
    return "❌ Incorrect"

def show_history():
    """Show prediction history."""
    # Initialize session state
    init_session_state()
    
    # Check login status and redirect if not logged in
    if not is_logged_in():
        st.switch_page("home.py")
        return
    
    # Remove sidebar and style columns
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        .stButton > button {
            width: 100%;
        }
        div[data-testid="column"] {
            background-color: white;
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        div[data-testid="column"]:nth-child(even) {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Show navigation
    show_navigation()
    
    st.title("📊 Prediction History")
    
    # Date filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Load predictions from database
    predictions = load_predictions(start_date, end_date)
    
    if not predictions:
        st.info("No predictions found for the selected date range.")
        return

    # Initialize NBA client
    nba_client = NBAGameResultsFetcher()
    
    # Get game results for each prediction
    all_results = {}
    current_time = datetime.now(timezone.utc)
    
    for pred in predictions:
        game_date = get_game_date(pred.get('scheduled_start'))
        if game_date:
            # Parse the game time
            game_time = datetime.fromisoformat(pred.get('scheduled_start').replace('Z', '+00:00'))
            
            # Only get results for past games
            if game_time < current_time:
                results = nba_client.get_game_results(
                    game_date,
                    home_team=pred.get('home_team'),
                    away_team=pred.get('away_team')
                )
                
                if results:
                    # Create a key using both teams to match with predictions
                    key = f"{pred.get('home_team')}_{pred.get('away_team')}"
                    all_results[key] = results[0]

    # Display table headers
    st.markdown("---")
    header_cols = st.columns([1.5, 2, 1.5, 1, 1, 1, 1.5, 1.5])
    with header_cols[0]:
        st.markdown("**Date**")
    with header_cols[1]:
        st.markdown("**Match**")
    with header_cols[2]:
        st.markdown("**Predicted Winner**")
    with header_cols[3]:
        st.markdown("**Win %**")
    with header_cols[4]:
        st.markdown("**Home Score**")
    with header_cols[5]:
        st.markdown("**Away Score**")
    with header_cols[6]:
        st.markdown("**Game Time**")
    with header_cols[7]:
        st.markdown("**Result**")
    st.markdown("---")

    # Display predictions with results
    for pred in predictions:
        # Get actual result
        game_key = f"{pred.get('home_team')}_{pred.get('away_team')}"
        actual_result = all_results.get(game_key)
        
        # Parse game time
        game_time = datetime.fromisoformat(pred.get('scheduled_start').replace('Z', '+00:00'))
        is_past_game = game_time < current_time
        
        cols = st.columns([1.5, 2, 1.5, 1, 1, 1, 1.5, 1.5])
        with cols[0]:
            st.write(format_datetime(pred.get('created_at', '')))
        with cols[1]:
            st.write(f"{pred.get('home_team', 'N/A')} vs {pred.get('away_team', 'N/A')}")
        with cols[2]:
            st.write(pred.get('predicted_winner', 'N/A'))
        with cols[3]:
            st.write(f"{pred.get('win_probability', 0):.1f}%")
        with cols[4]:
            st.write(f"{pred.get('home_score_min', 'N/A')}-{pred.get('home_score_max', 'N/A')}")
        with cols[5]:
            st.write(f"{pred.get('away_score_min', 'N/A')}-{pred.get('away_score_max', 'N/A')}")
        with cols[6]:
            st.write(format_datetime(pred.get('scheduled_start', '')))
        with cols[7]:
            if is_past_game:
                if actual_result:
                    st.write(f"{actual_result['home_score']}-{actual_result['away_score']}\n{get_prediction_accuracy(pred, actual_result)}")
                else:
                    st.write("No result")
            else:
                st.write("Upcoming")
    
    st.markdown("---")
    
    # Calculate and display prediction accuracy
    past_predictions = [p for p in predictions if datetime.fromisoformat(p.get('scheduled_start').replace('Z', '+00:00')) < current_time]
    if past_predictions:
        correct_predictions = sum(1 for p in past_predictions if get_prediction_accuracy(p, all_results.get(f"{p.get('home_team')}_{p.get('away_team')}")) == "✅ Correct")
        accuracy = (correct_predictions / len(past_predictions)) * 100
        st.info(f"📈 Overall Prediction Accuracy: {accuracy:.1f}% ({correct_predictions} correct out of {len(past_predictions)} past predictions)")
    
    # Add export functionality
    if st.button("📥 Export to CSV"):
        # Convert predictions to CSV format
        csv_rows = [
            "Date,Match,Predicted Winner,Win Probability,Home Score Range,Away Score Range,Game Time,Actual Score,Accuracy"
        ]
        for pred in predictions:
            game_key = f"{pred.get('home_team')}_{pred.get('away_team')}"
            actual_result = all_results.get(game_key)
            game_time = datetime.fromisoformat(pred.get('scheduled_start').replace('Z', '+00:00'))
            is_past_game = game_time < current_time
            
            actual_score = "Upcoming"
            accuracy = "Upcoming"
            if is_past_game:
                if actual_result:
                    actual_score = f"{actual_result['home_score']}-{actual_result['away_score']}"
                    accuracy = get_prediction_accuracy(pred, actual_result)
                else:
                    actual_score = "No result"
                    accuracy = "No result"
            
            csv_rows.append(
                f"{format_datetime(pred.get('created_at', ''))},{pred.get('home_team', 'N/A')} vs {pred.get('away_team', 'N/A')},"
                f"{pred.get('predicted_winner', 'N/A')},{pred.get('win_probability', 0):.1f}%,"
                f"{pred.get('home_score_min', 'N/A')}-{pred.get('home_score_max', 'N/A')},"
                f"{pred.get('away_score_min', 'N/A')}-{pred.get('away_score_max', 'N/A')},"
                f"{format_datetime(pred.get('scheduled_start', ''))},"
                f"{actual_score},{accuracy}"
            )
        
        csv_data = "\n".join(csv_rows)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"predictions_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    show_history()