import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz
from supabase import create_client, Client
from typing import Dict, List
from session_state import SessionState
import json
import os
from zoneinfo import ZoneInfo
from nba_api.stats.static import teams
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.library.parameters import SeasonAll

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

def load_predictions(start_date: datetime, end_date: datetime):
    """Load predictions from Supabase database within the specified date range."""
    try:
        # Convert dates to UTC for database query
        start_utc = start_date.astimezone(timezone.utc)
        end_utc = end_date.astimezone(timezone.utc)
        
        # Query predictions within date range
        result = (supabase.from_('predictions')
                 .select("*")
                 .gte('scheduled_start', start_utc.isoformat())
                 .lt('scheduled_start', end_utc.isoformat())
                 .order('scheduled_start', desc=True)
                 .execute())
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(result.data)
        
        if not df.empty:
            # Convert scheduled_start to datetime
            df['scheduled_start'] = pd.to_datetime(df['scheduled_start'])
            
            # Convert to Eastern Time
            df['game_time_et'] = df['scheduled_start'].apply(convert_to_et)
            
            # Get game results using API
            nba_teams = teams.get_teams()
            nba_teams_dict = {team['full_name']: team['id'] for team in nba_teams}
            df['home_team_id'] = df['home_team'].apply(lambda x: nba_teams_dict[x])
            df['away_team_id'] = df['away_team'].apply(lambda x: nba_teams_dict[x])
            df['game_id'] = df.apply(lambda x: boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=f"{x['season']}{x['game_id']}",
                season=SeasonAll.current_season,
                season_type_all_star="Regular Season").get_data_frames()[0].iloc[0]['GAME_ID'], axis=1)
            df['game_status'] = df.apply(lambda x: boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=x['game_id'],
                season=SeasonAll.current_season,
                season_type_all_star="Regular Season").get_data_frames()[0].iloc[0]['GAME_STATUS_TEXT'], axis=1)
            df['actual_home_score'] = df.apply(lambda x: boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=x['game_id'],
                season=SeasonAll.current_season,
                season_type_all_star="Regular Season").get_data_frames()[0].iloc[0]['PTS'], axis=1)
            df['actual_away_score'] = df.apply(lambda x: boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=x['game_id'],
                season=SeasonAll.current_season,
                season_type_all_star="Regular Season").get_data_frames()[0].iloc[1]['PTS'], axis=1)
            df['prediction_correct'] = df.apply(lambda x: x['predicted_winner'] == x['home_team'] and x['actual_home_score'] > x['actual_away_score'] or 
                x['predicted_winner'] == x['away_team'] and x['actual_away_score'] > x['actual_home_score'], axis=1)
            
            # Format prediction details
            df['prediction'] = df.apply(lambda x: f"{x['predicted_winner']} ({x['win_probability']:.1%})", axis=1)
            df['predicted_score'] = df.apply(lambda x: f"{x['home_team']} {x['home_score_min']}-{x['home_score_max']} vs {x['away_team']} {x['away_score_min']}-{x['away_score_max']}", axis=1)
            
            # Format actual results if available
            df['actual_result'] = df.apply(lambda x: 
                f"{x.get('actual_home_score', '-')} - {x.get('actual_away_score', '-')}" 
                if x.get('game_status') == 'Final' 
                else x.get('game_status', 'Scheduled'), axis=1)
            
            # Format game time
            df['game_time'] = df['game_time_et'].dt.strftime('%Y-%m-%d %I:%M %p ET')
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return pd.DataFrame()

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
    
    # Add date range selector
    st.subheader("Select Date Range")
    col1, col2 = st.columns(2)
    
    # Default to showing 2 days in the past to 2 weeks in the future
    default_start = datetime.now(timezone.utc) - timedelta(days=2)
    default_end = datetime.now(timezone.utc) + timedelta(weeks=2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start.date(),
            min_value=(default_start - timedelta(weeks=4)).date(),
            max_value=default_end.date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end.date(),
            min_value=start_date,
            max_value=(default_end + timedelta(weeks=4)).date()
        )
    
    # Convert dates to datetime with timezone
    start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    
    # Load predictions for selected date range
    df = load_predictions(start_datetime, end_datetime)
    
    if df.empty:
        st.warning(f"No predictions found between {start_date} and {end_date}.")
        return
    
    # Display summary statistics if we have completed games
    completed_games = df[df['game_status'] == 'Final']
    if not completed_games.empty:
        st.subheader("Prediction Accuracy")
        correct_predictions = completed_games['prediction_correct'].sum()
        total_predictions = len(completed_games)
        accuracy = correct_predictions / total_predictions
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("Correct Predictions", correct_predictions)
        with col3:
            st.metric("Accuracy", f"{accuracy:.1%}")
    
    # Create a table view
    st.subheader("Prediction History")
    table_df = df[[
        'game_time',
        'home_team',
        'away_team',
        'prediction',
        'predicted_score',
        'actual_result'
    ]].copy()
    
    # Rename columns for display
    table_df.columns = [
        'Game Time',
        'Home Team',
        'Away Team',
        'Predicted Winner',
        'Predicted Score Range',
        'Actual Result'
    ]
    
    # Display the table with custom styling
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Game Time": st.column_config.TextColumn(
                "Game Time",
                width="medium",
            ),
            "Predicted Winner": st.column_config.TextColumn(
                "Predicted Winner",
                width="medium",
            ),
            "Predicted Score Range": st.column_config.TextColumn(
                "Predicted Score Range",
                width="large",
            ),
            "Actual Result": st.column_config.TextColumn(
                "Actual Result",
                width="medium",
            ),
        }
    )
    
    # Add detailed view in expandable sections
    st.subheader("Detailed Predictions")
    for _, row in df.iterrows():
        with st.expander(f"{row['home_team']} vs {row['away_team']} - {row['game_time']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Home Team:**", row['home_team'])
                st.write("**Score Range:**", f"{row['home_score_min']}-{row['home_score_max']}")
                if row.get('game_status') == 'Final':
                    st.write("**Actual Score:**", row.get('actual_home_score', '-'))
            
            with col2:
                st.write("**Away Team:**", row['away_team'])
                st.write("**Score Range:**", f"{row['away_score_min']}-{row['away_score_max']}")
                if row.get('game_status') == 'Final':
                    st.write("**Actual Score:**", row.get('actual_away_score', '-'))
            
            st.write("**Predicted Winner:**", row['predicted_winner'])
            st.write("**Win Probability:**", f"{row['win_probability']:.1%}")
            
            if row.get('game_status') == 'Final':
                st.write("**Prediction Outcome:**", 
                        "✅ Correct" if row.get('prediction_correct', False) else "❌ Incorrect")
            else:
                st.write("**Game Status:**", row.get('game_status', 'Scheduled'))
            
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
