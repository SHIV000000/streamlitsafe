import streamlit as st
from datetime import datetime, timedelta, timezone
import sys
import os
import logging
from typing import List, Dict
import pandas as pd
import numpy as np

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
    """Format datetime string to show only the date."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
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
    if not actual_result or not prediction:
        return "Pending"
        
    predicted_winner = prediction.get('predicted_winner')
    home_score = actual_result.get('home_score')
    away_score = actual_result.get('away_score')
    
    # Determine actual winner based on scores
    if home_score > away_score:
        actual_winner = prediction.get('home_team')
    elif away_score > home_score:
        actual_winner = prediction.get('away_team')
    else:
        actual_winner = None  # Game tied
    
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
        /* Hide sidebar */
        [data-testid="stSidebar"] {display: none !important;}
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        
        /* Button styling */
        .stButton > button {
            width: 100%;
        }
        
        /* Table grid styling - explicit and strong */
        div.prediction-table div[data-testid="stHorizontalBlock"] {
            border-bottom: 2px solid #ccc !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        
        div.prediction-table div[data-testid="stHorizontalBlock"]:last-child {
            border-bottom: none !important;
        }
        
        div.prediction-table div[data-testid="column"] {
            border-right: 2px solid #ccc !important;
            border-bottom: 1px solid #eee !important;
            background-color: white !important;
            padding: 12px 8px !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        
        div.prediction-table div[data-testid="column"]:last-child {
            border-right: none !important;
        }
        
        /* Table header styling - stronger */
        div.table-header div[data-testid="column"] {
            background-color: #2c3e50 !important;
            color: white !important;
            font-weight: bold !important;
            border-bottom: 3px solid #1a252f !important;
            border-right: 2px solid #3d5a74 !important;
            padding: 15px 8px !important;
            font-size: 15px !important;
            text-transform: uppercase !important;
        }
        
        div.table-header div[data-testid="column"]:last-child {
            border-right: none !important;
        }
        
        /* Table container with explicit border */
        .prediction-table {
            border: 2px solid #aaa !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            margin-bottom: 20px !important;
        }
        
        /* Alternating row colors */
        div.table-row:nth-child(even) div[data-testid="column"] {
            background-color: #f5f7fa !important;
        }
        
        /* Highlight row on hover */
        div.table-row:hover div[data-testid="column"] {
            background-color: #e3f2fd !important;
        }
        
        /* Status indicators */
        .status-correct {
            color: #28a745 !important;
            font-weight: bold !important;
            background-color: rgba(40, 167, 69, 0.1) !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
            display: inline-block !important;
        }
        
        .status-incorrect {
            color: #dc3545 !important;
            font-weight: bold !important;
            background-color: rgba(220, 53, 69, 0.1) !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
            display: inline-block !important;
        }
        
        .status-pending {
            color: #6c757d !important;
            font-style: italic !important;
            background-color: rgba(108, 117, 125, 0.1) !important;
            padding: 4px 8px !important;
            border-radius: 4px !important;
            display: inline-block !important;
        }
        
        /* Team score styling */
        .score-display {
            font-weight: bold !important;
            font-family: 'Courier New', monospace !important;
            background-color: #f8f9fa !important;
            padding: 3px 6px !important;
            border-radius: 4px !important;
            border: 1px solid #e9ecef !important;
        }
        
        /* Export button styling */
        .stDownloadButton > button {
            background-color: #3498db !important;
            color: white !important;
            border: none !important;
            padding: 8px 16px !important;
            border-radius: 4px !important;
            font-weight: bold !important;
        }
        
        .stDownloadButton > button:hover {
            background-color: #2980b9 !important;
        }
        
        /* Date input styling */
        div[data-testid="stDateInput"] {
            border: 2px solid #3498db !important;
            border-radius: 8px !important;
            padding: 10px !important;
            background-color: #f8f9fa !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }
        
        div[data-testid="stDateInput"] label {
            font-weight: bold !important;
            color: #2c3e50 !important;
            font-size: 16px !important;
            margin-bottom: 5px !important;
        }
        
        div[data-testid="stDateInput"] input {
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
            padding: 8px !important;
        }
        
        div[data-testid="stDateInput"]:focus-within {
            border-color: #2980b9 !important;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Show navigation
    show_navigation()
    
    st.title("NBA GAMES PREDICTIONS HISTORY")
    
    # Date filters
    st.markdown('<div class="date-filter-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    st.markdown('</div>', unsafe_allow_html=True)
    
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

    # Calculate KPIs
    total_predictions = len(predictions)
    past_predictions = [p for p in predictions if datetime.fromisoformat(p.get('scheduled_start').replace('Z', '+00:00')) < current_time]
    pending_predictions = total_predictions - len(past_predictions)
    
    # Calculate correct predictions and success rate
    correct_predictions = 0
    for p in past_predictions:
        game_key = f"{p.get('home_team')}_{p.get('away_team')}"
        actual_result = all_results.get(game_key)
        if actual_result and get_prediction_accuracy(p, actual_result) == "✅ Correct":
            correct_predictions += 1
    
    success_rate = 0
    if past_predictions:
        success_rate = (correct_predictions / len(past_predictions)) * 100
    
    # Calculate profit and ROI (assuming £1 stake per prediction)
    stake_per_game = 1.0  # £1 per game
    # Assuming average odds of 1.9 for correct predictions
    avg_odds = 1.9
    total_stake = len(past_predictions) * stake_per_game
    total_returns = correct_predictions * avg_odds * stake_per_game
    total_profit = total_returns - total_stake
    roi = 0
    if total_stake > 0:
        roi = (total_profit / total_stake) * 100
    
    # Display KPIs in a nice dashboard
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Prediction Performance</h3>", unsafe_allow_html=True)
    
    # KPI Dashboard styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 12px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create 6 columns for the KPIs
    kpi_cols = st.columns(6)
    
    # Total Predictions
    with kpi_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(total_predictions), unsafe_allow_html=True)
    
    # Correct Predictions
    with kpi_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Correct Predictions</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(correct_predictions), unsafe_allow_html=True)
    
    # Success Rate
    with kpi_cols[2]:
        success_class = "positive" if success_rate >= 50 else "negative"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value {}">{}%</div>
        </div>
        """.format(success_class, round(success_rate, 1)), unsafe_allow_html=True)
    
    # Total Profit
    with kpi_cols[3]:
        profit_class = "positive" if total_profit >= 0 else "negative"
        profit_sign = "+" if total_profit > 0 else ""
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Profit</div>
            <div class="metric-value {}">{}£{:.2f}</div>
        </div>
        """.format(profit_class, profit_sign, total_profit), unsafe_allow_html=True)
    
    # ROI
    with kpi_cols[4]:
        roi_class = "positive" if roi >= 0 else "negative"
        roi_sign = "+" if roi > 0 else ""
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ROI</div>
            <div class="metric-value {}">{}{}%</div>
        </div>
        """.format(roi_class, roi_sign, round(roi, 2)), unsafe_allow_html=True)
    
    # Pending Predictions
    with kpi_cols[5]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pending Predictions</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(pending_predictions), unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Convert predictions to a pandas DataFrame for better display
    table_data = []
    for prediction in predictions:
        game_key = f"{prediction.get('home_team')}_{prediction.get('away_team')}"
        actual_result = all_results.get(game_key, {})
        
        # Get basic data
        date = format_datetime(prediction.get('scheduled_start'))
        home_team = prediction.get('home_team')
        away_team = prediction.get('away_team')
        home_score = actual_result.get('home_score', '-')
        away_score = actual_result.get('away_score', '-')
        
        # Format predicted scores
        home_range = f"{prediction.get('home_score_min')}-{prediction.get('home_score_max')}"
        away_range = f"{prediction.get('away_score_min')}-{prediction.get('away_score_max')}"
        
        # Get actual winner
        actual_winner = '-'
        if home_score != '-':
            if int(home_score) > int(away_score):
                actual_winner = home_team
            elif int(away_score) > int(home_score):
                actual_winner = away_team
            else:
                actual_winner = 'Tie'
        
        # Format actual score
        actual_score = f"{home_score}-{away_score}" if home_score != '-' else "Pending"
        
        # Get prediction accuracy
        accuracy = get_prediction_accuracy(prediction, actual_result)
        
        # Format win probability with confidence level
        win_prob = prediction.get('win_probability', 0)
        if win_prob >= 70:
            confidence = "HIGH"
        elif win_prob >= 50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
            
        table_data.append({
            'Date': date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Predicted Winner': prediction.get('predicted_winner'),
            'Confidence': confidence,
            'Home Score Range': home_range,
            'Away Score Range': away_range,
            'Actual Score': actual_score,
            'Actual Winner': actual_winner,
            'Result': accuracy
        })

    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by date (newest first)
    df['_date_sort'] = pd.to_datetime(df['Date'])
    df = df.sort_values('_date_sort', ascending=False)
    df = df.drop('_date_sort', axis=1)
    
    # Display the DataFrame with custom formatting
    st.markdown("### Prediction History")
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn(
                "Date",
                width="small",
            ),
            "Home Team": st.column_config.TextColumn(
                "Home Team",
                width="medium",
            ),
            "Away Team": st.column_config.TextColumn(
                "Away Team",
                width="medium",
            ),
            "Predicted Winner": st.column_config.TextColumn(
                "Predicted Winner",
                width="medium",
            ),
            "Confidence": st.column_config.Column(
                "Confidence",
                width="small",
                help="High (≥70%), Medium (50-69%), Low (<50%)",
                required=True
            ),
            "Home Score Range": st.column_config.TextColumn(
                "Home Score Range",
                width="small",
            ),
            "Away Score Range": st.column_config.TextColumn(
                "Away Score Range",
                width="small",
            ),
            "Actual Score": st.column_config.TextColumn(
                "Actual Score",
                width="small",
            ),
            "Actual Winner": st.column_config.TextColumn(
                "Actual Winner",
                width="medium",
            ),
            "Result": st.column_config.TextColumn(
                "Result",
                width="small",
            ),
        },
        height=600,
        use_container_width=True,
    )
    
    # Add custom CSS for better table styling
    st.markdown("""
        <style>
        /* Table styling */
        [data-testid="stDataFrame"] {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Header styling */
        [data-testid="stDataFrame"] th {
            background-color: #f8fafc !important;
            color: #475569 !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-size: 0.9em !important;
            padding: 12px 16px !important;
            border-bottom: 2px solid #e2e8f0 !important;
        }
        
        /* Cell styling */
        [data-testid="stDataFrame"] td {
            padding: 12px 16px !important;
            font-size: 14px !important;
            border-bottom: 1px solid #f1f5f9 !important;
        }
        
        /* Row hover effect */
        [data-testid="stDataFrame"] tr:hover td {
            background-color: #f8fafc !important;
        }
        
        /* Team name cells */
        [data-testid="stDataFrame"] td:nth-child(2),
        [data-testid="stDataFrame"] td:nth-child(3) {
            font-weight: 500 !important;
        }
        
        /* Score cells */
        [data-testid="stDataFrame"] td:nth-child(6),
        [data-testid="stDataFrame"] td:nth-child(7),
        [data-testid="stDataFrame"] td:nth-child(8) {
            font-family: 'SF Mono', 'Roboto Mono', monospace !important;
            background-color: #f8fafc !important;
        }
        
        /* Confidence cells */
        [data-testid="stDataFrame"] td:nth-child(5) {
            font-weight: 500 !important;
        }
        
        /* Color confidence levels */
        [data-testid="stDataFrame"] td:nth-child(5):contains("HIGH") {
            color: #166534 !important;
            background-color: #dcfce7 !important;
        }
        [data-testid="stDataFrame"] td:nth-child(5):contains("MEDIUM") {
            color: #854d0e !important;
            background-color: #fef9c3 !important;
        }
        [data-testid="stDataFrame"] td:nth-child(5):contains("LOW") {
            color: #991b1b !important;
            background-color: #fee2e2 !important;
        }
        
        /* Result cells */
        [data-testid="stDataFrame"] td:last-child {
            font-weight: 500 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
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
            "Date,Home Team,Away Team,Predicted,Confidence,Home,Away,Score,Winner,Result"
        ]
        for prediction in predictions:
            game_key = f"{prediction.get('home_team')}_{prediction.get('away_team')}"
            actual_result = all_results.get(game_key)
            game_time = datetime.fromisoformat(prediction.get('scheduled_start').replace('Z', '+00:00'))
            is_past_game = game_time < current_time
            
            actual_score = "Upcoming"
            actual_winner = "Upcoming"
            accuracy = "Upcoming"
            if is_past_game:
                if actual_result:
                    actual_score = f"{actual_result['home_score']}-{actual_result['away_score']}"
                    actual_winner = actual_result.get('home_team') if actual_result.get('winner') == 'home' else actual_result.get('away_team')
                    accuracy = get_prediction_accuracy(prediction, actual_result)
                else:
                    actual_score = "No result"
                    actual_winner = "No result"
                    accuracy = "No result"
            
            # Format date to only show the date part
            prediction_date = format_datetime(prediction.get('created_at', ''))
            
            csv_rows.append(
                f"{prediction_date},{prediction.get('home_team', 'N/A')},{prediction.get('away_team', 'N/A')},"
                f"{prediction.get('predicted_winner', 'N/A')},{prediction.get('confidence', 'N/A')},"
                f"{prediction.get('home_score_min', 'N/A')}-{prediction.get('home_score_max', 'N/A')},"
                f"{prediction.get('away_score_min', 'N/A')}-{prediction.get('away_score_max', 'N/A')},"
                f"{actual_score},{actual_winner},{accuracy}"
            )
        
        csv_data = "\n".join(csv_rows)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"predictions_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show_history()