# 1. Standard Library Imports
import os
import json
import time
import threading
import logging
import atexit
from datetime import datetime, timedelta
from typing import Dict

# 2. Third-Party Imports
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit_authenticator as stauth

# 3. Local/Custom Imports
from test_predictions import (
    run_continuous_predictions,
    LiveGamePredictor,
    NBAPredictor
)
from api_client import EnhancedNBAApiClient
from reward_system import RewardSystemManager



# 2. Page config and CSS
st.set_page_config(
    page_title="NBA Game Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with mobile-friendly updates
st.markdown("""
    <style>
    /* Main container with basketball texture */
    .main {
        background-color: #1a1a1a;
        background-image: linear-gradient(30deg, #2c2c2c 12%, transparent 12.5%, transparent 87%, #2c2c2c 87.5%, #2c2c2c),
        linear-gradient(150deg, #2c2c2c 12%, transparent 12.5%, transparent 87%, #2c2c2c 87.5%, #2c2c2c),
        linear-gradient(30deg, #2c2c2c 12%, transparent 12.5%, transparent 87%, #2c2c2c 87.5%, #2c2c2c),
        linear-gradient(150deg, #2c2c2c 12%, transparent 12.5%, transparent 87%, #2c2c2c 87.5%, #2c2c2c),
        linear-gradient(60deg, #323232 25%, transparent 25.5%, transparent 75%, #323232 75%, #323232),
        linear-gradient(60deg, #323232 25%, transparent 25.5%, transparent 75%, #323232 75%, #323232);
        background-size: 80px 140px;
        background-position: 0 0, 0 0, 40px 70px, 40px 70px, 0 0, 40px 70px;
        padding: 1rem;
        color: #ffffff;
    }

    /* Responsive text sizes */
    @media (max-width: 768px) {
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
        p, div { font-size: 0.95rem !important; }
    }

    /* Enhanced prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #2C3E50 0%, #1a1a1a 100%);
        border-radius: 15px;
        padding: clamp(1rem, 3vw, 1.5rem);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
        border-left: 5px solid #FF6B00;
        color: #ffffff;
    }

    /* Responsive metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        padding: 1rem 0;
    }

    /* Enhanced metric containers */
    .metric-container {
        background: linear-gradient(145deg, #ffffff, #f3f4f6);
        border-radius: 12px;
        padding: clamp(0.75rem, 2vw, 1rem);
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    .metric-label {
        color: #4b5563;
        font-size: clamp(0.75rem, 1.5vw, 0.875rem);
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .metric-value {
        color: #1f2937;
        font-size: clamp(1.25rem, 2.5vw, 1.5rem);
        font-weight: bold;
    }

    /* Team comparison section */
    .team-comparison {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem 0;
    }

    @media (min-width: 768px) {
        .team-comparison {
            flex-direction: row;
        }
    }

    /* Enhanced buttons */
    .stButton>button {
        width: 100%;
        max-width: 300px;
        background-color: #3b82f6;
        color: white;
        border-radius: 10px;
        padding: clamp(0.5rem, 1.5vw, 0.75rem) clamp(1rem, 3vw, 2rem);
        border: none;
        transition: all 0.3s ease;
        font-size: clamp(0.875rem, 1.5vw, 1rem);
    }

    /* Authentication form improvements */
    .auth-container {
        max-width: min(400px, 90vw);
        margin: 2rem auto;
        padding: clamp(1rem, 3vw, 2rem);
    }

    /* Sidebar improvements */
    .css-1d391kg {
        padding: clamp(1rem, 3vw, 2rem);
    }

    /* Tables responsiveness */
    .stats-table {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    /* Charts responsiveness */
    .plotly-chart {
        width: 100% !important;
        height: auto !important;
    }

    /* Score display improvements */
    .score-display {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: clamp(0.5rem, 2vw, 1rem);
        padding: clamp(0.5rem, 2vw, 1rem);
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Team names */
    .team-name {
        font-size: clamp(1rem, 2vw, 1.2rem);
        font-weight: bold;
        color: #60a5fa;
    }

    /* Live indicator enhancement */
    .live-game {
        padding: clamp(0.25rem, 1vw, 0.5rem) clamp(0.5rem, 2vw, 1rem);
        font-size: clamp(0.75rem, 1.5vw, 0.875rem);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Helper functions
def create_custom_metric(label, value, delta=None):
    """Create a custom styled metric"""
    metric_html = f"""
        <div class="custom-metric">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
    """
    return st.markdown(metric_html, unsafe_allow_html=True)

def create_team_comparison_chart(home_team, away_team, home_prob, away_prob):
    """Create an interactive team comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[home_team],
        x=[home_prob],
        orientation='h',
        name=home_team,
        marker_color='#3b82f6',
        text=f'{home_prob:.1%}',
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        y=[away_team],
        x=[away_prob],
        orientation='h',
        name=away_team,
        marker_color='#ef4444',
        text=f'{away_prob:.1%}',
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Win Probability Comparison",
        barmode='group',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_metric(label, value):
    """Create a custom styled metric"""
    return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """

def auto_update():
    """Enhanced auto-update function with live game detection"""
    current_time = time.time()
    
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = current_time
        return False
    
    # Update more frequently if there are live games
    update_interval = 300  # 5 minutes in seconds
    
    time_elapsed = current_time - st.session_state.last_update_time
    if time_elapsed >= update_interval:
        try:
            logging.info("Starting auto-update...")
            st.session_state.is_predicting = True
            clean_old_predictions()
            success = run_continuous_predictions(timeout_minutes=3)
            
            if success:
                st.session_state.last_update_time = current_time
                st.session_state.last_prediction_time = current_time
                st.session_state.update_counter += 1
                logging.info("Auto-update completed successfully")
                return True
                
        except Exception as e:
            logging.error(f"Error in auto-update: {str(e)}")
        finally:
            st.session_state.is_predicting = False
            
    return False


def show_update_status():
    """Show update status in sidebar with countdown timer"""
    if not hasattr(st.session_state, 'last_update_time'):
        return
        
    current_time = time.time()
    time_since_last_update = current_time - st.session_state.last_update_time
    time_until_next_update = max(300 - time_since_last_update, 0)
    
    minutes = int(time_until_next_update // 60)
    seconds = int(time_until_next_update % 60)
    
    st.sidebar.markdown("### Update Status")
    
    # Show last update time
    last_update = datetime.fromtimestamp(st.session_state.last_update_time)
    st.sidebar.info(f"Last update: {last_update.strftime('%H:%M:%S')}")
    
    # Show countdown
    if time_until_next_update > 0:
        st.sidebar.warning(f"Next update in: {minutes:02d}:{seconds:02d}")
    else:
        st.sidebar.success("Update due...")
    
    # Show prediction status
    if st.session_state.is_predicting:
        st.sidebar.warning("⏳ Predictions in progress...")
    else:
        st.sidebar.success("✅ Ready for next update")

# 3. Add Data Validation Function
def validate_prediction_data(prediction):
    """Validate prediction data structure and content"""
    try:
        # Check basic structure
        required_fields = ['game_info', 'prediction', 'timestamp']
        if not all(field in prediction for field in required_fields):
            return False

        # Validate game info
        game_info = prediction['game_info']
        required_game_fields = ['id', 'home_team', 'away_team', 'scheduled_start']
        if not all(field in game_info for field in required_game_fields):
            return False

        # Validate timestamp
        timestamp = datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00'))
        current_time = datetime.now(timestamp.tzinfo)
        
        # Check if prediction is not too old (more than 6 hours)
        if (current_time - timestamp).total_seconds() > 21600:
            return False

        return True
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False

def load_predictions(include_live=True):
    """Load only valid and recent predictions"""
    try:
        predictions = []
        current_time = time.time()
        
        def get_latest_prediction_files(directory):
            game_files = {}
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if file.endswith('.json'):
                        try:
                            game_id = file.split('_')[1]
                            file_path = os.path.join(directory, file)
                            mod_time = os.path.getmtime(file_path)
                            
                            # Only include files less than 6 hours old
                            if current_time - mod_time <= 21600:  # 6 hours in seconds
                                if game_id not in game_files or mod_time > game_files[game_id][1]:
                                    game_files[game_id] = (file_path, mod_time)
                        except Exception as e:
                            logging.error(f"Error processing file {file}: {str(e)}")
                            continue
            
            return [path for path, _ in game_files.values()]

        # Load predictions from both directories
        for directory in ["predictions/scheduled", "predictions/live"]:
            for file_path in get_latest_prediction_files(directory):
                try:
                    with open(file_path, 'r') as f:
                        pred = json.load(f)
                        if pred and validate_prediction_structure(pred):
                            # Ensure scheduled_start exists and is valid
                            if 'game_info' in pred and 'scheduled_start' in pred['game_info']:
                                pred['is_live'] = 'live' in directory
                                predictions.append(pred)
                except Exception as e:
                    logging.error(f"Error loading prediction file {file_path}: {str(e)}")
                    continue

        # Sort predictions by scheduled start time if available
        if predictions:
            # Use a default value for sorting to handle None cases
            predictions.sort(
                key=lambda x: x.get('game_info', {}).get('scheduled_start', '9999-12-31T23:59:59'),
                reverse=False
            )
            
        return predictions
        
    except Exception as e:
        logging.error(f"Error loading predictions: {str(e)}")
        return []


# 5. Add Update Control Function
def should_update_predictions():
    """Determine if predictions should be updated"""
    if 'last_update_time' not in st.session_state:
        return True
        
    current_time = time.time()
    time_since_update = current_time - st.session_state.last_update_time
    
    # Check if any games are live
    predictions = load_predictions()
    has_live_games = any(p.get('is_live', False) for p in predictions)
    
    # Update more frequently if there are live games
    update_interval = 180 if has_live_games else 300  # 3 or 5 minutes
    
    return time_since_update >= update_interval


def update_predictions():
    """Handle prediction updates with proper control"""
    try:
        # Force update when manually triggered
        from test_predictions import force_prediction_update
        force_prediction_update()
        
        logging.info("Starting manual prediction update...")
        
        # Clean old predictions first
        clean_old_predictions()
        
        # Run predictions with timeout
        success = run_continuous_predictions(timeout_minutes=3)
        
        if success:
            st.session_state.last_update_time = time.time()
            st.session_state.update_counter += 1
            logging.info("Prediction update completed successfully")
            return True
        else:
            logging.error("Prediction update failed")
            return False
            
    except Exception as e:
        logging.error(f"Error in update_predictions: {str(e)}")
        return False


# 4. Core functionality
def display_live_game_card(prediction, key_prefix=None):
    """Enhanced live game card with real-time updates"""
    try:
        game_info = prediction['game_info']
        pred_info = prediction['prediction']
        
        # Extract team names safely
        home_team_name = extract_team_name(game_info['home_team'])
        away_team_name = extract_team_name(game_info['away_team'])
        
        with st.container():
            st.markdown('<div class="prediction-card live-game">', unsafe_allow_html=True)
            
            # Live indicator with period and clock
            col1, col2 = st.columns([6,1])
            with col1:
                st.markdown(f"### 🔴 LIVE: Period {game_info.get('period', 'N/A')}")
            with col2:
                st.markdown(f"⏰ {game_info.get('clock', 'N/A')}")
            
            # Current score
            col1, col2, col3 = st.columns([2,1,2])
            with col1:
                st.metric(
                    label=home_team_name,
                    value=game_info.get('score', {}).get('home', 0)
                )
            with col2:
                st.markdown("VS")
            with col3:
                st.metric(
                    label=away_team_name,
                    value=game_info.get('score', {}).get('away', 0)
                )
            
            # Updated prediction
            st.markdown("### Live Win Probability")
            prob_chart = create_team_comparison_chart(
                home_team_name,
                away_team_name,
                pred_info['win_probability'] if pred_info['predicted_winner'] == home_team_name 
                else 1 - pred_info['win_probability'],
                pred_info['win_probability'] if pred_info['predicted_winner'] == away_team_name 
                else 1 - pred_info['win_probability']
            )
            st.plotly_chart(prob_chart, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        logging.error(f"Error displaying live game card: {str(e)}")
        st.error("Error displaying live game information")

def display_scheduled_game_card(prediction, key_prefix=None):
    """Display comprehensive scheduled game prediction card."""
    try:
        game_info = prediction.get('game_info', {})
        pred_info = prediction.get('prediction', {})
        
        # Extract team names safely
        home_team_name = extract_team_name(game_info['home_team'])
        away_team_name = extract_team_name(game_info['away_team'])
        
        # Extract win probability and predicted winner
        win_prob = float(pred_info.get('win_probability', 0.5))
        predicted_winner = pred_info.get('predicted_winner', '')
        
        # Calculate probabilities
        home_prob = win_prob if predicted_winner == home_team_name else 1 - win_prob
        away_prob = 1 - home_prob
        
        with st.container():
            st.markdown(f"<div class='prediction-card'>", unsafe_allow_html=True)
            
            # Matchup Section
            st.markdown("<div class='section-header'>🏀 MATCHUP:</div>", unsafe_allow_html=True)
            st.markdown(f"{home_team_name} (Home) vs {away_team_name} (Away)")
            
            # Win Probability Section
            st.markdown("<div class='section-header'>📊 WIN PROBABILITY:</div>", unsafe_allow_html=True)
            st.markdown(f"{home_team_name}: {home_prob:.1%}")
            st.markdown(f"{away_team_name}: {away_prob:.1%}")
            
            # Score prediction section
            score_pred = pred_info.get('score_prediction', {})
            if score_pred:
                st.markdown("<div class='section-header'>🎯 PREDICTED SCORE RANGE:</div>", unsafe_allow_html=True)
                st.markdown(f"{home_team_name}: {score_pred.get('home_low', 0)}-{score_pred.get('home_high', 0)} points")
                st.markdown(f"{away_team_name}: {score_pred.get('away_low', 0)}-{score_pred.get('away_high', 0)} points")
            
            # Add rewards section if game is completed
            if 'actual_result' in prediction:
                st.markdown("### 🏆 Rewards")
                rewards = prediction.get('rewards', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Coins Earned", rewards.get('coins', 0))
                with col2:
                    st.metric("Boost Points", rewards.get('boost_points', 0))
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        logging.error(f"Error displaying scheduled game card: {str(e)}")
        st.error("Error displaying game prediction details")

def clean_old_predictions():
    """Delete old prediction files and keep only the latest for each game"""
    try:
        # First, completely clean the live predictions directory
        live_dir = "predictions/live"
        if os.path.exists(live_dir):
            for file in os.listdir(live_dir):
                try:
                    os.remove(os.path.join(live_dir, file))
                    logging.debug(f"Removed live prediction file: {file}")
                except Exception as e:
                    logging.error(f"Error deleting live file {file}: {str(e)}")

        # Handle scheduled games directory (keep latest for each game)
        scheduled_dir = "predictions/scheduled"
        if os.path.exists(scheduled_dir):
            game_files = {}
            for file in os.listdir(scheduled_dir):
                if file.endswith(".json"):
                    game_id = file.split('_')[1]  # Extract game ID from filename
                    file_path = os.path.join(scheduled_dir, file)
                    if game_id not in game_files:
                        game_files[game_id] = []
                    game_files[game_id].append((file_path, os.path.getmtime(file_path)))
            
            # Keep only the latest file for each scheduled game
            for game_id, files in game_files.items():
                sorted_files = sorted(files, key=lambda x: x[1])
                for file_path, _ in sorted_files[:-1]:
                    try:
                        os.remove(file_path)
                        logging.debug(f"Removed old scheduled prediction file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting scheduled file {file_path}: {str(e)}")

    except Exception as e:
        logging.error(f"Error in clean_old_predictions: {str(e)}")


# 3. Fix Prediction Structure Validation
def validate_prediction_structure(prediction):
    """Validate the prediction structure including score prediction"""
    try:
        # Check if prediction is None
        if prediction is None:
            logging.error("Prediction is None")
            return False
            
        # Check basic structure
        if not isinstance(prediction, dict):
            logging.error("Prediction is not a dictionary")
            return False
            
        # Validate game_info
        game_info = prediction.get('game_info')
        if not isinstance(game_info, dict):
            logging.error("game_info is not a dictionary")
            return False
            
        # Ensure scheduled_start exists and is valid
        scheduled_start = game_info.get('scheduled_start')
        if not scheduled_start:
            game_info['scheduled_start'] = '9999-12-31T23:59:59'  # Default value
            
        return True
        
    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return False


# 3. Add Safe Access Helper Function
def safe_get(dictionary, *keys, default=None):
    """Safely get nested dictionary values"""
    try:
        result = dictionary
        for key in keys:
            if not isinstance(result, dict):
                return default
            result = result.get(key, default)
            if result is None:
                return default
        return result
    except Exception:
        return default

# 4. Update Display Predictions Function
def display_predictions(predictions, key=None):
    """Display predictions with enhanced statistics."""
    try:
        # Ensure predictions is not None
        if not predictions:
            st.warning("No predictions available. Please update predictions.")
            return
            
        # Sort predictions safely
        sorted_predictions = sort_predictions(predictions)
        
        # Filter valid predictions
        valid_predictions = [p for p in sorted_predictions if validate_prediction_structure(p)]
        
        if not valid_predictions:
            st.warning("No valid predictions available. Please update predictions.")
            return
        
        live_games = [p for p in valid_predictions if p.get('is_live', False)]
        scheduled_games = [p for p in valid_predictions if not p.get('is_live', False)]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric("Live Games", len(live_games)), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric("Scheduled Games", len(scheduled_games)), unsafe_allow_html=True)
        with col3:
            high_confidence = sum(1 for p in scheduled_games 
                                if safe_get(p, 'prediction', 'confidence_level') == 'High')
            st.markdown(create_metric("High Confidence", high_confidence), unsafe_allow_html=True)
        with col4:
            last_update = datetime.fromtimestamp(
                getattr(st.session_state, 'last_prediction_time', time.time())
            ).strftime("%H:%M:%S")
            st.markdown(create_metric("Last Update", last_update), unsafe_allow_html=True)
        
        # Display games
        if live_games:
            st.markdown("## 🔴 Live Games")
            for i, game in enumerate(live_games):
                display_live_game_card(game, key_prefix=f"live_{key}_{i}")
        
        if scheduled_games:
            st.markdown("## 📅 Scheduled Games")
            for i, game in enumerate(scheduled_games):
                display_scheduled_game_card(game, key_prefix=f"scheduled_{key}_{i}")
                
    except Exception as e:
        logging.error(f"Error displaying predictions: {str(e)}")
        st.error("An error occurred while displaying predictions.")

def show_prediction_status():
    """Show prediction service status"""
    if st.session_state.is_predicting:
        st.sidebar.warning("⏳ Prediction service is running...")
    else:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time)
        st.sidebar.info(f"✅ Last prediction: {last_update.strftime('%H:%M:%S')}")

def cleanup():
    """Clean up resources when the app stops"""
    if hasattr(st.session_state, 'auto_refresh_thread') and st.session_state.auto_refresh_thread:
        st.session_state.auto_refresh_thread = None
    clean_old_predictions()

atexit.register(cleanup)

def initialize_session_state():
    """Initialize all session state variables"""
    current_time = time.time()
    
    # Authentication state
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    
    # Add reward system
    if 'reward_system' not in st.session_state:
        st.session_state.reward_system = RewardSystemManager()
    
    # Existing session state initialization
    if 'start_time' not in st.session_state:
        st.session_state.start_time = current_time
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = current_time
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    if 'iterations' not in st.session_state:
        st.session_state.iterations = 0

def show_auto_refresh_status():
    """Show auto-refresh status in sidebar"""
    if st.session_state.auto_refresh_thread is not None:
        st.sidebar.success("🔄 Auto-refresh is active")
        next_update = datetime.fromtimestamp(st.session_state.last_prediction_time + 300)
        st.sidebar.info(f"Next update at: {next_update.strftime('%H:%M:%S')}")

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def log_update_cycle():
    """Log update cycle information"""
    current_time = time.time()
    time_since_last_update = current_time - st.session_state.last_update_time
    
    logging.info(f"""
    Update Cycle Status:
    - Current Time: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}
    - Last Update: {datetime.fromtimestamp(st.session_state.last_update_time).strftime('%H:%M:%S')}
    - Time Since Last Update: {time_since_last_update:.1f} seconds
    - Auto-Update Enabled: {st.session_state.get('auto_update_enabled', False)}
    - Is Predicting: {st.session_state.get('is_predicting', False)}
    """)

def create_timer():
    """Create a hidden timer component that triggers updates"""
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
        st.session_state.iterations = 0

    # Update every second
    placeholder = st.empty()
    current_time = time.time()
    elapsed = int(current_time - st.session_state.start_time)
    
    if elapsed >= 300:  # 5 minutes
        st.session_state.start_time = current_time
        st.session_state.iterations += 1
        return True
    
    return False

# 7. Update Main Function
def main():
    """Main application function with simplified authentication"""
    try:
        # Initialize session state
        initialize_session_state()

        # Simple hardcoded credentials (for demonstration)
        CREDENTIALS = {
            "matchday": {
                "password": "matchday123",
                "name": "matchday"
            }
        }

        # Check if user is already authenticated
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            st.session_state.current_user = None

        # If not authenticated, show login form
        if not st.session_state.authenticated:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.markdown("""
                    <div class="auth-container">
                        <div class="auth-header">
                            <h1>🏀 NBA Predictions</h1>
                            <h3>Login</h3>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.form("login_form"):
                    username = st.text_input("Username", key="username")
                    password = st.text_input("Password", type="password", key="password")
                    submit = st.form_submit_button("Login")
                    
                    if submit:
                        if username in CREDENTIALS and password == CREDENTIALS[username]["password"]:
                            st.session_state.authenticated = True
                            st.session_state.current_user = CREDENTIALS[username]["name"]
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                return

        # Main dashboard - only shown when authenticated
        with st.sidebar:
            st.title(f"Welcome {st.session_state.current_user}! 👋")
            
            if st.button("Logout", key="logout", type="primary"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()
            
            st.divider()

            # Dashboard controls
            st.title("Controls")
            auto_update = st.checkbox("Enable Auto-Update", value=True)
            
            if st.button("Update Now", type="primary"):
                if should_update_predictions():
                    with st.spinner("Updating predictions..."):
                        if run_predictions_with_retry():
                            st.success("Update completed!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Update failed. Please try again.")
                else:
                    st.warning("Please wait before next update.")

        # Main content area
        st.title("🏀 NBA Game Predictions Dashboard")
        
        # Auto-update logic
        if auto_update and should_update_predictions():
            with st.spinner("Running scheduled update..."):
                if run_predictions_with_retry():
                    st.rerun()

        # Load and display predictions
        predictions = load_predictions()
        if predictions:
            display_predictions(predictions, key=st.session_state.update_counter)
            show_update_status()
        else:
            st.info("No predictions available. Please update predictions.")

        # Rerun for countdown update if auto-update is enabled
        if auto_update and heartbeat():
            st.rerun()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again.")
        return

def show_update_status():
    """Show update status with accurate countdown"""
    if 'start_time' not in st.session_state:
        return
        
    current_time = time.time()
    elapsed = current_time - st.session_state.start_time
    time_remaining = max(300 - elapsed, 0)
    
    minutes = int(time_remaining // 60)
    seconds = int(time_remaining % 60)
    
    st.sidebar.markdown("### Update Status")
    
    # Show last update time
    if 'last_prediction_time' in st.session_state:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time)
        st.sidebar.info(f"Last update: {last_update.strftime('%H:%M:%S')}")
    
    # Show countdown
    if time_remaining > 0:
        st.sidebar.warning(f"Next update in: {minutes:02d}:{seconds:02d}")
        
        # Add a progress bar
        progress = 1 - (time_remaining / 300)
        st.sidebar.progress(progress)
    else:
        st.sidebar.success("Update due...")

# 4. Update Session State Initialization
def initialize_session_state():
    """Initialize all session state variables"""
    current_time = time.time()
    
    # Authentication state
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    
    # Add reward system
    if 'reward_system' not in st.session_state:
        st.session_state.reward_system = RewardSystemManager()
    
    # Existing session state initialization
    if 'start_time' not in st.session_state:
        st.session_state.start_time = current_time
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = current_time
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    if 'iterations' not in st.session_state:
        st.session_state.iterations = 0

# Add a Heartbeat Function
def heartbeat():
    """Create a heartbeat to ensure continuous updates"""
    if 'heartbeat' not in st.session_state:
        st.session_state.heartbeat = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.heartbeat >= 1:
        st.session_state.heartbeat = current_time
        return True
    return False

def run_predictions_with_retry():
    """Run predictions without retry logic"""
    try:
        with st.spinner("Generating predictions..."):
            success = run_continuous_predictions(timeout_minutes=3)
            if success:
                st.success("Predictions updated successfully!")
                return True
            else:
                st.warning("No new predictions needed at this time")
                return False
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        st.error("Failed to generate predictions. Please try again later.")
        return False

# 9. Continue Statistics Display
def display_team_stats(team_info: Dict):
    """Display team statistics with charts."""
    stats = team_info['stats']['statistics'][0]
    
    # Create radar chart for key stats
    fig = go.Figure(data=go.Scatterpolar(
        r=[
            stats.get('points', 0),
            stats.get('fieldGoalsPercentage', 0),
            stats.get('threePointsPercentage', 0),
            stats.get('reboundsTotal', 0),
            stats.get('assists', 0),
            stats.get('steals', 0),
            stats.get('blocks', 0)
        ],
        theta=['Points', 'FG%', '3P%', 'Rebounds', 'Assists', 'Steals', 'Blocks'],
        fill='toself',
        name=team_info['name']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed stats table
    st.markdown("#### Detailed Statistics")
    stats_df = pd.DataFrame({
        'Metric': [
            'Points per Game',
            'Field Goal %',
            '3-Point %',
            'Free Throw %',
            'Rebounds',
            'Assists',
            'Steals',
            'Blocks',
            'Turnovers'
        ],
        'Value': [
            f"{stats.get('points', 0):.1f}",
            f"{stats.get('fieldGoalsPercentage', 0):.1f}%",
            f"{stats.get('threePointsPercentage', 0):.1f}%",
            f"{stats.get('freeThrowsPercentage', 0):.1f}%",
            f"{stats.get('reboundsTotal', 0):.1f}",
            f"{stats.get('assists', 0):.1f}",
            f"{stats.get('steals', 0):.1f}",
            f"{stats.get('blocks', 0):.1f}",
            f"{stats.get('turnovers', 0):.1f}"
        ]
    })
    st.table(stats_df)

def display_injury_report(prediction: Dict):
    """Display comprehensive injury report."""
    st.markdown("### 🏥 Injury Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{prediction['game_info']['home_team']['name']} Injuries")
        if prediction['game_info']['home_team']['injuries']:
            injury_df = pd.DataFrame(prediction['game_info']['home_team']['injuries'])
            st.dataframe(injury_df, hide_index=True)
        else:
            st.info("No reported injuries")
            
    with col2:
        st.subheader(f"{prediction['game_info']['away_team']['name']} Injuries")
        if prediction['game_info']['away_team']['injuries']:
            injury_df = pd.DataFrame(prediction['game_info']['away_team']['injuries'])
            st.dataframe(injury_df, hide_index=True)
        else:
            st.info("No reported injuries")

def display_model_analysis(prediction: Dict):
    """Display detailed model analysis and predictions."""
    st.markdown("### 🤖 Model Analysis")
    
    # Model predictions chart
    model_data = []
    for model, pred in prediction['model_predictions'].items():
        model_data.append({
            'Model': model.upper(),
            'Confidence': pred * 100
        })
    
    df = pd.DataFrame(model_data)
    
    fig = px.bar(
        df,
        x='Model',
        y='Confidence',
        title='Model Confidence Levels',
        color='Confidence',
        color_continuous_scale='RdYlBu'
    )
    
    fig.update_layout(
        yaxis_title='Confidence (%)',
        xaxis_title='Model'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model agreement analysis
    st.markdown("#### Model Agreement Analysis")
    agreement_score = calculate_model_agreement(prediction['model_predictions'])
    
    agreement_color = (
        'green' if agreement_score > 0.8 else
        'orange' if agreement_score > 0.6 else
        'red'
    )
    
    st.markdown(
        f"Model Agreement Score: "
        f"<span style='color:{agreement_color}'>{agreement_score:.2%}</span>",
        unsafe_allow_html=True
    )

def display_context_factors(prediction: Dict):
    """Display context factors affecting the prediction."""
    st.markdown("### ���� Context Factors")
    
    factors = prediction['context_factors']
    
    # Create gauge charts for each factor
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_gauge_chart(
            "Injury Impact",
            factors['injury_impact'],
            "Impact of injuries on prediction"
        )
    
    with col2:
        create_gauge_chart(
            "Conference Factor",
            factors['conference_factor'],
            "Impact of conference strength"
        )
    
    with col3:
        create_gauge_chart(
            "Division Factor",
            factors['division_factor'],
            "Impact of division rivalry"
        )
    
    # Display factor explanations
    st.markdown("#### Factor Explanations")
    st.markdown("""
    - **Injury Impact**: Shows how team injuries affect the prediction
    - **Conference Factor**: Reflects the relative strength of conferences
    - **Division Factor**: Accounts for division rivalry effects
    """)

def create_gauge_chart(title: str, value: float, description: str):
    """Create a gauge chart for factor visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=abs(value) * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ]
        }
    ))
    
    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(description)

def calculate_model_agreement(model_predictions: Dict) -> float:
    """Calculate agreement score between models."""
    predictions = list(model_predictions.values())
    mean_pred = sum(predictions) / len(predictions)
    max_deviation = max(abs(p - mean_pred) for p in predictions)
    return 1 - max_deviation

# 3. Add Safe Comparison Function
def safe_compare_scheduled_start(pred1, pred2):
    """Safely compare two predictions by scheduled start time"""
    try:
        start1 = pred1.get('game_info', {}).get('scheduled_start', '9999-12-31T23:59:59')
        start2 = pred2.get('game_info', {}).get('scheduled_start', '9999-12-31T23:59:59')
        
        # If either value is None, use the default maximum date
        if start1 is None:
            start1 = '9999-12-31T23:59:59'
        if start2 is None:
            start2 = '9999-12-31T23:59:59'
            
        return start1 < start2
        
    except Exception as e:
        logging.error(f"Comparison error: {str(e)}")
        return False

# 4. Update Sort Logic
def sort_predictions(predictions):
    """Sort predictions safely by scheduled start time"""
    try:
        if not predictions:
            return []
            
        # Use custom comparison function
        return sorted(
            predictions,
            key=lambda x: x.get('game_info', {}).get('scheduled_start', '9999-12-31T23:59:59')
        )
        
    except Exception as e:
        logging.error(f"Sort error: {str(e)}")
        return predictions  # Return unsorted on error

# 10. Update Main App
def display_predictions(predictions, key=None):
    """Display predictions with enhanced statistics."""
    # Filter out invalid predictions
    valid_predictions = []
    for p in predictions:
        if validate_prediction_structure(p):
            valid_predictions.append(p)
        else:
            logging.warning(f"Invalid prediction structure for game {p.get('game_info', {}).get('id', 'unknown')}")
    
    live_games = [p for p in valid_predictions if p.get('is_live', False)]
    scheduled_games = [p for p in valid_predictions if not p.get('is_live', False)]
    
    # Display metrics using custom containers
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric("Live Games", len(live_games)), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric("Scheduled Games", len(scheduled_games)), unsafe_allow_html=True)
    with col3:
        high_confidence = sum(1 for p in scheduled_games 
                            if p['prediction'].get('confidence_level') == 'High')
        st.markdown(create_metric("High Confidence", high_confidence), unsafe_allow_html=True)
    with col4:
        last_update = datetime.fromtimestamp(st.session_state.last_prediction_time).strftime("%H:%M:%S")
        st.markdown(create_metric("Last Update", last_update), unsafe_allow_html=True)
    
    # Display games
    if live_games:
        st.markdown("## 🔴 Live Games")
        for i, game in enumerate(live_games):
            display_live_game_card(game, key_prefix=f"live_{key}_{i}")
    
    if scheduled_games:
        st.markdown("## 📅 Scheduled Games")
        for i, game in enumerate(scheduled_games):
            display_scheduled_game_card(game, key_prefix=f"scheduled_{key}_{i}")

# 3. Add Safe Team Name Extraction Function
def extract_team_name(team_info):
    """Safely extract team name from team information"""
    try:
        if isinstance(team_info, dict):
            return str(team_info.get('name', 'Unknown Team'))
        return str(team_info)
    except Exception:
        return 'Unknown Team'

# 6. Entry point
if __name__ == "__main__":
    initialize_session_state()
    main()
