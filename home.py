import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="NBA Game Predictions", page_icon="🏀", layout="wide")

from datetime import datetime, timezone, timedelta
import pytz
from supabase import Client
from session_state import SessionState
from api_client import EnhancedNBAApiClient
import logging
import random
import uuid
import json
import time
from nba_predictor import NBAPredictor

# Initialize session state
SessionState.init_state()

# Get Supabase client from session state
try:
    supabase = SessionState.get('supabase_client')
    if not supabase:
        SUPABASE_URL = "https://jdvxisvtktunywgdtxvz.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impkdnhpc3Z0a3R1bnl3Z2R0eHZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTE2MDAsImV4cCI6MjA1NTk2NzYwMH0.-Hdbq82ctFUCGjXkmzRDOUzlXkHjVZfp5ws4vpIFmi4"
        
        # Initialize Supabase client with minimal configuration
        from supabase.client import Client
        supabase = Client(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY
        )
        SessionState.set('supabase_client', supabase)
except Exception as e:
    st.error(f"Error initializing Supabase client: {str(e)}")
    supabase = None

# Initialize NBA API client with your API key
nba_client = EnhancedNBAApiClient(api_key="918ef216c6msh607da23f482096fp198faajsnc648d53dadc5")  # Replace with your actual API key

# Initialize predictor with models path
predictor = NBAPredictor(models_path="models")

def apply_custom_styles():
    """Apply custom CSS styling to the app."""
    st.markdown("""
        <style>
        .prediction-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .team-name {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .score-range {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        
        .vs-text {
            font-size: 16px;
            font-weight: bold;
            color: #95a5a6;
            margin: 10px 0;
            text-align: center;
        }
        
        .game-time {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .prediction-result {
            background-color: #e8f4f8;
            padding: 10px;
            border-radius: 5px;
            margin-top: 15px;
        }
        
        .winner {
            font-size: 16px;
            font-weight: bold;
            color: #2980b9;
            margin-bottom: 5px;
        }
        
        .probability {
            font-size: 14px;
            color: #34495e;
        }
        
        .date-filters {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stButton>button {
            width: 100%;
            background-color: #2980b9;
            color: white;
        }
        
        .stButton>button:hover {
            background-color: #2471a3;
        }
        </style>
    """, unsafe_allow_html=True)

def get_team_stats(team_name):
    """Get team statistics from NBA API with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            stats = nba_client.get_team_stats(team_name)
            if stats:
                logging.info(f"Successfully fetched stats for {team_name}")
                return stats
            else:
                logging.warning(f"No stats found for {team_name}, attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logging.error(f"Error fetching stats for {team_name}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None

def generate_prediction(game):
    """Generate prediction with proper error handling and logging."""
    try:
        home_team = game['teams']['home']
        away_team = game['teams']['away']
        
        # Get team statistics with retries
        home_stats = get_team_stats(home_team['name'])
        away_stats = get_team_stats(away_team['name'])
        
        if not home_stats or not away_stats:
            logging.error(f"Missing stats for {home_team['name']} or {away_team['name']}")
            return None
            
        # Log the stats we're using
        logging.info(f"Home team ({home_team['name']}) stats: {json.dumps(home_stats, indent=2)}")
        logging.info(f"Away team ({away_team['name']}) stats: {json.dumps(away_stats, indent=2)}")
        
        # Prepare features for prediction
        features = predictor.prepare_features(home_stats, away_stats)
        
        # Get prediction from ML models
        winner, probability = predictor.predict_game(features)
        
        # Calculate score ranges
        home_score_range = predictor.predict_score_range(home_stats, away_stats, is_home=True)
        away_score_range = predictor.predict_score_range(away_stats, home_stats, is_home=False)
        
        return {
            'id': str(uuid.uuid4()),
            'home_team': home_team['name'],
            'away_team': away_team['name'],
            'predicted_winner': winner,
            'win_probability': round(probability * 100, 1),
            'scheduled_start': game['date']['start'],
            'home_score_min': home_score_range[0],
            'home_score_max': home_score_range[1],
            'away_score_min': away_score_range[0],
            'away_score_max': away_score_range[1],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error generating prediction: {str(e)}", exc_info=True)
        return None

def fetch_and_save_games():
    """Fetch latest games and save predictions to Supabase"""
    try:
        # Set date range for fetching games (2 days ago to 2 weeks ahead)
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=2)
        end_date = now + timedelta(weeks=2)
        
        # Get upcoming games
        games = nba_client.get_upcoming_games(start_date=start_date, end_date=end_date)
        if not games:
            st.warning("No upcoming games found.")
            return []
        
        # Clean up old predictions (before today)
        try:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            supabase.table('predictions').delete().lt('scheduled_start', today_start).execute()
            logging.info("Cleaned up predictions from previous days")
        except Exception as e:
            logging.error(f"Error cleaning up old predictions: {str(e)}")
        
        # Get existing predictions for today and tomorrow
        tomorrow_end = (now + timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        existing = (
            supabase.table('predictions')
            .select('*')
            .gte('scheduled_start', today_start)
            .lt('scheduled_start', tomorrow_end)
            .execute()
        )
        
        # Create a set of existing game keys
        existing_games = {
            f"{pred['home_team']}_{pred['away_team']}_{pred['scheduled_start']}"
            for pred in (existing.data or [])
        }
        
        # Process each game
        predictions = existing.data or []
        new_predictions = []
        
        for game in games:
            try:
                game_key = f"{game['teams']['home']['name']}_{game['teams']['away']['name']}_{game['date']['start']}"
                
                # Skip if prediction already exists
                if game_key in existing_games:
                    logging.info(f"Using existing prediction for {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
                    continue
                
                # Generate new prediction
                prediction = generate_prediction(game)
                if not prediction:
                    logging.warning(f"Could not generate prediction for {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
                    continue
                
                # Save to Supabase
                data = {
                    'id': str(uuid.uuid4()),
                    'home_team': game['teams']['home']['name'],
                    'away_team': game['teams']['away']['name'],
                    'predicted_winner': prediction['predicted_winner'],
                    'win_probability': prediction['win_probability'],
                    'scheduled_start': game['date']['start'],
                    'home_score_min': prediction['home_score_min'],
                    'home_score_max': prediction['home_score_max'],
                    'away_score_min': prediction['away_score_min'],
                    'away_score_max': prediction['away_score_max'],
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                
                result = supabase.table('predictions').insert(data).execute()
                
                if result.data:
                    new_predictions.extend(result.data)
                    logging.info(f"Added game: {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
                
            except Exception as e:
                logging.error(f"Error processing game: {str(e)}")
                continue
        
        all_predictions = predictions + new_predictions
        logging.info(f"Total predictions: {len(all_predictions)} ({len(new_predictions)} new)")
        return all_predictions
        
    except Exception as e:
        st.error(f"Error fetching games: {str(e)}")
        logging.error(f"Error in fetch_and_save_games: {str(e)}")
        return []

def save_prediction(data):
    """Save a prediction to Supabase."""
    try:
        # Create a unique key for the game
        game_key = f"{data['home_team']}_{data['away_team']}_{data['scheduled_start']}"
        
        # Check if prediction already exists
        existing = (
            supabase.table('predictions')
            .select('*')
            .eq('home_team', data['home_team'])
            .eq('away_team', data['away_team'])
            .eq('scheduled_start', data['scheduled_start'])
            .execute()
        )
        
        if existing.data:
            # Update existing prediction
            return supabase.table('predictions').update(data).eq('id', existing.data[0]['id']).execute()
        else:
            # Insert new prediction
            return supabase.table('predictions').insert(data).execute()
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")
        return None

def load_predictions(start_date=None, end_date=None, include_live=False):
    """Load predictions from Supabase database."""
    try:
        # Start with base query
        query = supabase.table('predictions').select('*')
        
        # Add date filters if provided
        if start_date:
            query = query.gte('scheduled_start', start_date.isoformat())
        if end_date:
            query = query.lte('scheduled_start', end_date.isoformat())
            
        # Execute query
        result = query.execute()
        predictions = result.data if result else []
        
        if not predictions:
            return []
            
        # Sort predictions by scheduled start time
        predictions.sort(key=lambda x: x['scheduled_start'])
        
        # Update with live scores if requested
        if include_live:
            try:
                live_games = nba_client.get_live_games()
                if live_games:
                    for pred in predictions:
                        game_key = f"{pred['home_team']}_{pred['away_team']}"
                        for live in live_games:
                            live_key = f"{live['teams']['home']['name']}_{live['teams']['away']['name']}"
                            if game_key == live_key:
                                pred['live_scores'] = {
                                    'home': live['scores']['home']['points'],
                                    'away': live['scores']['away']['points'],
                                    'status': live['status']
                                }
            except Exception as e:
                logging.error(f"Error fetching live scores: {str(e)}")
                
        return predictions
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        logging.error(f"Error in load_predictions: {str(e)}")
        return []

def clean_duplicate_predictions():
    """Remove duplicate predictions from the database."""
    try:
        # Get all predictions
        result = supabase.table('predictions').select('*').execute()
        if not result.data:
            return
            
        # Track unique games and their first occurrence
        seen_games = {}
        duplicates = []
        
        for pred in result.data:
            game_key = f"{pred['home_team']}_{pred['away_team']}_{pred['scheduled_start']}"
            if game_key in seen_games:
                # This is a duplicate
                duplicates.append(pred['id'])
            else:
                seen_games[game_key] = pred['id']
        
        # Delete all duplicates
        if duplicates:
            supabase.table('predictions').delete().in_('id', duplicates).execute()
            logging.info(f"Cleaned up {len(duplicates)} duplicate predictions")
            
    except Exception as e:
        logging.error(f"Error cleaning duplicate predictions: {str(e)}")

def refresh_predictions():
    """Refresh predictions by updating only missing or outdated ones"""
    try:
        # Clean up any existing duplicates first
        clean_duplicate_predictions()
        
        # Set date range for fetching games (2 days ago to 2 weeks ahead)
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=2)
        end_date = now + timedelta(weeks=2)
        
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        tomorrow_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Get existing predictions for today
        existing_predictions = (
            supabase.table('predictions')
            .select('*')
            .gte('scheduled_start', today_start)
            .lt('scheduled_start', tomorrow_start)
            .execute()
        )
        logging.info(f"Loaded {len(existing_predictions.data) if existing_predictions.data else 0} predictions for today")
        
        # Create a set of existing game keys
        existing_games = {
            f"{pred['home_team']}_{pred['away_team']}_{pred['scheduled_start']}"
            for pred in (existing_predictions.data or [])
        }
        
        # Fetch new games and generate predictions only for missing games
        games = nba_client.get_upcoming_games(start_date=start_date, end_date=end_date)
        new_predictions = []
        
        if games:
            for game in games:
                game_key = f"{game['teams']['home']['name']}_{game['teams']['away']['name']}_{game['date']['start']}"
                
                # Skip if prediction already exists
                if game_key in existing_games:
                    logging.info(f"Prediction already exists for {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
                    continue
                
                prediction = generate_prediction(game)
                if prediction:
                    result = save_prediction(prediction)
                    if result and result.data:
                        new_predictions.extend(result.data)
                        logging.info(f"Added game: {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
        
        # Load all predictions again to ensure we have the latest data
        latest_predictions = (
            supabase.table('predictions')
            .select('*')
            .gte('scheduled_start', today_start)
            .lt('scheduled_start', tomorrow_start)
            .execute()
        )
        
        return latest_predictions.data or []
        
    except Exception as e:
        logging.error(f"Error refreshing predictions: {str(e)}")
        st.error(f"Error refreshing predictions: {str(e)}")
        return []

def display_game_card(prediction):
    """Display a game prediction card."""
    try:
        # Convert UTC time to IST for display
        game_time = datetime.fromisoformat(prediction['scheduled_start'])
        ist_time = game_time.astimezone(pytz.timezone('Asia/Kolkata'))
        
        # Format score ranges and time
        home_score = f"{prediction['home_score_min']}-{prediction['home_score_max']}"
        away_score = f"{prediction['away_score_min']}-{prediction['away_score_max']}"
        game_time_str = ist_time.strftime('%Y-%m-%d %H:%M IST')
        win_prob = f"{prediction['win_probability']}%"
        
        # Create a container for the prediction card
        with st.container():
            st.markdown("---")  # Divider
            
            # Home team
            st.markdown(f"### {prediction['home_team']}")
            st.write(f"Score Range: {home_score}")
            
            # Game info
            st.markdown("### vs")
            st.write(f"Game Time: {game_time_str}")
            
            # Away team
            st.markdown(f"### {prediction['away_team']}")
            st.write(f"Score Range: {away_score}")
            
            # Prediction result
            st.info(f"Predicted Winner: {prediction['predicted_winner']}\nWin Probability: {win_prob}")
            
    except Exception as e:
        st.error(f"Error displaying prediction: {str(e)}")
        logging.error(f"Error in display_game_card: {str(e)}")

def create_navbar():
    """Create a navigation bar at the top of the page."""
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            .navbar {
                background-color: #f8f9fa;
                padding: 1rem;
                margin-bottom: 2rem;
                border-bottom: 1px solid #e1e4e8;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .stButton > button {
                background-color: transparent;
                border: 1px solid #ddd;
                padding: 0.5rem 1rem;
                margin: 0 0.5rem;
                border-radius: 5px;
                color: #444;
                font-weight: 500;
            }
            .stButton > button:hover {
                background-color: #e9ecef;
                border-color: #ccc;
                color: #000;
            }
            /* Container for buttons */
            div[data-testid="column"] {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 1rem;
            }
            /* Make buttons appear in a row */
            div[data-testid="column"] > div {
                display: inline-block;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Create a container for the navbar
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    
    # Use a single column for centered buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            if st.button("📊 History", use_container_width=True):
                st.switch_page("pages/01_History.py")
        with col_b:
            if st.button("🔄 Refresh", use_container_width=True):
                refresh_predictions()
                st.rerun()
        with col_c:
            if st.button("🚪 Logout", use_container_width=True):
                SessionState.set('authenticated', False)
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    try:
        if not SessionState.get('authenticated'):
            show_login_page()
            return

        # Create navbar
        create_navbar()
        
        # Add date filters in a container
        with st.container():
            st.title("🏀 NBA Game Predictions")
            col1, col2 = st.columns([2, 2])
            
            # Get today's date in UTC
            now = datetime.now(timezone.utc)
            today = now.date()
            
            # Date inputs (convert to UTC for consistency)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=today,
                    min_value=today - timedelta(days=7),
                    max_value=today + timedelta(weeks=2)
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=today,
                    min_value=start_date,
                    max_value=today + timedelta(weeks=2)
                )

        # Convert dates to UTC datetime
        start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

        # Load predictions
        with st.spinner("Loading predictions..."):
            predictions = load_predictions(start_datetime, end_datetime)

        if not predictions:
            st.warning("No predictions available for the selected date range.")
            return

        # Group predictions by date
        predictions_by_date = {}
        for pred in predictions:
            game_date = datetime.fromisoformat(pred['scheduled_start']).strftime('%Y-%m-%d')
            if game_date not in predictions_by_date:
                predictions_by_date[game_date] = []
            predictions_by_date[game_date].append(pred)

        # Display predictions grouped by date
        for date in sorted(predictions_by_date.keys()):
            st.subheader(f"Games on {date}")
            
            # Remove duplicates based on teams and start time
            seen = set()
            unique_predictions = []
            for pred in predictions_by_date[date]:
                key = (pred['home_team'], pred['away_team'], pred['scheduled_start'])
                if key not in seen:
                    seen.add(key)
                    unique_predictions.append(pred)
            
            # Display predictions in a single column
            for prediction in unique_predictions:
                display_game_card(prediction)

    except Exception as e:
        st.error(f"Error in main: {str(e)}")
        logging.error(f"Error in main: {str(e)}")

def show_login_page():
    """Show the login page with username and password fields."""
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏀 NBA Predictions Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Login"):
            if username == "match_wizard" and password == "GoalMaster":
                SessionState.set('authenticated', True)
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def delete_all_predictions():
    """Delete all predictions from Supabase"""
    try:
        # Get today's date in UTC
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Delete predictions for today and older
        result = (
            supabase.table('predictions')
            .delete()
            .lte('scheduled_start', today_start)  # Delete predictions for today and older
            .execute()
        )
        
        logging.info("Successfully deleted old predictions")
        
        # Fetch new predictions
        fetch_and_save_games()
        
    except Exception as e:
        st.error(f"Error deleting predictions: {str(e)}")
        logging.error(f"Delete error: {str(e)}")

if __name__ == "__main__":
    main()
