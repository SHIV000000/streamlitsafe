import streamlit as st
import logging
import json
from datetime import datetime, timezone, timedelta
import uuid
from typing import Dict, List, Optional, Tuple
import requests
from nba_api_client import NBAGameResultsFetcher
from prediction_service import NBAPredictor
from nba_stats import NBA_TEAM_STATS
from session_state import init_session_state, is_logged_in, login, logout, get_username
from supabase import Client, create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Initialize clients
nba_client = NBAGameResultsFetcher()
predictor = NBAPredictor()

# Supabase configuration
SUPABASE_URL = "https://jdvxisvtktunywgdtxvz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impkdnhpc3Z0a3R1bnl3Z2R0eHZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTE2MDAsImV4cCI6MjA1NTk2NzYwMH0.-Hdbq82ctFUCGjXkmzRDOUzlXkHjVZfp5ws4vpIFmi4"

def init_supabase():
    """Initialize Supabase client."""
    if 'supabase' not in st.session_state:
        st.session_state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return st.session_state.supabase

def get_todays_matches():
    """Get today's matches from NBA API."""
    try:
        url = "https://api-nba-v1.p.rapidapi.com/games"
        querystring = {"date": datetime.now().strftime('%Y-%m-%d')}
        headers = {
            "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
            "x-rapidapi-key": "918ef216c6msh607da23f482096fp198faajsnc648d53dadc5"
        }
        response = requests.get(url, headers=headers, params=querystring)
        logging.info(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            
            for game in data.get('response', []):
                if 'teams' in game and 'date' in game:
                    matches.append({
                        'teams': {
                            'home': {'name': game['teams']['home']['name']},
                            'away': {'name': game['teams']['visitors']['name']}
                        },
                        'date': {
                            'start': game['date']['start']
                        }
                    })
            
            if not matches:
                # If no matches found, use some sample data for testing
                matches = [
                    {
                        'teams': {
                            'home': {'name': 'Los Angeles Lakers'},
                            'away': {'name': 'Golden State Warriors'}
                        },
                        'date': {
                            'start': datetime.now().isoformat()
                        }
                    },
                    {
                        'teams': {
                            'home': {'name': 'Boston Celtics'},
                            'away': {'name': 'Miami Heat'}
                        },
                        'date': {
                            'start': datetime.now().isoformat()
                        }
                    }
                ]
                logging.info("Using sample matches for testing")
            else:
                logging.info(f"Found {len(matches)} matches from API")
            
            return matches
        else:
            logging.error(f"API request failed: {response.status_code}")
            return []
            
    except Exception as e:
        logging.error(f"Error fetching matches: {str(e)}", exc_info=True)
        return []

def save_prediction(prediction: Dict):
    """Save prediction to Supabase."""
    try:
        supabase = init_supabase()
        
        # Log the prediction object for debugging
        logging.info(f"Saving prediction: {json.dumps(prediction, default=str)}")
        
        # Check for existing prediction with the same game details
        existing = (
            supabase.table('predictions')
            .select('*')
            .eq('home_team', prediction['home_team'])
            .eq('away_team', prediction['away_team'])
            .eq('scheduled_start', prediction['scheduled_start'])
            .execute()
        )
        
        # Filter prediction to only include fields that exist in the database schema
        # Known valid fields based on the database schema
        valid_fields = [
            'id', 'created_at', 'home_team', 'away_team', 'predicted_winner',
            'win_probability', 'home_score_min', 'home_score_max',
            'away_score_min', 'away_score_max', 'scheduled_start'
        ]
        
        filtered_prediction = {k: v for k, v in prediction.items() if k in valid_fields}
        
        if existing.data:
            # Check if we need to update (if prediction has changed)
            existing_pred = existing.data[0]
            needs_update = False
            
            # Compare key fields to see if prediction has changed
            for key in ['predicted_winner', 'win_probability', 'home_score_min', 
                       'home_score_max', 'away_score_min', 'away_score_max']:
                if key in filtered_prediction and str(filtered_prediction[key]) != str(existing_pred.get(key, '')):
                    needs_update = True
                    logging.info(f"Prediction changed for {key}: {existing_pred.get(key, '')} -> {filtered_prediction[key]}")
                    break
            
            if needs_update:
                # Update existing prediction
                logging.info(f"Updating existing prediction for {prediction['home_team']} vs {prediction['away_team']}")
                result = (
                    supabase.table('predictions')
                    .update(filtered_prediction)
                    .eq('id', existing_pred['id'])
                    .execute()
                )
                return result.data if result else None
            else:
                logging.info(f"Prediction already exists and hasn't changed for {prediction['home_team']} vs {prediction['away_team']}")
                return existing_pred
        else:
            # Insert new prediction
            logging.info(f"Creating new prediction for {prediction['home_team']} vs {prediction['away_team']}")
            
            # Ensure we have an id field instead of prediction_id
            if 'prediction_id' in filtered_prediction:
                del filtered_prediction['prediction_id']  # Remove if it exists
                
            filtered_prediction['id'] = str(uuid.uuid4())
            result = supabase.table('predictions').insert(filtered_prediction).execute()
            
            return result.data if result else None
        
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")
        return None

def show_login():
    """Show login page."""
    st.title("🔒 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == "match_wizard" and password == "GoalMaster":
                login(username)
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def show_navigation():
    """Show navigation bar."""
    st.markdown("""
        <style>
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background-color: #f8f9fa;
            margin-bottom: 2rem;
            border-radius: 5px;
        }
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        .nav-link {
            color: #1f77b4;
            font-weight: bold;
            cursor: pointer;
        }
        .nav-link:hover {
            color: #0056b3;
        }
        .logout-btn {
            color: #dc3545;
            cursor: pointer;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2,2,1])
    
    with col1:
        st.markdown(f"👤 Welcome, {get_username()}")
    with col2:
        if st.button("📊 History"):
            st.switch_page("pages/01_History.py")
    with col3:
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

def show_predictions():
    """Show predictions page."""
    show_navigation()
    st.title("🏀 NBA Game Predictions")
    
    # Remove sidebar
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading matches..."):
        matches = get_todays_matches()
        
    if not matches:
        st.warning("No matches found for today.")
        return
        
    # Display each match prediction
    for match in matches:
        prediction = generate_prediction(match)
        if prediction:
            # Add created_at timestamp
            prediction['created_at'] = datetime.now(timezone.utc).isoformat()
            
            # Save to database - only include fields that exist in the database schema
            # Do not include username as it's not in the schema
            saved_prediction = save_prediction(prediction)
            
            if saved_prediction:
                logging.info(f"Saved prediction for {prediction['home_team']} vs {prediction['away_team']}")
            else:
                logging.warning(f"Failed to save prediction for {prediction['home_team']} vs {prediction['away_team']}")
            
            with st.container():
                # Create columns for better layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Teams and scores
                    st.markdown(f"### {prediction['home_team']}")
                    st.write(f"Score Range: {prediction['home_score_min']}-{prediction['home_score_max']}")
                    
                    st.write("vs")
                    
                    st.markdown(f"### {prediction['away_team']}")
                    st.write(f"Score Range: {prediction['away_score_min']}-{prediction['away_score_max']}")
                
                with col2:
                    # Game time and prediction
                    game_time = format_game_time(prediction['scheduled_start'])
                    st.write(f"Game Time (German):")
                    st.write(game_time)
                    
                    winner = prediction['predicted_winner']
                    prob = prediction['win_probability']
                    
                    st.markdown(f"**Predicted Winner:**")
                    st.markdown(f"**{winner}**")
                    st.markdown(f"**Win Probability:**")
                    st.markdown(f"**{prob:.1f}%**")
                
                st.markdown("---")

def get_upcoming_games():
    """Get upcoming games."""
    try:
        return nba_client.get_upcoming_games()
    except Exception as e:
        logging.error(f"Error getting upcoming games: {str(e)}", exc_info=True)
        return []

def generate_prediction(game: Dict) -> Optional[Dict]:
    """Generate prediction for a game."""
    try:
        logging.info(f"Generating prediction for {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
        
        # Get team stats
        home_team = game['teams']['home']['name']
        away_team = game['teams']['away']['name']
        
        # Get team stats from the stats dictionary
        home_stats = NBA_TEAM_STATS.get(home_team, {})
        away_stats = NBA_TEAM_STATS.get(away_team, {})
        
        if not home_stats or not away_stats:
            logging.warning(f"Missing stats for {home_team} or {away_team}")
            return None
            
        # Add team names to stats
        home_stats['team_name'] = home_team
        away_stats['team_name'] = away_team
        
        # Make prediction
        predictor = NBAPredictor()
        winner_side, win_probability = predictor.predict_game(home_stats, away_stats)
        
        # Convert 'home'/'away' to actual team name
        winner_team = home_team if winner_side == 'home' else away_team
        
        try:
            # Try to calculate score ranges
            home_score_range = predictor.calculate_score_range(home_stats)
            away_score_range = predictor.calculate_score_range(away_stats)
        except Exception as e:
            logging.warning(f"Error calculating score ranges: {str(e)}")
            # Use default ranges if calculation fails
            home_score_range = (100, 110)
            away_score_range = (100, 110)
        
        # Create prediction object
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': winner_team,  # Store actual team name instead of 'home'/'away'
            'win_probability': win_probability,
            'home_score_min': home_score_range[0],
            'home_score_max': home_score_range[1],
            'away_score_min': away_score_range[0],
            'away_score_max': away_score_range[1],
            'scheduled_start': game['date']['start']
            # Removed prediction_id as it's not in the database schema
        }
        
        return prediction
        
    except Exception as e:
        logging.error(f"Error generating prediction: {str(e)}", exc_info=True)
        return None

def format_game_time(utc_time: str) -> str:
    """Convert UTC time to German time."""
    try:
        # Parse UTC time
        utc_dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
        # Add 1 hour for German time (UTC+1)
        german_dt = utc_dt + timedelta(hours=1)
        # Format in German style
        return german_dt.strftime('%d.%m.%Y %H:%M')
    except Exception as e:
        logging.error(f"Error formatting time: {str(e)}")
        return utc_time

def main():
    """Main function."""
    # Initialize session state
    init_session_state()
    
    # Hide Streamlit's default menu and footer
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Show login page if not logged in
    if not is_logged_in():
        show_login()
    else:
        show_predictions()

if __name__ == "__main__":
    main()
