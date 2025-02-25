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
from session_state import SessionState
from supabase import Client

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

# Initialize Supabase client
SUPABASE_URL = "https://jdvxisvtktunywgdtxvz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impkdnhpc3Z0a3R1bnl3Z2R0eHZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTE2MDAsImV4cCI6MjA1NTk2NzYwMH0.-Hdbq82ctFUCGjXkmzRDOUzlXkHjVZfp5ws4vpIFmi4"

def init_supabase():
    """Initialize Supabase client if not already initialized."""
    if not SessionState.get('supabase_client'):
        supabase = Client(SUPABASE_URL, SUPABASE_KEY)
        SessionState.set('supabase_client', supabase)
    return SessionState.get('supabase_client')

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
        
        # Check for existing prediction
        existing = (
            supabase.table('predictions')
            .select('*')
            .eq('home_team', prediction['home_team'])
            .eq('away_team', prediction['away_team'])
            .eq('scheduled_start', prediction['scheduled_start'])
            .execute()
        )
        
        if existing.data:
            # Update existing prediction
            result = (
                supabase.table('predictions')
                .update(prediction)
                .eq('id', existing.data[0]['id'])
                .execute()
            )
        else:
            # Insert new prediction
            prediction['id'] = str(uuid.uuid4())
            result = supabase.table('predictions').insert(prediction).execute()
            
        return result.data if result else None
        
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")
        return None

def show_login():
    """Show login page."""
    st.title("🏀 NBA Game Predictions")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == "match_wizard" and password == "GoalMaster":
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def show_navigation():
    """Show navigation bar."""
    # Add custom CSS for navigation bar
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
            text-decoration: none;
            color: #1f77b4;
            font-weight: bold;
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
        
        <div class="navbar">
            <div class="nav-links">
                <a href="/" class="nav-link">Home</a>
                <a href="/History" class="nav-link">History</a>
            </div>
            <div class="logout-btn" onclick="window.location.href='/'">Logout</div>
        </div>
    """, unsafe_allow_html=True)

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
                    winner_name = prediction['home_team'] if winner == 'home' else prediction['away_team']
                    prob = prediction['win_probability']
                    
                    st.markdown(f"**Predicted Winner:**")
                    st.markdown(f"**{winner_name}**")
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
        winner, win_probability = predictor.predict_game(home_stats, away_stats)
        
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
            'predicted_winner': winner,
            'win_probability': win_probability,
            'home_score_min': home_score_range[0],
            'home_score_max': home_score_range[1],
            'away_score_min': away_score_range[0],
            'away_score_max': away_score_range[1],
            'scheduled_start': game['date']['start'],
            'prediction_id': str(uuid.uuid4())
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
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        
    # Hide Streamlit's default menu and footer
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Show login page if not logged in
    if not st.session_state.logged_in:
        show_login()
    else:
        show_predictions()
        
if __name__ == "__main__":
    main()
