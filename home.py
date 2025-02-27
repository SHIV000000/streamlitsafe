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
        # Clear any cached data to ensure fresh results
        if 'todays_matches' in st.session_state:
            del st.session_state.todays_matches
        
        url = "https://api-nba-v1.p.rapidapi.com/games"
        today = datetime.now().strftime('%Y-%m-%d')
        querystring = {"date": today}
        headers = {
            "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
            "x-rapidapi-key": "918ef216c6msh607da23f482096fp198faajsnc648d53dadc5"
        }
        
        logging.info(f"Fetching NBA games for date: {today}")
        
        try:
            # Increased timeout to 30 seconds to handle slow API responses
            response = requests.get(url, headers=headers, params=querystring, timeout=30)
            logging.info(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                matches = []
                
                # Log the raw API response for debugging
                logging.debug(f"Raw API response: {json.dumps(data)[:500]}...")
                
                for game in data.get('response', []):
                    if 'teams' in game and 'date' in game:
                        # Extract team names, ensuring we have valid data
                        home_team = game['teams']['home'].get('name', 'Unknown Home Team')
                        away_team = game['teams']['visitors'].get('name', 'Unknown Away Team')
                        
                        # Skip games with invalid team names
                        if home_team == 'Unknown Home Team' or away_team == 'Unknown Away Team':
                            logging.warning(f"Skipping game with invalid team names: {game['id']}")
                            continue
                        
                        match_data = {
                            'teams': {
                                'home': {
                                    'name': home_team,
                                    'code': game['teams']['home'].get('code', 'UNK')
                                },
                                'away': {
                                    'name': away_team,
                                    'code': game['teams']['visitors'].get('code', 'UNK')
                                }
                            },
                            'date': {
                                'start': game['date']['start']
                            },
                            'id': game['id'],
                            'status': game['status']['long']
                        }
                        
                        # Add scores if available
                        if 'scores' in game:
                            home_score = game['scores']['home'].get('points')
                            away_score = game['scores']['visitors'].get('points')
                            
                            if home_score is not None and away_score is not None:
                                match_data['scores'] = {
                                    'home': home_score,
                                    'away': away_score
                                }
                        
                        matches.append(match_data)
                        logging.info(f"Found game: {home_team} vs {away_team} (Status: {match_data['status']})")
                
                if matches:
                    logging.info(f"Found {len(matches)} matches from API")
                    
                    # Store in session state with timestamp to track freshness
                    st.session_state.todays_matches = {
                        'data': matches,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return matches
            else:
                logging.error(f"API request failed: {response.status_code}, Response: {response.text[:200]}")
        
        except requests.exceptions.Timeout:
            logging.error("API request timed out. Using fallback data.")
            st.warning("NBA API request timed out. Using sample data instead.")
        
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {str(e)}")
            st.warning("Could not connect to NBA API. Using sample data instead.")
        
        # If we get here, either the API call failed or returned no matches
        # Use sample data as fallback
        logging.warning("Using sample data as fallback")
        matches = [
            {
                'teams': {
                    'home': {'name': 'Los Angeles Lakers', 'code': 'LAL'},
                    'away': {'name': 'Golden State Warriors', 'code': 'GSW'}
                },
                'date': {
                    'start': datetime.now().isoformat()
                },
                'id': 'sample1',
                'status': 'Scheduled'
            },
            {
                'teams': {
                    'home': {'name': 'Boston Celtics', 'code': 'BOS'},
                    'away': {'name': 'Miami Heat', 'code': 'MIA'}
                },
                'date': {
                    'start': datetime.now().isoformat()
                },
                'id': 'sample2',
                'status': 'Scheduled'
            }
        ]
        logging.info("Using sample matches for testing")
        
        # Store sample data in session state
        st.session_state.todays_matches = {
            'data': matches,
            'timestamp': datetime.now().isoformat(),
            'is_sample': True
        }
        
        return matches
            
    except Exception as e:
        logging.error(f"Error fetching matches: {str(e)}", exc_info=True)
        
        # Return sample data as a last resort
        matches = [
            {
                'teams': {
                    'home': {'name': 'Los Angeles Lakers', 'code': 'LAL'},
                    'away': {'name': 'Golden State Warriors', 'code': 'GSW'}
                },
                'date': {
                    'start': datetime.now().isoformat()
                },
                'id': 'sample1',
                'status': 'Scheduled'
            }
        ]
        
        # Store sample data in session state
        st.session_state.todays_matches = {
            'data': matches,
            'timestamp': datetime.now().isoformat(),
            'is_sample': True
        }
        
        return matches

def save_prediction(prediction: Dict):
    """Save prediction to Supabase."""
    try:
        supabase = init_supabase()
        
        # Log the prediction object for debugging
        logging.info(f"Saving prediction: {json.dumps(prediction, default=str)}")
        
        # Check for existing prediction with the same game details
        try:
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
        
        except Exception as db_error:
            logging.error(f"Database operation error: {str(db_error)}", exc_info=True)
            # Continue with the app even if database operations fail
            return None
            
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}", exc_info=True)
        return None

def show_login():
    """Show login page."""
    # Load custom CSS
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Remove sidebar
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    # Create a centered login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">🏀 NBA Predictions</h1>', unsafe_allow_html=True)
    
    # Login form
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
    
    # Registration link
    st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close login-container

def show_navigation():
    """Show navigation bar."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown(f"👤 Welcome, {get_username()}")
    
    with col2:
        if st.button("📊 History"):
            st.switch_page("pages/01_History.py")
    
    with col3:
        if st.button("🔄 Refresh Data", key="refresh_btn"):
            # Clear any cached data
            if 'todays_matches' in st.session_state:
                del st.session_state.todays_matches
            st.experimental_rerun()
    
    with col4:
        if st.button("🚪 Logout"):
            logout()
            st.experimental_rerun()

def show_predictions():
    """Show predictions page."""
    show_navigation()
    st.title("🏀 NBA Game Predictions")
    
    # Load custom CSS
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Remove sidebar
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading today's NBA games..."):
        matches = get_todays_matches()
        
    if not matches:
        st.warning("No NBA games found for today.")
        return
    
    # Display timestamp of when the data was fetched
    if 'todays_matches' in st.session_state:
        if 'timestamp' in st.session_state.todays_matches:
            fetch_time = datetime.fromisoformat(st.session_state.todays_matches['timestamp'])
            st.caption(f"Last updated: {fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display notice if using sample data
        if st.session_state.todays_matches.get('is_sample', False):
            st.warning("⚠️ Using sample data because the NBA API is currently unavailable. Predictions shown are for demonstration purposes only.")
    
    # Display each match prediction
    for match in matches:
        with st.spinner(f"Generating prediction for {match['teams']['home']['name']} vs {match['teams']['away']['name']}..."):
            prediction = generate_prediction(match)
            
        if prediction:
            # Create a card-like container for each prediction
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                # Main prediction display
                st.markdown('<div class="prediction-main">', unsafe_allow_html=True)
                
                # Left team column
                st.markdown('<div class="team-column">', unsafe_allow_html=True)
                st.markdown(f'<div class="team-name">{prediction["home_team"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="score-label">Current Score</div>', unsafe_allow_html=True)
                if 'scores' in match and match['scores']['home'] is not None:
                    st.markdown(f'<div class="score">{match["scores"]["home"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="score">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="range-label">Predicted Range</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="range">({prediction["home_score_min"]}-{prediction["home_score_max"]})</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Middle column (VS and Status)
                st.markdown('<div class="middle-column">', unsafe_allow_html=True)
                st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
                if isinstance(match["status"], dict) and "long" in match["status"]:
                    status_text = match["status"]["long"]
                else:
                    status_text = str(match["status"])
                st.markdown(f'<div class="game-status">{status_text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Right team column
                st.markdown('<div class="team-column">', unsafe_allow_html=True)
                st.markdown(f'<div class="team-name">{prediction["away_team"]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="score-label">Current Score</div>', unsafe_allow_html=True)
                if 'scores' in match and match['scores']['away'] is not None:
                    st.markdown(f'<div class="score">{match["scores"]["away"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="score">-</div>', unsafe_allow_html=True)
                st.markdown('<div class="range-label">Predicted Range</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="range">({prediction["away_score_min"]}-{prediction["away_score_max"]})</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close prediction-main
                
                # Game details section - all in one line
                st.markdown(f'''
                    <div class="game-details">
                        <div class="detail-item">
                            <span class="detail-label">Game Time:</span>
                            <span class="detail-value">{format_game_time(prediction['scheduled_start'])}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Predicted Winner:</span>
                            <span class="detail-value">{prediction['predicted_winner']} ({prediction['win_probability']:.1f}%)</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Confidence:</span>
                            <span class="detail-value">
                                <span class="confidence-level {'confidence-high' if prediction['win_probability'] > 70 else 'confidence-medium' if prediction['win_probability'] >= 50 else 'confidence-low'}">
                                    {'High' if prediction['win_probability'] > 70 else 'Medium' if prediction['win_probability'] >= 50 else 'Low'}
                                </span>
                            </span>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close prediction-card
        else:
            st.warning(f"Could not generate prediction for {match['teams']['home']['name']} vs {match['teams']['away']['name']}")
    
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
        
        # If we don't have stats for these teams, try to find the closest match by name
        if not home_stats:
            for team_name, stats in NBA_TEAM_STATS.items():
                if home_team.lower() in team_name.lower() or team_name.lower() in home_team.lower():
                    home_stats = stats
                    logging.info(f"Using stats for {team_name} as a substitute for {home_team}")
                    break
        
        if not away_stats:
            for team_name, stats in NBA_TEAM_STATS.items():
                if away_team.lower() in team_name.lower() or team_name.lower() in away_team.lower():
                    away_stats = stats
                    logging.info(f"Using stats for {team_name} as a substitute for {away_team}")
                    break
        
        # If we still don't have stats, use league averages
        if not home_stats:
            logging.warning(f"No stats found for {home_team}, using league averages")
            home_stats = {
                'wins': 41,
                'losses': 41,
                'points_per_game': 110,
                'points_allowed': 110,
                'offensive_rating': 110,
                'defensive_rating': 110,
                'win_streak': 0
            }
        
        if not away_stats:
            logging.warning(f"No stats found for {away_team}, using league averages")
            away_stats = {
                'wins': 41,
                'losses': 41,
                'points_per_game': 110,
                'points_allowed': 110,
                'offensive_rating': 110,
                'defensive_rating': 110,
                'win_streak': 0
            }
            
        # Add team names to stats
        home_stats['team_name'] = home_team
        away_stats['team_name'] = away_team
        
        # Make prediction
        predictor = NBAPredictor()
        winner_side, win_probability = predictor.predict_game(home_stats, away_stats)
        
        # Convert 'home'/'away' to actual team name
        winner_team = home_team if winner_side == 'home' else away_team
        
        # Calculate score ranges with error handling
        try:
            home_score_min, home_score_max = predictor.predict_score_range(home_stats, away_stats, True)
            away_score_min, away_score_max = predictor.predict_score_range(away_stats, home_stats, False)
        except Exception as e:
            logging.error(f"Error calculating score ranges: {str(e)}", exc_info=True)
            # Use fallback score ranges
            home_score_min, home_score_max = 95, 105
            away_score_min, away_score_max = 95, 105
        
        # Log score range calculations for debugging
        logging.info(f"""
            Score Range Calculation for {home_team}:
            - Base Score: 110
            - Offensive Rating: {home_stats['offensive_rating']}
            - Offensive Factor: {home_stats['offensive_rating']/100:.2f}
            - Predicted Score: {(home_stats['offensive_rating']/100) * 110:.1f}
            - Final Range: {home_score_min}-{home_score_max}
            """)
            
        logging.info(f"""
            Score Range Calculation for {away_team}:
            - Base Score: 110
            - Offensive Rating: {away_stats['offensive_rating']}
            - Offensive Factor: {away_stats['offensive_rating']/100:.2f}
            - Predicted Score: {(away_stats['offensive_rating']/100) * 110:.1f}
            - Final Range: {away_score_min}-{away_score_max}
            """)
        
        # Create prediction object with a timestamp to ensure freshness
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': winner_team,
            'win_probability': win_probability,
            'home_score_min': home_score_min,
            'home_score_max': home_score_max,
            'away_score_min': away_score_min,
            'away_score_max': away_score_max,
            'scheduled_start': game['date']['start'],
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Add confidence level
        if win_probability > 70:
            prediction['confidence_level'] = 'High'
        elif win_probability >= 50:
            prediction['confidence_level'] = 'Medium'
        else:
            prediction['confidence_level'] = 'Low'
        
        # Save the prediction to the database with error handling
        try:
            save_result = save_prediction(prediction)
            if save_result:
                logging.info(f"Saved prediction for {home_team} vs {away_team}")
            else:
                logging.warning(f"Failed to save prediction for {home_team} vs {away_team}")
        except Exception as e:
            logging.error(f"Error saving prediction: {str(e)}", exc_info=True)
            # Continue even if saving fails
        
        return prediction
        
    except Exception as e:
        logging.error(f"Error generating prediction: {str(e)}", exc_info=True)
        return None

def format_game_time(utc_time: str) -> str:
    """Convert UTC time to German time (CET/CEST)."""
    try:
        # Parse UTC time
        if 'Z' in utc_time:
            utc_dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
        elif '+' in utc_time or '-' in utc_time:
            # Handle timestamps that already have timezone info
            utc_dt = datetime.fromisoformat(utc_time)
        else:
            # Handle timestamps without timezone info (assume UTC)
            utc_dt = datetime.fromisoformat(utc_time).replace(tzinfo=timezone.utc)
        
        # Determine if it's CET (UTC+1) or CEST (UTC+2)
        # This is a simplified approach - for production, use a proper timezone library
        now = datetime.now()
        is_summer_time = now.month > 3 and now.month < 10
        
        # Add 1 hour for CET or 2 hours for CEST
        hours_offset = 2 if is_summer_time else 1
        german_dt = utc_dt + timedelta(hours=hours_offset)
        
        # Format in German style with timezone indicator
        tz_name = "CEST" if is_summer_time else "CET"
        formatted_time = german_dt.strftime('%d.%m.%Y %H:%M')
        
        return f"{formatted_time} {tz_name}"
    except Exception as e:
        logging.error(f"Error formatting time: {str(e)}", exc_info=True)
        return utc_time

def main():
    """Main function."""
    # Initialize session state
    init_session_state()
    
    # Hide Streamlit's default menu, footer, and sidebar
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Clear any cached data at startup to ensure fresh predictions
    if 'todays_matches' in st.session_state:
        del st.session_state.todays_matches
    
    # Show login page if not logged in
    if not is_logged_in():
        show_login()
    else:
        # Show predictions page
        show_predictions()

if __name__ == "__main__":
    main()
