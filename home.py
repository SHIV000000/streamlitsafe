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

# Initialize NBA API client
NBA_API_KEY = "918ef216c6msh607da23f482096fp198faajsnc648d53dadc5"
nba_client = EnhancedNBAApiClient(NBA_API_KEY)

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

def get_team_strength(team_name):
    """Get team strength based on 2024 season performance (0.0 to 1.0 scale)"""
    # Standardize team names
    team_mapping = {
        'LA Clippers': ['Los Angeles Clippers', 'LAC', 'LA Clippers'],
        'Los Angeles Lakers': ['LA Lakers', 'LAL', 'Lakers'],
        'Brooklyn Nets': ['BKN', 'Nets'],
        'New York Knicks': ['NY Knicks', 'NYK'],
        'Philadelphia 76ers': ['PHI', 'Sixers', '76ers'],
        'Toronto Raptors': ['TOR'],
        'Chicago Bulls': ['CHI'],
        'Cleveland Cavaliers': ['CLE', 'Cavs'],
        'Detroit Pistons': ['DET'],
        'Indiana Pacers': ['IND'],
        'Milwaukee Bucks': ['MIL'],
        'Atlanta Hawks': ['ATL'],
        'Charlotte Hornets': ['CHA'],
        'Miami Heat': ['MIA'],
        'Orlando Magic': ['ORL'],
        'Washington Wizards': ['WAS'],
        'Denver Nuggets': ['DEN'],
        'Minnesota Timberwolves': ['MIN', 'Wolves'],
        'Oklahoma City Thunder': ['OKC'],
        'Portland Trail Blazers': ['POR', 'Blazers'],
        'Utah Jazz': ['UTA'],
        'Golden State Warriors': ['GSW', 'Warriors'],
        'Phoenix Suns': ['PHX'],
        'Sacramento Kings': ['SAC'],
        'Dallas Mavericks': ['DAL', 'Mavs'],
        'Houston Rockets': ['HOU'],
        'Memphis Grizzlies': ['MEM'],
        'New Orleans Pelicans': ['NOP', 'Pels'],
        'San Antonio Spurs': ['SAS'],
        'Boston Celtics': ['BOS']
    }
    
    # Find the standardized team name
    standardized_name = team_name
    for full_name, variations in team_mapping.items():
        if team_name in variations or team_name == full_name:
            standardized_name = full_name
            break
    
    # Team strength ratings (0.0 to 1.0 scale)
    strength_ratings = {
        'Boston Celtics': 0.95,
        'Denver Nuggets': 0.90,
        'Minnesota Timberwolves': 0.89,
        'LA Clippers': 0.88,
        'Oklahoma City Thunder': 0.87,
        'Milwaukee Bucks': 0.86,
        'Cleveland Cavaliers': 0.85,
        'Phoenix Suns': 0.84,
        'New York Knicks': 0.83,
        'Sacramento Kings': 0.82,
        'New Orleans Pelicans': 0.81,
        'Dallas Mavericks': 0.80,
        'Philadelphia 76ers': 0.79,
        'Miami Heat': 0.78,
        'Indiana Pacers': 0.77,
        'Los Angeles Lakers': 0.76,
        'Orlando Magic': 0.75,
        'Golden State Warriors': 0.74,
        'Houston Rockets': 0.73,
        'Chicago Bulls': 0.72,
        'Atlanta Hawks': 0.71,
        'Utah Jazz': 0.70,
        'Brooklyn Nets': 0.69,
        'Toronto Raptors': 0.68,
        'Memphis Grizzlies': 0.67,
        'Portland Trail Blazers': 0.66,
        'Washington Wizards': 0.64,
        'Charlotte Hornets': 0.62,
        'San Antonio Spurs': 0.61,
        'Detroit Pistons': 0.55
    }
    
    # Return team strength or default value
    strength = strength_ratings.get(standardized_name)
    if strength is None:
        logging.warning(f"Unknown team name: {team_name} (standardized: {standardized_name})")
        return 0.70  # Default strength for unknown teams
    return strength

def generate_prediction(game):
    try:
        home_team = game['teams']['home']
        away_team = game['teams']['away']
        
        home_strength = get_team_strength(home_team['name'])
        away_strength = get_team_strength(away_team['name'])
        
        # Calculate win probability with higher impact of strength difference
        strength_diff = home_strength - away_strength
        home_advantage = 0.04
        win_prob = 0.5 + (strength_diff * 5.0) + home_advantage
        
        # Force higher probabilities for big strength differences
        if abs(strength_diff) > 0.2:
            if strength_diff > 0:
                win_prob = max(0.75, win_prob)
            else:
                win_prob = min(0.25, win_prob)
        
        win_prob = max(0.05, min(0.95, win_prob + random.uniform(-0.02, 0.02)))
        
        predicted_winner = home_team['name'] if win_prob > 0.5 else away_team['name']
        if win_prob <= 0.5:
            win_prob = 1 - win_prob
        
        # Calculate scores based on team strengths
        league_avg = 115.0
        home_base = league_avg + ((home_strength - 0.70) * 30) + 3.5
        away_base = league_avg + ((away_strength - 0.70) * 30)
        
        # Adjust scores based on win probability
        score_adjust = (win_prob - 0.5) * 15
        if predicted_winner == home_team['name']:
            home_base += score_adjust
            away_base -= score_adjust * 0.7
        else:
            away_base += score_adjust
            home_base -= score_adjust * 0.7
        
        # Add variance
        home_var = random.uniform(8, 11) if home_strength > 0.8 else random.uniform(9, 13)
        away_var = random.uniform(8, 11) if away_strength > 0.8 else random.uniform(9, 13)
        
        return {
            'id': str(uuid.uuid4()),
            'home_team': home_team['name'],
            'away_team': away_team['name'],
            'predicted_winner': predicted_winner,
            'win_probability': round(win_prob, 3),
            'scheduled_start': game['date']['start'],
            'home_score_min': max(100, int(home_base - home_var)),
            'home_score_max': min(140, int(home_base + home_var)),
            'away_score_min': max(100, int(away_base - away_var)),
            'away_score_max': min(140, int(away_base + away_var)),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logging.error(f"Error generating prediction: {str(e)}")
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
        
        # Format score ranges
        home_score = f"{prediction['home_score_min']}-{prediction['home_score_max']}"
        away_score = f"{prediction['away_score_min']}-{prediction['away_score_max']}"
        game_time_str = ist_time.strftime('%Y-%m-%d %H:%M IST')
        win_prob = f"{prediction['win_probability']*100:.1f}"
        
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
                .game-info {
                    text-align: center;
                    margin: 15px 0;
                }
                .prediction-result {
                    background-color: #e8f4f8;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 15px;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="prediction-card">
                <div class="team-name">{prediction['home_team']}</div>
                <div class="score-range">Score Range: {home_score}</div>
                
                <div class="game-info">
                    <strong>vs</strong><br>
                    Game Time: {game_time_str}
                </div>
                
                <div class="team-name">{prediction['away_team']}</div>
                <div class="score-range">Score Range: {away_score}</div>
                
                <div class="prediction-result">
                    <strong>Predicted Winner:</strong> {prediction['predicted_winner']}<br>
                    <strong>Win Probability:</strong> {win_prob}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying prediction: {str(e)}")
        logging.error(f"Error in display_game_card: {str(e)}")

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
            st.switch_page("pages/01_History.py")
    
    with col4:
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            SessionState.clear()
            st.rerun()
    
    st.divider()

def show_login_page():
    """Show the login page with username and password fields."""
    try:
        st.markdown("""
            <style>
                .stApp {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                [data-testid="stSidebar"] {
                    display: none;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.title("🏀 NBA Predictions Login")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "match_wizard" and password == "GoalMaster":
                SessionState.authenticated = True
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
                
    except Exception as e:
        st.error(f"Error in login: {str(e)}")
        logging.error(f"Error in show_login_page: {str(e)}")

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

def main():
    """Main function to run the Streamlit app."""
    try:
        if not SessionState.get('authenticated'):
            show_login_page()
            return

        # Apply custom styles
        apply_custom_styles()
        
        # Add date filters in a container
        with st.container():
            st.title("🏀 NBA Game Predictions")
            col1, col2, col3 = st.columns([2, 2, 1])
            
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
            
            with col3:
                refresh = st.button("🔄 Refresh")

        # Convert dates to UTC datetime
        start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

        # Load or refresh predictions
        with st.spinner("Loading predictions..."):
            if refresh:
                predictions = refresh_predictions()
            else:
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

if __name__ == "__main__":
    main()
