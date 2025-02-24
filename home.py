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
        supabase = Client(SUPABASE_URL, SUPABASE_KEY)
        SessionState.set('supabase_client', supabase)
except Exception as e:
    st.error(f"Error initializing Supabase client: {str(e)}")
    supabase = None

# Initialize NBA API client
NBA_API_KEY = "918ef216c6msh607da23f482096fp198faajsnc648d53dadc5"
nba_client = EnhancedNBAApiClient(NBA_API_KEY)

def apply_custom_styles():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        /* Hide sidebar and menu */
        [data-testid="stSidebar"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Remove default Streamlit padding and margins */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
            max-width: 95% !important;
        }
        
        .element-container, .stMarkdown {
            margin-bottom: 0 !important;
        }
        
        /* Game card styling */
        .game-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .game-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
        }
        
        /* Team names */
        .team-name {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a237e;
            margin: 0;
            padding: 0;
        }
        
        /* VS text */
        .vs-text {
            font-size: 1.2rem;
            font-weight: 500;
            color: #9e9e9e;
            text-align: center;
            margin: 0.5rem 0;
            padding: 0;
        }
        
        /* Score range */
        .score-range {
            font-size: 1.1rem;
            color: #424242;
            padding: 0.5rem 1rem;
            background: #f5f5f5;
            border-radius: 8px;
            display: inline-block;
            margin: 0.5rem 0 0 0;
        }
        
        /* Prediction details */
        .prediction-details {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
        }
        
        .prediction-winner {
            font-size: 1.2rem;
            color: #2e7d32;
            font-weight: 600;
            margin: 0;
            padding: 0;
        }
        
        .win-probability {
            font-size: 1.1rem;
            color: #1565c0;
            font-weight: 500;
            margin: 0.5rem 0 0 0;
            padding: 0;
        }
        
        /* Game time */
        .game-time {
            margin: 0.5rem 0 0 0;
            padding: 0;
            font-size: 1rem;
            color: #757575;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Admin controls */
        .admin-controls {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Title styling */
        h1 {
            color: #1a237e;
            font-weight: 700;
            margin: 1rem 0;
            text-align: center;
        }
        
        /* Column layout */
        .st-emotion-cache-1n76uvr {
            gap: 0.5rem !important;
        }
        
        /* Button styling */
        .stButton > button {
            margin: 0 !important;
            border: 1px solid #ddd;
            background-color: white;
            color: #333;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background-color: #f0f0f0;
            border-color: #ccc;
            transform: translateY(-2px);
        }
        
        /* Info message styling */
        .stAlert {
            background: #e3f2fd;
            border-radius: 10px;
            border: none;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def get_team_strength(team_name):
    """Get team strength based on 2024 season performance (0.0 to 1.0 scale)"""
    return {
        'Boston Celtics': 0.95, 'Denver Nuggets': 0.90, 'Minnesota Timberwolves': 0.89,
        'LA Clippers': 0.88, 'Oklahoma City Thunder': 0.87, 'Milwaukee Bucks': 0.86,
        'Cleveland Cavaliers': 0.85, 'Phoenix Suns': 0.84, 'New York Knicks': 0.83,
        'Sacramento Kings': 0.82, 'New Orleans Pelicans': 0.81, 'Dallas Mavericks': 0.80,
        'Philadelphia 76ers': 0.79, 'Miami Heat': 0.78, 'Indiana Pacers': 0.77,
        'Los Angeles Lakers': 0.76, 'Orlando Magic': 0.75, 'Golden State Warriors': 0.74,
        'Houston Rockets': 0.73, 'Chicago Bulls': 0.72, 'Atlanta Hawks': 0.71,
        'Utah Jazz': 0.70, 'Brooklyn Nets': 0.69, 'Toronto Raptors': 0.68,
        'Memphis Grizzlies': 0.67, 'Portland Trail Blazers': 0.66, 'Washington Wizards': 0.64,
        'Charlotte Hornets': 0.62, 'San Antonio Spurs': 0.61, 'Detroit Pistons': 0.55
    }.get(team_name, 0.70)

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
        # Get upcoming games
        games = nba_client.get_upcoming_games()
        if not games:
            st.warning("No upcoming games found.")
            return []
        
        # Delete old predictions first
        try:
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            supabase.table('predictions').delete().lt('scheduled_start', today_start).execute()
        except Exception as e:
            logging.error(f"Error deleting old predictions: {str(e)}")
        
        # Process each game
        predictions = []
        for game in games:
            try:
                # Check if prediction already exists for this game and date
                existing = (
                    supabase.table('predictions')
                    .select('*')
                    .eq('home_team', game['teams']['home']['name'])
                    .eq('away_team', game['teams']['away']['name'])
                    .eq('scheduled_start', game['date']['start'])
                    .execute()
                )
                
                if existing.data:
                    # Use existing prediction
                    predictions.append(existing.data[0])
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
                
                result = (
                    supabase.table('predictions')
                    .insert(data)
                    .execute()
                )
                
                if result.data:
                    predictions.append(result.data[0])
                    logging.info(f"Saved new prediction for {game['teams']['home']['name']} vs {game['teams']['away']['name']}")
                
            except Exception as e:
                logging.error(f"Error processing game: {str(e)}")
                continue
        
        return predictions
        
    except Exception as e:
        st.error(f"Error fetching games: {str(e)}")
        logging.error(f"Error in fetch_and_save_games: {str(e)}")
        return []

def save_prediction(data):
    """Save a prediction to Supabase."""
    try:
        return supabase.table('predictions').insert(data).execute()
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")
        return None

def load_predictions(include_live=False):
    """Load predictions from Supabase database."""
    try:
        # Get today's date in UTC
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        tomorrow_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Query predictions
        query = (
            supabase.table('predictions')
            .select('*')
            .gte('scheduled_start', today_start)
            .lt('scheduled_start', tomorrow_start)
            .order('scheduled_start', desc=False)
            .execute()
        )
        
        # Process results
        if query.data:
            # Remove any duplicates based on teams and scheduled_start
            seen = set()
            unique_predictions = []
            for pred in query.data:
                key = (pred['home_team'], pred['away_team'], pred['scheduled_start'])
                if key not in seen:
                    seen.add(key)
                    unique_predictions.append(pred)
            
            logging.info(f"Loaded {len(unique_predictions)} predictions for today")
            return unique_predictions
            
        logging.info("No predictions found for today")
        return []
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        logging.error(f"Error in load_predictions: {str(e)}")
        return []

def refresh_predictions():
    """Delete existing predictions and generate new ones"""
    try:
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        tomorrow_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        # Delete today's predictions
        supabase.table('predictions').delete().gte('scheduled_start', today_start).lt('scheduled_start', tomorrow_start).execute()
        logging.info("Deleted existing predictions for today")
        
        # Fetch new games and generate new predictions
        games = nba_client.get_upcoming_games()
        new_predictions = []
        
        if games:
            for game in games:
                prediction = generate_prediction(game)
                if prediction:
                    result = save_prediction(prediction)
                    if result and result.data:
                        new_predictions.extend(result.data)
        
        return new_predictions
    except Exception as e:
        logging.error(f"Error refreshing predictions: {str(e)}")
        st.error(f"Error refreshing predictions: {str(e)}")
        return []

def display_game_card(prediction):
    """Display a game prediction card."""
    try:
        # Extract game information
        home_team = prediction['home_team']
        away_team = prediction['away_team']
        predicted_winner = prediction['predicted_winner']
        win_probability = prediction['win_probability']
        scheduled_start = prediction['scheduled_start']
        
        # Convert game time to local time
        try:
            game_time_utc = datetime.fromisoformat(scheduled_start.replace('Z', '+00:00'))
        except ValueError:
            # If the date is already in ISO format with timezone
            game_time_utc = datetime.fromisoformat(scheduled_start)
            
        # Get user's local timezone
        local_tz = pytz.timezone('Asia/Kolkata')  # Using Indian Standard Time
        game_time_local = game_time_utc.astimezone(local_tz)
        
        # Format times for display
        utc_display = game_time_utc.strftime('%Y-%m-%d %H:%M UTC')
        local_display = game_time_local.strftime('%Y-%m-%d %H:%M %Z')
        
        # Create columns for the game card
        col1, col2, col3 = st.columns([2,3,2])
        
        with col1:
            st.markdown(f"### {away_team}")
            st.write(f"Score Range: {prediction['away_score_min']}-{prediction['away_score_max']}")
            
        with col2:
            st.markdown("### vs")
            st.write(f"Game Time: {local_display}")
            st.write(f"Predicted Winner: {predicted_winner}")
            st.write(f"Win Probability: {win_probability:.1%}")
            
        with col3:
            st.markdown(f"### {home_team}")
            st.write(f"Score Range: {prediction['home_score_min']}-{prediction['home_score_max']}")
            
        st.markdown("---")
    except Exception as e:
        logging.error(f"Error displaying game card: {str(e)}")
        st.error("Error displaying game prediction")

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
    """Display the login page"""
    st.title("NBA Predictions")
    
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

def main():
    apply_custom_styles()
    
    if not SessionState.get('authenticated'):
        show_login_page()
        return
    
    create_navigation()
    st.title("🏀 NBA Game Predictions")
    
    # Add refresh button
    refresh = st.button("🔄 Generate New Predictions")
    
    with st.spinner("Loading predictions..."):
        if refresh:
            predictions = refresh_predictions()
        else:
            predictions = load_predictions(include_live=False)
        
        if predictions:
            # Sort predictions by game time
            predictions.sort(key=lambda x: x['scheduled_start'])
            for prediction in predictions:
                display_game_card(prediction)
        else:
            st.info("No predictions available. Click 'Generate New Predictions' to create predictions.")

if __name__ == "__main__":
    main()
