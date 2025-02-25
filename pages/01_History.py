import streamlit as st
import logging
from datetime import datetime, timezone, timedelta
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

def init_supabase():
    """Initialize Supabase client if not already initialized."""
    if not SessionState.get('supabase_client'):
        SUPABASE_URL = "https://jdvxisvtktunywgdtxvz.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impkdnhpc3Z0a3R1bnl3Z2R0eHZ6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTE2MDAsImV4cCI6MjA1NTk2NzYwMH0.-Hdbq82ctFUCGjXkmzRDOUzlXkHjVZfp5ws4vpIFmi4"
        supabase = Client(SUPABASE_URL, SUPABASE_KEY)
        SessionState.set('supabase_client', supabase)
    return SessionState.get('supabase_client')

def load_predictions(start_date=None, end_date=None):
    """Load predictions from Supabase."""
    try:
        supabase = init_supabase()
        
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
        
        if predictions:
            # Sort predictions by scheduled start time
            predictions.sort(key=lambda x: x['scheduled_start'])
            
        return predictions
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        logging.error(f"Error loading predictions: {str(e)}")
        return []

def show_navigation():
    """Show navigation bar."""
    st.markdown(
        """
        <style>
        .navbar {
            padding: 1rem;
            margin-bottom: 2rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container():
        st.markdown('<div class="navbar">', unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 1])
        
        with cols[0]:
            st.markdown(f"👤 Welcome, {SessionState.get_username()}")
        with cols[1]:
            if st.button("🏠 Home"):
                st.switch_page("home.py")
        with cols[2]:
            if st.button("🔄 Refresh"):
                st.rerun()
        with cols[3]:
            if st.button("🚪 Logout"):
                SessionState.logout()
                st.switch_page("home.py")
                
        st.markdown('</div>', unsafe_allow_html=True)

def show_history():
    """Show prediction history."""
    show_navigation()
    st.title("📊 Prediction History")
    
    # Date filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now(timezone.utc).date() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(timezone.utc).date()
        )
    
    # Convert dates to UTC datetime
    start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    
    # Load predictions
    with st.spinner("Loading predictions..."):
        predictions = load_predictions(start_datetime, end_datetime)
    
    if not predictions:
        st.info("No predictions found for the selected date range.")
        return
    
    # Display predictions
    for prediction in predictions:
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {prediction['home_team']}")
                st.write(f"Score Range: {prediction['home_score_min']}-{prediction['home_score_max']}")
                
            with col2:
                st.markdown(f"### {prediction['away_team']}")
                st.write(f"Score Range: {prediction['away_score_min']}-{prediction['away_score_max']}")
            
            st.write(f"Game Time: {prediction['scheduled_start']}")
            
            # Prediction
            winner = prediction['predicted_winner']
            winner_name = prediction['home_team'] if winner == 'home' else prediction['away_team']
            prob = prediction['win_probability']
            
            st.markdown(f"**Predicted Winner:** {winner_name}")
            st.markdown(f"**Win Probability:** {prob:.1f}%")
            st.markdown("---")

def main():
    # Initialize session state
    SessionState.init_state()
    
    # Check authentication
    if not SessionState.is_authenticated():
        st.warning("Please log in first!")
        st.switch_page("home.py")
    else:
        show_history()

if __name__ == "__main__":
    main()