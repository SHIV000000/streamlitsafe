# pages/01_History.py

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import logging
from nba_api_client import NBAGameResultsFetcher
import time
from datetime import timezone
from zoneinfo import ZoneInfo

def convert_to_et(utc_time: datetime) -> datetime:
    """Convert UTC time to Eastern Time"""
    et_zone = ZoneInfo("America/New_York")
    return utc_time.replace(tzinfo=timezone.utc).astimezone(et_zone)

# Page config
st.set_page_config(
    page_title="Prediction History",
    page_icon="📊",
    layout="wide"
)

# Reuse the CSS from main app
st.markdown("""
    <style>
    /* Add your CSS here */
    </style>
""", unsafe_allow_html=True)

class RewardSystem:
    def __init__(self):
        self.total_coins = 0
        self.boost_points = 0
        self.target_coins = 2460
        self.target_boost_points = 2337
        
    def calculate_rewards(self, prediction: Dict, actual_result: Dict) -> Dict:
        """Calculate rewards for a single prediction"""
        rewards = {
            'coins': 0,
            'boost_points': 0,
            'accuracy': 0.0
        }
        
        # Basic win/loss prediction (1 coin)
        if self._check_winner_prediction(prediction, actual_result):
            rewards['coins'] = 1
        
        # Calculate score accuracy
        score_accuracy = self._calculate_score_accuracy(prediction, actual_result)
        rewards['accuracy'] = score_accuracy
        
        # Award boost points for high accuracy
        if score_accuracy >= 0.95:
            rewards['boost_points'] = 1
            
        return rewards
    
    def _check_winner_prediction(self, prediction: Dict, actual_result: Dict) -> bool:
        """Check if winner prediction was correct"""
        pred_winner = prediction['prediction']['predicted_winner']
        actual_winner = actual_result['winner']
        return pred_winner == actual_winner
    
    def _calculate_score_accuracy(self, prediction: Dict, actual_result: Dict) -> float:
        """Calculate score prediction accuracy"""
        pred_score = prediction['prediction']['score_prediction']
        actual_score = actual_result['score']
        
        # Calculate average predicted scores
        pred_home = (pred_score['home_low'] + pred_score['home_high']) / 2
        pred_away = (pred_score['away_low'] + pred_score['away_high']) / 2
        
        # Calculate accuracy based on point difference
        max_diff = abs(actual_score['home'] - pred_home) + abs(actual_score['away'] - pred_away)
        return max(0, 1 - (max_diff / (actual_score['home'] + actual_score['away'])))

def load_prediction_history() -> List[Dict]:
    """Load historical predictions and fetch actual results from NBA API"""
    history = []
    history_dir = "history"
    nba_fetcher = NBAGameResultsFetcher()
    current_time = datetime.now()
    
    if not os.path.exists(history_dir):
        return []
    
    # Load and group predictions by date
    date_predictions = {}
    for file in os.listdir(history_dir):
        if file.endswith('.json'):
            try:
                with open(os.path.join(history_dir, file), 'r') as f:
                    prediction = json.load(f)
                    # Extract date and convert to Eastern Time (NBA's timezone)
                    game_date = datetime.fromisoformat(
                        prediction['game_info']['scheduled_start'].replace('Z', '+00:00')
                    )
                    # Convert to Eastern Time
                    et_date = game_date - timedelta(hours=5)  # EST offset
                    game_date = et_date.date()
                    
                    if game_date not in date_predictions:
                        date_predictions[game_date] = []
                    date_predictions[game_date].append(prediction)
                    
            except Exception as e:
                logging.error(f"Error loading prediction file {file}: {str(e)}")
                continue
    
    # Process each date's predictions
    for game_date, predictions in date_predictions.items():
        try:
            logging.info(f"Processing predictions for date: {game_date}")
            # Convert date to string in correct format for API
            date_str = game_date.strftime('%Y-%m-%d')
            results = nba_fetcher.get_game_results(game_date)
            
            if results:
                for prediction in predictions:
                    # Handle both dictionary and string formats for team names
                    home_team = (prediction['game_info']['home_team']['name'] 
                               if isinstance(prediction['game_info']['home_team'], dict) 
                               else prediction['game_info']['home_team'])
                    away_team = (prediction['game_info']['away_team']['name'] 
                               if isinstance(prediction['game_info']['away_team'], dict) 
                               else prediction['game_info']['away_team'])
                    
                    # Match with results
                    for game_result in results.values():
                        if (game_result['home_team'] == home_team and 
                            game_result['away_team'] == away_team):
                            prediction['nba_result'] = {
                                'winner': game_result['winner'],
                                'score': {
                                    'home': game_result['home_score'],
                                    'away': game_result['away_score']
                                }
                            }
                            logging.info(f"Matched result for {home_team} vs {away_team}")
                            break
                    
                    history.append(prediction)
            else:
                # If no results found but the game should be completed, try again
                if game_date <= current_time.date():
                    logging.warning(f"No results found for completed date {game_date}, retrying...")
                    time.sleep(2)  # Add delay before retry
                    results = nba_fetcher.get_game_results(game_date)
                    if results:
                        # Process results as above
                        for prediction in predictions:
                            home_team = (prediction['game_info']['home_team']['name'] 
                                       if isinstance(prediction['game_info']['home_team'], dict) 
                                       else prediction['game_info']['home_team'])
                            away_team = (prediction['game_info']['away_team']['name'] 
                                       if isinstance(prediction['game_info']['away_team'], dict) 
                                       else prediction['game_info']['away_team'])
                            
                            for game_result in results.values():
                                if (game_result['home_team'] == home_team and 
                                    game_result['away_team'] == away_team):
                                    prediction['nba_result'] = {
                                        'winner': game_result['winner'],
                                        'score': {
                                            'home': game_result['home_score'],
                                            'away': game_result['away_score']
                                        }
                                    }
                                    logging.info(f"Matched result for {home_team} vs {away_team}")
                                    break
                            
                            history.append(prediction)
                    else:
                        logging.warning(f"Still no results found for date {game_date}")
                        history.extend(predictions)
                else:
                    # Future games
                    history.extend(predictions)
                
        except Exception as e:
            logging.error(f"Error processing date {game_date}: {str(e)}")
            history.extend(predictions)
    
    return sorted(history, key=lambda x: x['timestamp'], reverse=True)

def display_history_dashboard():
    """Display enhanced history dashboard with rewards"""
    st.title("📊 Prediction History & Rewards")
    
    # Load history
    history = load_prediction_history()
    
    if not history:
        st.info("No prediction history available yet.")
        return
    
    # Calculate statistics for completed games only
    completed_games = [game for game in history if 'nba_result' in game]
    total_predictions = len(completed_games)
    
    if total_predictions > 0:
        correct_predictions = sum(
            1 for game in completed_games 
            if game['prediction']['predicted_winner'] == game['nba_result']['winner']
        )
        accuracy = (correct_predictions / total_predictions) * 100
        
        # Calculate rewards
        total_coins = correct_predictions
        boost_points = sum(
            1 for game in completed_games 
            if (game['prediction']['predicted_winner'] == game['nba_result']['winner'] 
                and float(game['prediction']['win_probability']) > 0.75)
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("Correct Predictions", correct_predictions)
        with col3:
            st.metric("Coins Earned", total_coins)
        with col4:
            st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Create and display history table
    df = create_history_dataframe(history)
    
    # Add filters with better error handling
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            min_date = pd.to_datetime(df['Date']).min().date()
            max_date = pd.to_datetime(df['Date']).max().date()
            
            # Handle single and range date selections
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Convert single date to tuple if needed
            if isinstance(date_range, tuple):
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range
                
        except Exception as e:
            logging.error(f"Error with date filter: {str(e)}")
            start_date = end_date = datetime.now().date()
    
    with col2:
        result_filter = st.multiselect(
            "Filter by Result",
            options=['Correct', 'Incorrect', 'Pending'],
            default=['Correct', 'Incorrect', 'Pending']
        )
    
    # Apply filters safely
    try:
        mask = (
            (pd.to_datetime(df['Date']).dt.date >= start_date) &
            (pd.to_datetime(df['Date']).dt.date <= end_date) &
            (df['Result'].isin(result_filter))
        )
        filtered_df = df[mask].copy()  # Create a copy to avoid SettingWithCopyWarning
    except Exception as e:
        logging.error(f"Error applying filters: {str(e)}")
        filtered_df = df.copy()
    
    # Configure column display
    column_config = {
        'Date': st.column_config.DatetimeColumn(
            'Date',
            format='MM/DD/YYYY HH:mm'
        ),
        'Predicted Score': st.column_config.TextColumn(
            'Predicted Score',
            help='Predicted score range (Home vs Away)'
        ),
        'Actual Score': st.column_config.TextColumn(
            'Actual Score',
            help='Final game score (Home-Away)'
        ),
        'Confidence': st.column_config.ProgressColumn(
            'Confidence',
            min_value=0,
            max_value=100,
            format='%.1f%%'
        )
    }
    
    # Display filtered data with column configuration
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
    
    # Display charts
    display_performance_charts(filtered_df)

def create_history_dataframe(history: List[Dict]) -> pd.DataFrame:
    """Convert history to DataFrame with enhanced information and handle multiple formats"""
    records = []
    current_date = datetime.now()

    for game in history:
        try:
            # Extract prediction details
            pred_info = game['prediction']
            game_info = game['game_info']
            
            # Handle different team info formats
            home_team = (game_info['home_team']['name'] 
                        if isinstance(game_info['home_team'], dict) 
                        else game_info['home_team'])
            away_team = (game_info['away_team']['name'] 
                        if isinstance(game_info['away_team'], dict) 
                        else game_info['away_team'])
            
            # Get predicted score ranges
            score_pred = pred_info['score_prediction']
            predicted_score = (
                f"{score_pred['home_low']}-{score_pred['home_high']} vs "
                f"{score_pred['away_low']}-{score_pred['away_high']}"
            )
            
            # Get game date and ensure it's in the correct format
            try:
                game_date = datetime.fromisoformat(
                    game_info['scheduled_start'].replace('Z', '+00:00')
                )
                game_date = game_date.astimezone().replace(tzinfo=None)
            except Exception as e:
                logging.error(f"Error parsing date: {str(e)}")
                continue
            
            # Handle actual results
            if 'nba_result' in game:
                nba_result = game['nba_result']
                score_pred = pred_info['score_prediction']
                actual_score = f"{nba_result['score']['home']}-{nba_result['score']['away']}"
                
                # Check winner prediction
                is_correct = pred_info['predicted_winner'] == nba_result['winner']
                result = 'Correct' if is_correct else 'Incorrect'
                
                # Check if actual scores are within predicted ranges
                home_in_range = (
                    score_pred['home_low'] <= nba_result['score']['home'] <= score_pred['home_high']
                )
                away_in_range = (
                    score_pred['away_low'] <= nba_result['score']['away'] <= score_pred['away_high']
                )
                
                # Award boost point if both scores are within range
                gets_boost = home_in_range and away_in_range
            else:
                is_correct = None
                actual_score = "Pending"
                result = 'Pending'
                gets_boost = False
            
            # Handle confidence value
            try:
                confidence = float(pred_info['win_probability'])
            except (ValueError, TypeError):
                confidence = 0.0
            
            # Create record with updated boost points logic
            record = {
                'Date': game_date,
                'Home Team': str(home_team),
                'Away Team': str(away_team),
                'Predicted Winner': str(pred_info['predicted_winner']),
                'Confidence': f"{confidence:.1%}",
                'Predicted Score': predicted_score,
                'Actual Score': actual_score,
                'Result': result,
                'Coins': 1 if is_correct else 0,
                'Boost Points': 1 if gets_boost else 0,
                'Game Status': 'Upcoming' if game_date > current_date else 'Completed'
            }
            records.append(record)
            
        except Exception as e:
            logging.error(f"Error processing game record: {str(e)}")
            continue
    
    # Create DataFrame with explicit string types for team columns
    df = pd.DataFrame(records).astype({
        'Home Team': str,
        'Away Team': str,
        'Predicted Winner': str
    })
    
    # Add confidence analysis columns
    if not df.empty:
        df = df.assign(
            Confidence_Float=df['Confidence'].str.rstrip('%').astype(float) / 100,
            Confidence_Level=pd.cut(
                df['Confidence'].str.rstrip('%').astype(float) / 100,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
        )
    
    return df

def display_performance_charts(df: pd.DataFrame):
    """Display enhanced performance analysis charts"""
    if df.empty:
        st.warning("No data available for charts")
        return
        
    st.subheader("Performance Analysis")
    
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Ensure Date column exists
    if 'Date' not in df.columns:
        st.error("Date column not found in DataFrame")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily prediction accuracy
        try:
            daily_stats = (df.groupby(df['Date'].dt.date)
                          ['Result'].agg(['count', lambda x: (x == 'Correct').mean()])
                          .rename(columns={'count': 'Total Games', '<lambda_0>': 'Accuracy'}))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_stats.index,
                y=daily_stats['Accuracy'] * 100,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#3b82f6')
            ))
            
            fig.update_layout(
                title='Daily Prediction Accuracy',
                yaxis_title='Accuracy (%)',
                xaxis_title='Date',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating accuracy chart: {str(e)}")
    
    with col2:
        # Rewards Distribution
        rewards_data = pd.DataFrame({
            'Type': ['Coins', 'Boost Points'],
            'Amount': [df['Coins'].sum(), df['Boost Points'].sum()]
        })
        
        fig = px.pie(
            rewards_data,
            values='Amount',
            names='Type',
            title='Rewards Distribution',
            color_discrete_sequence=['#3b82f6', '#ef4444']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional Analysis Section
    st.subheader("Detailed Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Team Performance Analysis
        home_stats = (df.groupby('Home Team', observed=True)['Result']
                     .agg(['count', lambda x: (x == 'Correct').mean()]))
        away_stats = (df.groupby('Away Team', observed=True)['Result']
                     .agg(['count', lambda x: (x == 'Correct').mean()]))
        
        team_stats = pd.concat([home_stats, away_stats], axis=1)
        team_stats.columns = ['Home Games', 'Home Accuracy', 'Away Games', 'Away Accuracy']
        
        # Calculate totals using loc to avoid warnings
        team_stats.loc[:, 'Total Games'] = team_stats['Home Games'] + team_stats['Away Games']
        team_stats.loc[:, 'Overall Accuracy'] = (
            (team_stats['Home Games'] * team_stats['Home Accuracy'] + 
             team_stats['Away Games'] * team_stats['Away Accuracy']) / 
            team_stats['Total Games']
        )
        
        # Display top performing teams
        st.markdown("#### Top Performing Teams")
        top_teams = team_stats.sort_values('Overall Accuracy', ascending=False).head(5)
        
        fig = px.bar(
            top_teams,
            y=top_teams.index,
            x='Overall Accuracy',
            title='Top 5 Teams by Prediction Accuracy',
            labels={'Overall Accuracy': 'Accuracy', 'y': 'Team'},
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence Analysis
        st.markdown("#### Prediction Confidence Analysis")
        
        # Process confidence data safely
        df.loc[:, 'Confidence_Float'] = df['Confidence'].str.rstrip('%').astype(float) / 100
        df.loc[:, 'Confidence_Level'] = pd.cut(
            df['Confidence_Float'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        confidence_analysis = (df.groupby('Confidence_Level', observed=True)
                             .agg({
                                 'Result': lambda x: (x == 'Correct').mean(),
                                 'Coins': 'sum',
                                 'Boost Points': 'sum'
                             })
                             .round(3))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=confidence_analysis.index,
            y=confidence_analysis['Result'] * 100,
            name='Accuracy',
            marker_color='#3b82f6'
        ))
        
        fig.update_layout(
            title='Accuracy by Confidence Level',
            yaxis_title='Accuracy (%)',
            xaxis_title='Confidence Level',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)


def calculate_current_streak(df: pd.DataFrame) -> int:
    """Calculate current prediction streak"""
    if df.empty:
        return 0
    
    results = df.sort_values('Date', ascending=True)['Result'].values
    streak = 0
    
    for result in reversed(results):
        if result == 'Correct':
            streak += 1
        else:
            break
    
    return streak

def calculate_best_streak(df: pd.DataFrame) -> int:
    """Calculate best prediction streak"""
    if df.empty:
        return 0
    
    results = df.sort_values('Date', ascending=True)['Result'].values
    current_streak = 0
    best_streak = 0
    
    for result in results:
        if result == 'Correct':
            current_streak += 1
            best_streak = max(best_streak, current_streak)
        else:
            current_streak = 0
    
    return best_streak

def main():
    """Main function for history page"""
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        st.warning("Please log in from the main dashboard to access this page.")
        return
        
    display_history_dashboard()

if __name__ == "__main__":
    main()



