# pages/01_History.py

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import logging
from nba_api_client import NBAGameResultsFetcher

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
    
    if not os.path.exists(history_dir):
        return []
    
    for file in os.listdir(history_dir):
        if file.endswith('.json'):
            try:
                with open(os.path.join(history_dir, file), 'r') as f:
                    prediction = json.load(f)
                    
                    # Get game date from scheduled_start
                    game_date = datetime.fromisoformat(
                        prediction['game_info']['scheduled_start'].replace('Z', '+00:00')
                    ).date()
                    
                    # Only fetch results for past games
                    if game_date < datetime.now().date():
                        # Fetch actual result from NBA API
                        results = nba_fetcher.get_game_results(game_date)
                        
                        # Match teams and determine actual winner
                        home_team = prediction['game_info']['home_team']
                        away_team = prediction['game_info']['away_team']
                        
                        for game_result in results.values():
                            if (game_result['home_team'] == home_team and 
                                game_result['away_team'] == away_team):
                                # Add actual result to prediction
                                prediction['nba_result'] = {
                                    'winner': game_result['winner'],
                                    'score': {
                                        'home': game_result['home_score'],
                                        'away': game_result['away_score']
                                    }
                                }
                                break
                    
                    history.append(prediction)
                    
            except Exception as e:
                logging.error(f"Error loading prediction file {file}: {str(e)}")
    
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
    
    # Add filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        date_filter = st.date_input(
            "Select Date Range",
            value=(
                pd.to_datetime(df['Date']).min().date(),
                pd.to_datetime(df['Date']).max().date()
            )
        )
    
    with col2:
        result_filter = st.multiselect(
            "Filter by Result",
            options=['Correct', 'Incorrect'],
            default=['Correct', 'Incorrect']
        )
    
    # Apply filters
    mask = (
        (pd.to_datetime(df['Date']).dt.date >= date_filter[0]) &
        (pd.to_datetime(df['Date']).dt.date <= date_filter[1]) &
        (df['Result'].isin(result_filter))
    )
    filtered_df = df[mask]
    
    # Display filtered data
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Display charts
    display_performance_charts(filtered_df)

def create_history_dataframe(history: List[Dict]) -> pd.DataFrame:
    """Convert history to DataFrame with enhanced information"""
    records = []
    for game in history:
        # Extract prediction details
        pred_info = game['prediction']
        game_info = game['game_info']
        
        # Only include completed games with results
        if 'nba_result' in game:
            nba_result = game['nba_result']
            is_correct = pred_info['predicted_winner'] == nba_result['winner']
            actual_score = f"{nba_result['score']['home']}-{nba_result['score']['away']}"
            result = 'Correct' if is_correct else 'Incorrect'
        else:
            is_correct = False
            actual_score = "Pending"
            result = 'Pending'
        
        # Handle confidence value
        confidence = float(pred_info['win_probability'])
        
        record = {
            'Date': datetime.fromisoformat(game['timestamp']).strftime('%Y-%m-%d %H:%M'),
            'Home Team': game_info['home_team'],
            'Away Team': game_info['away_team'],
            'Predicted Winner': pred_info['predicted_winner'],
            'Confidence': f"{confidence:.1%}",
            'Actual Score': actual_score,
            'Result': result,
            'Coins': 1 if is_correct else 0,
            'Boost Points': 1 if confidence > 0.75 and is_correct else 0
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def display_performance_charts(df: pd.DataFrame):
    """Display enhanced performance analysis charts"""
    st.subheader("Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily prediction accuracy
        daily_stats = df.groupby(
            pd.to_datetime(df['Date']).dt.date
        )['Result'].agg(['count', lambda x: (x == 'Correct').mean()])
        daily_stats.columns = ['Total Games', 'Accuracy']
        
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
        team_stats = pd.DataFrame()
        
        # Combine home and away predictions
        home_stats = df.groupby('Home Team')['Result'].agg(['count', lambda x: (x == 'Correct').mean()])
        away_stats = df.groupby('Away Team')['Result'].agg(['count', lambda x: (x == 'Correct').mean()])
        
        team_stats = pd.concat([home_stats, away_stats], axis=1)
        team_stats.columns = ['Home Games', 'Home Accuracy', 'Away Games', 'Away Accuracy']
        team_stats['Total Games'] = team_stats['Home Games'] + team_stats['Away Games']
        team_stats['Overall Accuracy'] = (
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
        
        # Convert confidence strings to floats
        df['Confidence_Float'] = df['Confidence'].str.rstrip('%').astype(float) / 100
        
        # Create manual bins for confidence
        df['Confidence_Level'] = pd.cut(
            df['Confidence_Float'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        confidence_analysis = df.groupby('Confidence_Level').agg({
            'Result': lambda x: (x == 'Correct').mean(),
            'Coins': 'sum',
            'Boost Points': 'sum'
        }).round(3)
        
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

    # Summary Statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Overall Stats
        st.markdown("#### Overall Statistics")
        total_games = len(df)
        correct_predictions = (df['Result'] == 'Correct').sum()
        overall_accuracy = (correct_predictions / total_games) * 100
        
        st.metric("Total Games Predicted", total_games)
        st.metric("Correct Predictions", correct_predictions)
        st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    
    with col2:
        # Rewards Stats
        st.markdown("#### Rewards Statistics")
        total_coins = df['Coins'].sum()
        total_boost = df['Boost Points'].sum()
        avg_coins_per_day = df.groupby(pd.to_datetime(df['Date']).dt.date)['Coins'].sum().mean()
        
        st.metric("Total Coins Earned", total_coins)
        st.metric("Total Boost Points", total_boost)
        st.metric("Average Daily Coins", f"{avg_coins_per_day:.1f}")
    
    with col3:
        # Streak Analysis
        st.markdown("#### Streak Analysis")
        current_streak = calculate_current_streak(df)
        best_streak = calculate_best_streak(df)
        
        st.metric("Current Streak", f"{current_streak} correct")
        st.metric("Best Streak", f"{best_streak} correct")
        st.metric("Success Rate", f"{(df['Result'] == 'Correct').mean()*100:.1f}%")

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



