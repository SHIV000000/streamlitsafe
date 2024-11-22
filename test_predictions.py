# test_predictions.py

import logging
import json
from datetime import datetime
import os
from time import sleep
from api_client import EnhancedNBAApiClient
from prediction_service import NBAPredictor
from typing import Dict, Any, List
from datetime import datetime, timedelta
from typing import Optional
import time
import atexit
import logging
from datetime import datetime, timedelta

import atexit
import shutil
import threading
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(process)d - %(threadName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("live_predictions.log"),
        logging.StreamHandler()
    ]
)

class LiveGamePredictor:
    def __init__(self, base_predictor: NBAPredictor):
        self.base_predictor = base_predictor
        self.game_cache = {}
        self.update_interval = 30  # seconds

    def predict_live_game(self, game_info: Dict) -> Dict[str, Any]:
        """Make and update predictions for a live game."""
        game_id = game_info['gameId']
        current_time = datetime.now()

        if self._should_update_prediction(game_id, current_time):
            prediction = self._make_live_prediction(game_info)
            self.game_cache[game_id] = {
                'last_update': current_time,
                'prediction': prediction
            }
            return prediction
        
        return self.game_cache[game_id]['prediction']

    def _should_update_prediction(self, game_id: str, current_time: datetime) -> bool:
        if game_id not in self.game_cache:
            return True

        last_update = self.game_cache[game_id]['last_update']
        time_diff = (current_time - last_update).total_seconds()
        return time_diff >= self.update_interval

    def _make_live_prediction(self, game_info: Dict) -> Dict[str, Any]:
        base_prediction, model_predictions = self.base_predictor.predict_game(
            game_info['home_stats'],
            game_info['away_stats']
        )

        momentum_factor = self._calculate_momentum(game_info)
        performance_factor = self._calculate_performance_factor(game_info)
        time_pressure = self._calculate_time_pressure(game_info)

        adjusted_prediction = self._adjust_prediction(
            base_prediction,
            momentum_factor,
            performance_factor,
            time_pressure,
            game_info
        )

        return {
            'base_prediction': base_prediction,
            'adjusted_prediction': adjusted_prediction,
            'model_predictions': model_predictions,
            'factors': {
                'momentum': momentum_factor,
                'performance': performance_factor,
                'time_pressure': time_pressure
            },
            'game_state': {
                'period': game_info['current_period'],
                'time_remaining': self._parse_game_clock(game_info['clock']),
                'score_difference': game_info['home_score'] - game_info['away_score']
            }
        }

    def _calculate_momentum(self, game_info: Dict) -> float:
        """Calculate momentum factor with fallback for scheduled games."""
        try:
            # For scheduled games, return neutral momentum
            if not game_info.get('scores'):
                return 0.0

            # For live games, calculate momentum from scores
            home_scores = [int(score) for score in game_info['scores']['home']['linescore']]
            away_scores = [int(score) for score in game_info['scores']['away']['linescore']]
            
            recent_home = sum(home_scores[-2:])
            recent_away = sum(away_scores[-2:])
            
            momentum = (recent_home - recent_away) / max(recent_home + recent_away, 1)
            return max(min(momentum, 1.0), -1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating momentum: {str(e)}")
            return 0.0

    def _calculate_performance_factor(self, game_info: Dict) -> float:
        try:
            home_stats = game_info['home_stats']['statistics'][0]
            away_stats = game_info['away_stats']['statistics'][0]
            
            home_ppg = float(home_stats.get('points', 0))
            away_ppg = float(away_stats.get('points', 0))
            
            current_home_pace = game_info['home_score'] / max(game_info['current_period'], 1)
            current_away_pace = game_info['away_score'] / max(game_info['current_period'], 1)
            
            home_performance = current_home_pace / home_ppg if home_ppg > 0 else 1.0
            away_performance = current_away_pace / away_ppg if away_ppg > 0 else 1.0
            
            return home_performance - away_performance
            
        except Exception as e:
            logging.warning(f"Error calculating performance factor: {str(e)}")
            return 0.0

    def _calculate_time_pressure(self, game_info: Dict) -> float:
        try:
            total_minutes = 48.0
            current_minute = (game_info['current_period'] - 1) * 12
            
            if game_info['clock']:
                minutes, seconds = map(float, game_info['clock'].split(':'))
                current_minute += (12 - minutes - seconds/60)
            
            return min(current_minute / total_minutes, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating time pressure: {str(e)}")
            return 0.0

    def _parse_game_clock(self, clock_str: str) -> float:
        try:
            if not clock_str:
                return 12.0
            minutes, seconds = map(float, clock_str.split(':'))
            return minutes + (seconds / 60)
        except Exception as e:
            logging.warning(f"Error parsing game clock: {str(e)}")
            return 12.0

    def _adjust_prediction(
        self,
        base_pred: float,
        momentum: float,
        performance: float,
        time_pressure: float,
        game_info: Dict
    ) -> float:
        try:
            momentum_weight = 0.2
            performance_weight = 0.3
            score_weight = 0.5
            
            score_diff = game_info['home_score'] - game_info['away_score']
            max_diff = 20.0
            score_factor = max(min(score_diff / max_diff, 1.0), -1.0)
            
            adjustment = (
                momentum * momentum_weight +
                performance * performance_weight +
                score_factor * score_weight
            ) * time_pressure
            
            adjusted_pred = base_pred + (adjustment * (1 - base_pred))
            return max(min(adjusted_pred, 1.0), 0.0)
            
        except Exception as e:
            logging.warning(f"Error adjusting prediction: {str(e)}")
            return base_pred

def run_continuous_predictions(timeout_minutes=3):
    """Run predictions with single attempt and proper interval control"""
    try:
        # Get current time and check last update
        current_time = time.time()
        last_update = getattr(run_continuous_predictions, 'last_run', 0)
        time_since_update = current_time - last_update
        
        # If less than 5 minutes since last update, skip
        if time_since_update < 300:  # 5 minutes in seconds
            logging.info(f"Not updating - only {time_since_update:.1f} seconds since last update")
            return False
            
        logging.info("Starting prediction update...")
        
        # Initialize clients and predictors
        api_key = '89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c'
        api_client = EnhancedNBAApiClient(api_key)
        base_predictor = NBAPredictor('saved_models')
        live_predictor = LiveGamePredictor(base_predictor)
        
        # Get live games with detailed logging
        logging.info("Fetching live games...")
        live_games = api_client.get_live_games()
        predictions_made = False
        
        if live_games:
            logging.info(f"Found {len(live_games)} live games")
            for game in live_games:
                try:
                    game_info = prepare_game_info(game, api_client)
                    prediction = live_predictor.predict_live_game(game_info)
                    save_prediction(game_info, prediction, is_live=True)
                    predictions_made = True
                except Exception as e:
                    logging.error(f"Error processing game {game.get('id')}: {str(e)}")
                    continue
        else:
            today_games = get_todays_schedule(api_client)
            if today_games:
                process_scheduled_games(today_games, api_client, live_predictor)
                predictions_made = True
            else:
                logging.info("No games found")
        
        if predictions_made:
            run_continuous_predictions.last_run = time.time()
            logging.info("Predictions completed successfully")
            return True
            
        return False
        
    except Exception as e:
        logging.error(f"Error in prediction service: {str(e)}")
        return False

# Initialize the last run time
run_continuous_predictions.last_run = 0

def should_update_predictions():
    """Determine if predictions should be updated"""
    current_time = time.time()
    last_run = getattr(run_continuous_predictions, 'last_run', 0)
    time_since_last_run = current_time - last_run
    
    # Check if enough time has passed (5 minutes)
    if time_since_last_run < 300:
        logging.info(f"Not updating - only {time_since_last_run:.1f} seconds since last update")
        return False
        
    return True


def prepare_game_info(game: Dict, api_client: EnhancedNBAApiClient) -> Dict:
    """Prepare comprehensive game information."""
    try:
        teams = game.get('teams', {})
        home_team = teams.get('home', {})
        away_team = teams.get('away', {})
        
        # Get team information
        home_team_info = api_client.get_team_info(home_team.get('id'))
        away_team_info = api_client.get_team_info(away_team.get('id'))
        
        # Get injury information
        home_injuries = api_client.get_team_injuries(home_team.get('id'))
        away_injuries = api_client.get_team_injuries(away_team.get('id'))
        
        game_info = {
            'gameId': game.get('id'),
            'home_team': {
                'name': home_team.get('name'),
                'info': home_team_info,
                'injuries': home_injuries
            },
            'away_team': {
                'name': away_team.get('name'),
                'info': away_team_info,
                'injuries': away_injuries
            },
            'current_period': game.get('periods', {}).get('current', 1),
            'clock': game.get('status', {}).get('clock', '12:00'),
            'home_score': int(game.get('scores', {}).get('home', {}).get('points', 0)),
            'away_score': int(game.get('scores', {}).get('away', {}).get('points', 0)),
            'scores': game.get('scores')  # Add the entire scores object
        }
        
        # Get and process team stats
        home_stats = api_client.get_team_stats(home_team.get('id'))
        away_stats = api_client.get_team_stats(away_team.get('id'))
        
        # Adjust stats based on injuries
        home_stats = adjust_stats_for_injuries(home_stats, home_injuries)
        away_stats = adjust_stats_for_injuries(away_stats, away_injuries)
        
        game_info['home_stats'] = home_stats
        game_info['away_stats'] = away_stats
        
        return game_info
        
    except Exception as e:
        logging.error(f"Error preparing game info: {str(e)}")
        raise

def adjust_stats_for_injuries(stats: Dict, injuries: List[Dict]) -> Dict:
    """Adjust team statistics based on injured players."""
    try:
        if not injuries or not stats.get('statistics'):
            return stats
            
        # Calculate impact factor based on number and status of injuries
        impact_factor = sum(
            0.1 if injury['status'] == 'Out' else 0.05
            for injury in injuries
        )
        
        # Adjust statistics
        adjusted_stats = stats.copy()
        stat_keys = ['points', 'assists', 'rebounds', 'steals', 'blocks']
        
        for key in stat_keys:
            if key in adjusted_stats['statistics'][0]:
                adjusted_stats['statistics'][0][key] *= (1 - impact_factor)
                
        return adjusted_stats
        
    except Exception as e:
        logging.error(f"Error adjusting stats for injuries: {str(e)}")
        return stats

def log_prediction(game_info: Dict, prediction: Dict):
    """Log detailed prediction information."""
    logging.info(f"""
    ============ Game Update ============
    {game_info['home_team']} vs {game_info['away_team']}
    Period: {game_info['current_period']} | Time: {game_info['clock']}
    Score: {game_info['home_score']} - {game_info['away_score']}
    
    Predictions:
    - Base: {prediction['base_prediction']:.2%}
    - Adjusted: {prediction['adjusted_prediction']:.2%}
    
    Adjustment Factors:
    - Momentum: {prediction['factors']['momentum']:.3f}
    - Performance: {prediction['factors']['performance']:.3f}
    - Time Pressure: {prediction['factors']['time_pressure']:.3f}
    
    Model Predictions:
    {json.dumps(prediction['model_predictions'], indent=2)}
    
    Game State:
    - Time Remaining: {prediction['game_state']['time_remaining']:.1f} minutes
    - Score Difference: {prediction['game_state']['score_difference']} points
    =====================================
    """)

def save_prediction(game_info: Dict, prediction: Dict, is_live: bool = True):
    """Save prediction with complete structure."""
    try:
        timestamp = datetime.now()
        
        # Calculate win probabilities
        home_win_prob = prediction['adjusted_prediction']
        predicted_winner = game_info['home_team'] if home_win_prob > 0.5 else game_info['away_team']
        win_probability = home_win_prob if home_win_prob > 0.5 else (1 - home_win_prob)
        
        # Create complete prediction structure
        result = {
            'timestamp': timestamp.isoformat(),
            'game_info': {
                'id': game_info['gameId'],
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'scheduled_start': game_info.get('scheduled_start'),
                'period': game_info.get('current_period', 0),
                'clock': game_info.get('clock', '12:00'),
                'score': {
                    'home': game_info.get('home_score', 0),
                    'away': game_info.get('away_score', 0)
                }
            },
            'prediction': {
                'base': float(prediction.get('base_prediction', 0.5)),
                'adjusted': float(prediction.get('adjusted_prediction', 0.5)),
                'predicted_winner': predicted_winner,
                'win_probability': float(win_probability),
                'confidence_level': get_confidence_level(win_probability),
                'model_predictions': prediction.get('model_predictions', {}),
                'score_prediction': {
                    'home_low': 0,
                    'home_high': 0,
                    'away_low': 0,
                    'away_high': 0
                }
            }
        }
        
        directory = 'predictions/live' if is_live else 'predictions/scheduled'
        os.makedirs(directory, exist_ok=True)
        
        filename = f'{directory}/game_{game_info["gameId"]}_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
            
        logging.debug(f"Saved prediction to {filename}")
        return result
        
    except Exception as e:
        logging.error(f"Error saving prediction: {str(e)}")
        return None

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.6:
        return "Medium"
    else:
        return "Low"

def debug_game_status(game: Dict) -> str:
    """Helper function to debug game status"""
    return (f"Game ID: {game.get('id')} - "
            f"Status: {game.get('status', {}).get('long')} - "
            f"Clock: {game.get('status', {}).get('clock')} - "
            f"Period: {game.get('periods', {}).get('current')} - "
            f"{game.get('teams', {}).get('home', {}).get('name')} "
            f"vs {game.get('teams', {}).get('visitors', {}).get('name')}")

def get_todays_schedule(api_client: EnhancedNBAApiClient) -> List[Dict]:
    """Get today's game schedule with fallback to tomorrow."""
    try:
        # Get today's date and tomorrow's date
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        dates_to_try = [today, tomorrow]
        
        all_games = []
        season = "2024"  # Current NBA season
        
        # Try each date until we find games
        for current_date in dates_to_try:
            date_str = current_date.strftime("%Y-%m-%d")
            logging.info(f"Checking schedule for date: {date_str}")
            
            endpoint = f"{api_client.base_url}/games"
            params = {
                'date': date_str,
                'season': season,
                'league': 'standard'
            }
            
            try:
                response = api_client._make_request('GET', endpoint, params)
                games = response.get('response', [])
                
                if games:
                    logging.info(f"Found {len(games)} games for {date_str}")
                    processed_games = []
                    
                    for game in games:
                        # Process each game
                        game_date_str = game.get('date', {}).get('start')
                        if not game_date_str:
                            continue
                            
                        try:
                            game_date = datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                            
                            # Only include future games
                            if game_date > datetime.now():
                                processed_game = {
                                    'id': game.get('id'),
                                    'teams': {
                                        'home': {
                                            'id': game.get('teams', {}).get('home', {}).get('id'),
                                            'name': game.get('teams', {}).get('home', {}).get('name')
                                        },
                                        'away': {
                                            'id': game.get('teams', {}).get('visitors', {}).get('id'),
                                            'name': game.get('teams', {}).get('visitors', {}).get('name')
                                        }
                                    },
                                    'date': {
                                        'start': game_date_str,
                                    },
                                    'status': {
                                        'long': game.get('status', {}).get('long'),
                                        'short': game.get('status', {}).get('short')
                                    }
                                }
                                processed_games.append(processed_game)
                                
                                logging.info(f"Added game: {processed_game['teams']['home']['name']} vs "
                                           f"{processed_game['teams']['away']['name']} at {game_date_str}")
                        except Exception as e:
                            logging.error(f"Error processing game: {str(e)}")
                            continue
                    
                    if processed_games:
                        return processed_games
                        
            except Exception as e:
                logging.error(f"Error fetching schedule for {date_str}: {str(e)}")
                continue
        
        logging.info("No games found for today or tomorrow")
        return []
        
    except Exception as e:
        logging.error(f"Error in get_todays_schedule: {str(e)}")
        return []

def parse_game_time(time_str: str) -> Optional[datetime]:
    """Parse game time string to datetime object."""
    try:
        if not time_str:
            return None
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception as e:
        logging.error(f"Error parsing game time {time_str}: {str(e)}")
        return None

def process_scheduled_games(games: List[Dict], api_client: EnhancedNBAApiClient, predictor: LiveGamePredictor):
    """Process scheduled games and make predictions."""
    for game in games:
        try:
            # Get team IDs
            home_team = game.get('teams', {}).get('home', {})
            away_team = game.get('teams', {}).get('away', {})
            
            home_id = str(home_team.get('id', ''))
            away_id = str(away_team.get('id', ''))
            
            if not home_id or not away_id:
                logging.warning(f"Missing team ID for game {game.get('id')}")
                continue
            
            # Get team stats
            home_stats = api_client.get_team_stats_with_retry(home_id)
            away_stats = api_client.get_team_stats_with_retry(away_id)
            
            # Prepare game info
            game_info = {
                'gameId': game.get('id'),
                'home_team': home_team.get('name'),
                'away_team': away_team.get('name'),
                'home_stats': home_stats,
                'away_stats': away_stats,
                'scheduled_start': game.get('date', {}).get('start'),
                'current_period': 0,
                'clock': '12:00',
                'home_score': 0,
                'away_score': 0
            }
            
            # Make prediction
            prediction = predictor.predict_live_game(game_info)
            
            # Save prediction
            save_scheduled_prediction(game_info, prediction)
            
            # Display prediction summary
            display_prediction_summary(game_info, prediction)
            
        except Exception as e:
            logging.error(f"Error processing game {game.get('id')}: {str(e)}")
            continue

def generate_score_prediction(home_stats: Dict, away_stats: Dict) -> Dict:
    """Generate score prediction ranges based on team statistics."""
    try:
        # Get average points for each team
        home_avg = float(home_stats['statistics'][0].get('points', 100))
        away_avg = float(away_stats['statistics'][0].get('points', 100))
        
        # Calculate ranges (±10% of average)
        home_margin = home_avg * 0.10
        away_margin = away_avg * 0.10
        
        return {
            'home_low': int(home_avg - home_margin),
            'home_high': int(home_avg + home_margin),
            'away_low': int(away_avg - away_margin),
            'away_high': int(away_avg + away_margin)
        }
    except Exception as e:
        logging.error(f"Error generating score prediction: {str(e)}")
        return {
            'home_low': 0,
            'home_high': 0,
            'away_low': 0,
            'away_high': 0
        }

def save_scheduled_prediction(game_info: Dict, prediction: Dict):
    """Save prediction for scheduled games with complete structure."""
    try:
        timestamp = datetime.now()
        
        # Calculate win probability
        home_win_prob = prediction['adjusted_prediction']
        win_probability = home_win_prob if home_win_prob > 0.5 else (1 - home_win_prob)
        predicted_winner = (
            game_info['home_team'] 
            if home_win_prob > 0.5 
            else game_info['away_team']
        )
        
        # Generate score prediction
        score_prediction = generate_score_prediction(
            game_info['home_stats'],
            game_info['away_stats']
        )
        
        result = {
            'timestamp': timestamp.isoformat(),
            'game_info': {
                'id': game_info['gameId'],
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team'],
                'scheduled_start': game_info['scheduled_start'],
                'period': 0,
                'clock': '12:00',
                'score': {
                    'home': 0,
                    'away': 0
                }
            },
            'prediction': {
                'base': float(prediction['base_prediction']),
                'adjusted': float(prediction['adjusted_prediction']),
                'predicted_winner': predicted_winner,
                'win_probability': float(win_probability),
                'confidence_level': get_confidence_level(win_probability),
                'model_predictions': prediction.get('model_predictions', {}),
                'score_prediction': score_prediction
            }
        }
        
        directory = 'predictions/scheduled'
        os.makedirs(directory, exist_ok=True)
        
        filename = f'{directory}/game_{game_info["gameId"]}_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
            
        logging.debug(f"Saved scheduled prediction to {filename}")
        return result
        
    except Exception as e:
        logging.error(f"Error saving scheduled prediction: {str(e)}")
        return None

def display_prediction_summary(game_info: Dict, prediction: Dict):
    """Display a summary of the prediction."""
    try:
        home_win_prob = prediction['adjusted_prediction']
        away_win_prob = 1 - home_win_prob
        
        # Generate score prediction
        score_pred = generate_score_prediction(
            game_info['home_stats'],
            game_info['away_stats']
        )
        
        logging.info(f"""
        Prediction Summary:
        {game_info['home_team']} vs {game_info['away_team']}
        Scheduled Start: {game_info['scheduled_start']}
        
        Win Probabilities:
        - {game_info['home_team']}: {home_win_prob:.1%}
        - {game_info['away_team']}: {away_win_prob:.1%}
        
        Predicted Score Ranges:
        - {game_info['home_team']}: {score_pred['home_low']}-{score_pred['home_high']}
        - {game_info['away_team']}: {score_pred['away_low']}-{score_pred['away_high']}
        
        Predicted Winner: {game_info['home_team'] if home_win_prob > 0.5 else game_info['away_team']}
        Confidence: {max(home_win_prob, away_win_prob):.1%}
        """)
        
    except Exception as e:
        logging.error(f"Error displaying prediction summary: {str(e)}")


