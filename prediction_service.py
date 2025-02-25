# prediction_service.py

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timezone
from nba_stats import NBA_TEAM_STATS  # Import the stats
import json
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier


class NBAPredictor:
    def __init__(self, models_path: str = None):
        """Initialize the NBA predictor."""
        self.feature_names = [
            'wins', 'losses', 'points_per_game', 'points_allowed',
            'field_goal_pct', 'three_point_pct', 'win_streak'
        ]
        
    def prepare_features(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Prepare features for prediction."""
        try:
            # Calculate basic stats
            home_total_games = home_stats['wins'] + home_stats['losses']
            away_total_games = away_stats['wins'] + away_stats['losses']
            
            home_win_pct = home_stats['wins'] / home_total_games if home_total_games > 0 else 0.5
            away_win_pct = away_stats['wins'] / away_total_games if away_total_games > 0 else 0.5
            
            # Prepare features dictionary
            features = {
                'home_win_pct': home_win_pct,
                'away_win_pct': away_win_pct,
                'home_ppg': home_stats['points_per_game'],
                'away_ppg': away_stats['points_per_game'],
                'home_papg': home_stats['points_allowed'],
                'away_papg': away_stats['points_allowed'],
                'home_off_rtg': home_stats['offensive_rating'],
                'away_off_rtg': away_stats['offensive_rating'],
                'home_def_rtg': home_stats['defensive_rating'],
                'away_def_rtg': away_stats['defensive_rating']
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error preparing features: {str(e)}", exc_info=True)
            return None
            
    def predict_game(self, home_stats: Dict, away_stats: Dict) -> Tuple[str, float]:
        """Make prediction using statistical analysis."""
        try:
            if not home_stats or not away_stats:
                return ('unknown', 50.0)
                
            # Team strength factors
            home_strength = (
                home_stats['offensive_rating'] / 100 +
                (100 - home_stats['defensive_rating']) / 100 +
                home_stats['wins'] / (home_stats['wins'] + home_stats['losses'])
            )
            
            away_strength = (
                away_stats['offensive_rating'] / 100 +
                (100 - away_stats['defensive_rating']) / 100 +
                away_stats['wins'] / (away_stats['wins'] + away_stats['losses'])
            )
            
            # Calculate win probability
            base_prob = 50 + (home_strength - away_strength) * 20
            
            # Add home court advantage
            home_advantage = 10
            win_prob = base_prob + home_advantage
            
            # Ensure probability is within bounds
            win_prob = max(20, min(80, win_prob))
            
            # Determine winner based on team strengths
            if home_strength > away_strength:
                return ('home', win_prob)
            else:
                return ('away', win_prob)
                
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}", exc_info=True)
            return ('unknown', 50.0)

    def predict_score_range(self, team_stats: Dict, opponent_stats: Dict, is_home: bool) -> Tuple[int, int]:
        """Predict score range for a team."""
        try:
            # Get team's scoring stats
            team_ppg = team_stats['points_per_game']
            opp_defense = opponent_stats['points_allowed']
            
            # Calculate base score prediction
            base_score = (team_ppg + opp_defense) / 2
            
            # Add home court advantage if applicable
            if is_home:
                base_score += 2  # Home teams typically score 2-3 more points
                
            # Add random variation
            variation = 5
            min_score = int(base_score - variation)
            max_score = int(base_score + variation)
            
            return (min_score, max_score)
            
        except Exception as e:
            logging.error(f"Error predicting score range: {str(e)}", exc_info=True)
            return (95, 105)  # Fallback range

    def save_prediction(
        self,
        game_id: str,
        prediction_data: Dict[str, Any],
        timestamp: datetime = None
    ) -> None:
        """Save prediction to file with timestamp."""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            filename = f"predictions/game_{game_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(
                    {
                        'timestamp': timestamp.isoformat(),
                        'game_id': game_id,
                        'predictions': prediction_data
                    },
                    f,
                    indent=4
                )
                
            logging.info(f"Saved prediction to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving prediction: {str(e)}")

    def predict_live_game(self, game_stats: Dict) -> Dict[str, Any]:
        """Make predictions for a live game with current statistics."""
        try:
            home_stats = game_stats.get('home_team', {})
            away_stats = game_stats.get('away_team', {})
            
            # Get base prediction
            win_prob, model_preds = self.predict_game(home_stats, away_stats)
            
            # Adjust for current game state
            current_score_diff = (
                float(game_stats.get('home_score', 0)) - 
                float(game_stats.get('away_score', 0))
            )
            period = int(game_stats.get('period', 1))
            time_remaining = self._parse_game_clock(game_stats.get('clock', '12:00'))
            
            # Adjust win probability based on current game state
            adjusted_prob = self._adjust_win_probability(
                win_prob,
                current_score_diff,
                period,
                time_remaining
            )
            
            return {
                'base_prediction': float(win_prob),
                'adjusted_prediction': float(adjusted_prob),
                'model_predictions': model_preds,
                'game_state': {
                    'period': period,
                    'time_remaining': time_remaining,
                    'score_difference': current_score_diff
                }
            }
            
        except Exception as e:
            logging.error(f"Error in live game prediction: {str(e)}")
            raise

    def _parse_game_clock(self, clock_str: str) -> float:
        """Convert game clock string to minutes remaining."""
        try:
            if not clock_str:
                return 12.0
                
            minutes, seconds = map(float, clock_str.split(':'))
            return minutes + (seconds / 60)
        except Exception as e:
            logging.error(f"Error parsing game clock: {str(e)}")
            return 12.0

    def _adjust_win_probability(
        self,
        base_prob: float,
        score_diff: float,
        period: int,
        time_remaining: float
    ) -> float:
        """Adjust win probability based on current game state."""
        try:
            # Calculate time factor (0 to 1, where 1 is game start)
            total_time = 48.0  # Regular game length in minutes
            elapsed_time = ((period - 1) * 12) + (12 - time_remaining)
            time_factor = 1 - (elapsed_time / total_time)
            
            # Score impact increases as game progresses
            score_impact = (score_diff / 10) * (1 - time_factor)
            
            # Combine base prediction with current game state
            adjusted_prob = (base_prob * time_factor) + (
                0.5 + score_impact
            ) * (1 - time_factor)
            
            # Ensure probability stays between 0 and 1
            return max(0.0, min(1.0, adjusted_prob))
            
        except Exception as e:
            logging.error(f"Error adjusting win probability: {str(e)}")
            return base_prob

    def predict_game_with_context(self, game_info: Dict) -> Dict[str, Any]:
        """Enhanced prediction with team context and injuries."""
        try:
            # Get base prediction
            base_prediction, model_predictions = self.predict_game(
                game_info['home_stats'],
                game_info['away_stats']
            )
            
            # Calculate context factors
            injury_factor = self._calculate_injury_impact(game_info)
            conference_factor = self._calculate_conference_factor(game_info)
            division_factor = self._calculate_division_factor(game_info)
            
            # Adjust prediction
            adjusted_prediction = self._adjust_prediction_with_context(
                base_prediction,
                injury_factor,
                conference_factor,
                division_factor
            )
            
            return {
                'base_prediction': base_prediction,
                'adjusted_prediction': adjusted_prediction,
                'model_predictions': model_predictions,
                'context_factors': {
                    'injury_impact': injury_factor,
                    'conference_factor': conference_factor,
                    'division_factor': division_factor
                }
            }
            
        except Exception as e:
            logging.error(f"Error in contextual prediction: {str(e)}")
            raise

    def _calculate_injury_impact(self, game_info: Dict) -> float:
        """Calculate impact of injuries on prediction."""
        try:
            home_injuries = game_info['home_team']['injuries']
            away_injuries = game_info['away_team']['injuries']
            
            # Calculate impact scores
            def get_impact_score(injuries):
                return sum(
                    1.0 if injury['status'] == 'Out' else
                    0.5 if injury['status'] == 'Questionable' else
                    0.25 if injury['status'] == 'Probable' else 0
                    for injury in injuries
                )
            
            home_impact = get_impact_score(home_injuries)
            away_impact = get_impact_score(away_injuries)
            
            # Return normalized differential
            return (away_impact - home_impact) / max(10, home_impact + away_impact)
            
        except Exception as e:
            logging.error(f"Error calculating injury impact: {str(e)}")
            return 0.0

    def _calculate_conference_factor(self, game_info: Dict) -> float:
        """Calculate conference strength factor."""
        try:
            home_conf = game_info['home_team']['info']['conference']
            away_conf = game_info['away_team']['info']['conference']
            
            # Conference strength factors (could be updated based on current season data)
            conf_strength = {
                'Eastern': 0.48,  # Example values
                'Western': 0.52
            }
            
            return conf_strength.get(home_conf, 0.5) - conf_strength.get(away_conf, 0.5)
            
        except Exception as e:
            logging.error(f"Error calculating conference factor: {str(e)}")
            return 0.0

    def _calculate_division_factor(self, game_info: Dict) -> float:
        """Calculate division rivalry factor."""
        try:
            home_div = game_info['home_team']['info']['division']
            away_div = game_info['away_team']['info']['division']
            
            # Add small boost for division games
            return 0.05 if home_div == away_div else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating division factor: {str(e)}")
            return 0.0

    def _adjust_prediction_with_context(
        self,
        base_pred: float,
        injury_factor: float,
        conference_factor: float,
        division_factor: float
    ) -> float:
        """Combine all factors to adjust prediction."""
        try:
            # Weight factors
            injury_weight = 0.15
            conference_weight = 0.10
            division_weight = 0.05
            
            # Calculate adjustment
            adjustment = (
                injury_factor * injury_weight +
                conference_factor * conference_weight +
                division_factor * division_weight
            )
            
            # Apply adjustment to base prediction
            adjusted_pred = base_pred + adjustment
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, adjusted_pred))
            
        except Exception as e:
            logging.error(f"Error adjusting prediction with context: {str(e)}")
            return base_pred

    def adjust_predictions(self, home_stats, away_stats, raw_home_prob):
        """Adjust predictions based on various factors to reduce home team bias."""
        try:
            # Calculate team strength indicators
            home_strength = self._calculate_team_strength(home_stats)
            away_strength = self._calculate_team_strength(away_stats)
            
            # Calculate form-based adjustment
            home_form = self._calculate_team_form(home_stats)
            away_form = self._calculate_team_form(away_stats)
            
            # Calculate head-to-head adjustment
            h2h_factor = self._calculate_h2h_factor(home_stats, away_stats)
            
            # Base adjustment factors
            strength_diff = (away_strength - home_strength) * 0.15  # Reduce impact of strength difference
            form_diff = (away_form - home_form) * 0.1  # Reduce impact of form
            
            # Calculate final adjustment
            total_adjustment = strength_diff + form_diff + h2h_factor
            
            # Apply adjustment to raw probability
            adjusted_prob = raw_home_prob + total_adjustment
            
            # Ensure probability stays within bounds
            adjusted_prob = max(0.3, min(0.7, adjusted_prob))  # Cap at 30-70% range
            
            return adjusted_prob
        except Exception as e:
            logging.error(f"Error in prediction adjustment: {str(e)}")
            return raw_home_prob

    def _calculate_team_strength(self, stats):
        """Calculate overall team strength based on season statistics."""
        try:
            stats_obj = stats.get('statistics', [{}])[0]
            
            # Get key statistics
            wins = float(stats_obj.get('wins', 0))
            losses = float(stats_obj.get('losses', 0))
            points = float(stats_obj.get('points', 0))
            points_allowed = float(stats_obj.get('pointsAllowed', 0))
            
            # Calculate win percentage
            total_games = wins + losses
            win_pct = wins / total_games if total_games > 0 else 0.5
            
            # Calculate point differential per game
            point_diff = (points - points_allowed) / max(total_games, 1)
            
            # Normalize point differential to 0-1 scale
            norm_point_diff = (point_diff + 20) / 40  # Assuming max point diff is ±20
            norm_point_diff = max(0, min(1, norm_point_diff))
            
            # Combine factors with higher weight on win percentage
            strength = (win_pct * 0.7) + (norm_point_diff * 0.3)
            
            return strength
            
        except Exception as e:
            logging.error(f"Error calculating team strength: {str(e)}")
            return 0.5

    def _calculate_team_form(self, stats):
        """Calculate team's recent form."""
        try:
            stats_obj = stats.get('statistics', [{}])[0]
            
            # Get streak information
            streak = float(stats_obj.get('streak', 0))
            
            # Normalize streak impact
            form_factor = min(max(streak * 0.02, -0.1), 0.1)  # Cap at ±10%
            
            return form_factor
        except Exception as e:
            logging.error(f"Error calculating team form: {str(e)}")
            return 0.0

    def _calculate_h2h_factor(self, home_stats, away_stats):
        """Calculate head-to-head adjustment factor."""
        try:
            # Get win percentages
            home_stats_obj = home_stats.get('statistics', [{}])[0]
            away_stats_obj = away_stats.get('statistics', [{}])[0]
            
            home_wins = float(home_stats_obj.get('wins', 0))
            home_losses = float(home_stats_obj.get('losses', 0))
            away_wins = float(away_stats_obj.get('wins', 0))
            away_losses = float(away_stats_obj.get('losses', 0))
            
            # Calculate win percentages
            home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5
            away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
            
            # Calculate h2h factor based on win percentage difference
            h2h_factor = (away_win_pct - home_win_pct) * 0.1  # Small adjustment based on win percentage difference
            
            return h2h_factor
        except Exception as e:
            logging.error(f"Error calculating h2h factor: {str(e)}")
            return 0.0

    def predict(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Make prediction with adjusted probabilities."""
        try:
            # Get base team strengths
            home_strength = self._calculate_base_strength(home_stats)
            away_strength = self._calculate_base_strength(away_stats)
            
            # Get team statistics
            home_stats_obj = home_stats.get('statistics', [{}])[0]
            away_stats_obj = away_stats.get('statistics', [{}])[0]
            
            # Calculate offensive and defensive ratings
            home_off_rating = float(home_stats_obj.get('points', 0))
            home_def_rating = float(home_stats_obj.get('pointsAllowed', 0))
            away_off_rating = float(away_stats_obj.get('points', 0))
            away_def_rating = float(away_stats_obj.get('pointsAllowed', 0))
            
            # Calculate efficiency differentials
            home_net_rating = home_off_rating - home_def_rating
            away_net_rating = away_off_rating - away_def_rating
            
            # Get win-loss records
            home_record = self._get_team_record(home_stats)
            away_record = self._get_team_record(away_stats)
            
            # Calculate win percentage impact
            home_winpct = home_record['win_pct']
            away_winpct = away_record['win_pct']
            winpct_factor = (home_winpct - away_winpct) * 0.3
            
            # Calculate streak impact
            home_streak = float(home_stats_obj.get('streak', 0))
            away_streak = float(away_stats_obj.get('streak', 0))
            streak_factor = (home_streak - away_streak) * 0.05
            
            # Calculate base probability using multiple factors
            base_prob = 0.5  # Start at neutral
            base_prob += (home_net_rating - away_net_rating) * 0.02  # Net rating impact
            base_prob += winpct_factor  # Win percentage impact
            base_prob += streak_factor  # Streak impact
            base_prob += 0.03  # Home court advantage
            
            # Ensure probability is within bounds
            final_prob = max(0.35, min(0.65, base_prob))  # Limit extreme predictions
            
            # Calculate score predictions
            home_score_range = self._predict_score_range(home_stats)
            away_score_range = self._predict_score_range(away_stats)
            
            return {
                'home_probability': final_prob,
                'away_probability': 1 - final_prob,
                'home_score_range': home_score_range,
                'away_score_range': away_score_range,
                'prediction_details': {
                    'home_strength': home_strength,
                    'away_strength': away_strength,
                    'home_record': home_record,
                    'away_record': away_record,
                    'home_net_rating': home_net_rating,
                    'away_net_rating': away_net_rating,
                    'winpct_factor': winpct_factor,
                    'streak_factor': streak_factor
                }
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {
                'home_probability': 0.5,
                'away_probability': 0.5,
                'home_score_range': (95, 105),
                'away_score_range': (95, 105),
                'prediction_details': {}
            }

    def _calculate_base_strength(self, stats):
        """Calculate team's base strength."""
        try:
            stats_obj = stats.get('statistics', [{}])[0]
            
            # Get key statistics
            wins = float(stats_obj.get('wins', 0))
            losses = float(stats_obj.get('losses', 0))
            points = float(stats_obj.get('points', 0))
            points_allowed = float(stats_obj.get('pointsAllowed', 0))
            
            # Calculate win percentage
            total_games = wins + losses
            win_pct = wins / total_games if total_games > 0 else 0.5
            
            # Calculate point differential per game
            point_diff = (points - points_allowed) / max(total_games, 1)
            
            # Normalize point differential to 0-1 scale
            norm_point_diff = (point_diff + 20) / 40  # Assuming max point diff is ±20
            norm_point_diff = max(0, min(1, norm_point_diff))
            
            # Combine factors with higher weight on win percentage
            strength = (win_pct * 0.7) + (norm_point_diff * 0.3)
            
            return strength
            
        except Exception as e:
            logging.error(f"Error calculating base strength: {str(e)}")
            return 0.5

    def _get_team_record(self, stats):
        """Get team's win-loss record and win percentage."""
        try:
            stats_obj = stats.get('statistics', [{}])[0]
            wins = float(stats_obj.get('wins', 0))
            losses = float(stats_obj.get('losses', 0))
            
            total_games = wins + losses
            win_pct = wins / total_games if total_games > 0 else 0.5
            
            return {
                'wins': wins,
                'losses': losses,
                'win_pct': win_pct
            }
        except Exception as e:
            logging.error(f"Error getting team record: {str(e)}")
            return {'wins': 0, 'losses': 0, 'win_pct': 0.5}

    def _predict_score_range(self, stats):
        """Predict score range based on team statistics."""
        try:
            stats_obj = stats.get('statistics', [{}])[0]
            
            # Get scoring statistics
            avg_points = float(stats_obj.get('points', 100))
            avg_points_allowed = float(stats_obj.get('pointsAllowed', 100))
            fgp = float(stats_obj.get('fgp', 45))  # Field goal percentage
            pace = float(stats_obj.get('pace', 100))  # Team's pace factor
            
            # Calculate expected points using multiple factors
            expected_points = (
                avg_points * 0.5 +  # Historical scoring
                (fgp * 2) * 0.3 +  # Shooting efficiency impact
                (pace / 100 * 100) * 0.2  # Pace impact
            )
            
            # Calculate variance based on team's consistency
            variance = abs(avg_points - avg_points_allowed) * 0.15
            
            # Set score range with dynamic bounds
            lower_bound = int(max(expected_points - variance, 85))
            upper_bound = int(min(expected_points + variance, 140))
            
            logging.info(f"""
            Score Range Prediction:
            - Base PPG: {avg_points:.1f}
            - Opp Points Allowed: {avg_points_allowed:.1f}
            - Expected Points: {expected_points:.1f}
            - Variance: {variance:.1f}
            - Final Range: {lower_bound}-{upper_bound}
            """)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logging.error(f"Error predicting score range: {str(e)}")
            # Return more varied default ranges based on league averages
            return (95, 105)

    def _weighted_average(self, predictions: Dict[str, float]) -> float:
        """Calculate weighted average of predictions."""
        try:
            weighted_sum = 0
            weight_sum = 0
            
            for model_name, prob in predictions.items():
                if model_name in self.model_weights:
                    weight = self.model_weights[model_name]
                    weighted_sum += prob * weight
                    weight_sum += weight
            
            # Normalize the prediction
            if weight_sum > 0:
                avg_pred = weighted_sum / weight_sum
                # Apply slight regression to the mean to reduce extreme predictions
                return 0.7 * avg_pred + 0.3 * 0.5
            return 0.5
            
        except Exception as e:
            logging.error(f"Error in weighted average calculation: {str(e)}")
            return 0.5

    def generate_prediction(self, game):
        """Generate prediction with proper error handling and logging."""
        try:
            home_team = game['teams']['home']
            away_team = game['teams']['away']
            
            # Get team statistics with retries
            home_stats = self.nba_client.get_team_stats(home_team['name'])
            away_stats = self.nba_client.get_team_stats(away_team['name'])
            
            logging.info(f"Raw home team stats for {home_team['name']}: {json.dumps(home_stats, indent=2)}")
            logging.info(f"Raw away team stats for {away_team['name']}: {json.dumps(away_stats, indent=2)}")
            
            if not home_stats or not away_stats:
                logging.error(f"Missing stats for {home_team['name']} or {away_team['name']}")
                return None
            
            # Prepare features for prediction
            features = self.prepare_features(home_stats, away_stats)
            features['home_team'] = home_team['name']
            features['away_team'] = away_team['name']
            
            logging.info(f"Prepared features: {json.dumps(features, indent=2)}")
            
            # Get prediction from ML models
            winner, probability = self.predict_game(features)
            
            logging.info(f"Initial prediction: Winner={winner}, Probability={probability}")
            
            # Calculate score ranges
            home_score_range = self.predict_score_range(home_stats, away_stats, is_home=True)
            away_score_range = self.predict_score_range(away_stats, home_stats, is_home=False)
            
            logging.info(f"Score ranges: Home={home_score_range}, Away={away_score_range}")
            
            return {
                'id': str(uuid.uuid4()),
                'home_team': home_team['name'],
                'away_team': away_team['name'],
                'predicted_winner': winner,
                'win_probability': probability,
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

    def calculate_score_range(self, team_stats: Dict) -> Tuple[int, int]:
        """Calculate the predicted score range for a team."""
        try:
            # Base score from points per game
            base_score = 110  # League average points per game
            
            # Adjust based on offensive rating relative to league average (100)
            offensive_factor = team_stats['offensive_rating'] / 100
            
            # Calculate predicted score
            predicted_score = base_score * offensive_factor
            
            # Add variance for range
            variance = 10  # Points of variance for the range
            min_score = max(85, int(predicted_score - variance))  # Minimum reasonable NBA score
            max_score = min(135, int(predicted_score + variance))  # Maximum reasonable NBA score
            
            # Log the calculation
            logging.info(f"""
            Score Range Calculation for {team_stats.get('team_name', 'Unknown Team')}:
            - Base Score: {base_score}
            - Offensive Rating: {team_stats['offensive_rating']}
            - Offensive Factor: {offensive_factor:.2f}
            - Predicted Score: {predicted_score:.1f}
            - Final Range: {min_score}-{max_score}
            """)
            
            return (min_score, max_score)
            
        except Exception as e:
            logging.error(f"Error calculating score range: {str(e)}")
            return (100, 110)  # Default reasonable NBA score range
