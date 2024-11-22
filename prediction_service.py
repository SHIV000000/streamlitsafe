# prediction_service.py

import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
import json
import os
import pickle


class NBAPredictor:
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.scaler = None
        self.feature_names = None
        self.load_scaler()
        self.models = {}
        self._load_models()
        self.model_weights = {
            'random_forest': 0.35,
            'xgboost': 0.35,
            'svm': 0.15,
            'lstm': 0.20,
            'gru': 0.15
        }

    def load_scaler(self):
        """Load the scaler from disk."""
        try:
            scaler_path = os.path.join(self.models_path, 'scaler_20241111_040330.joblib')
            self.scaler = joblib.load(scaler_path)
            
            # Initialize feature names from scaler if available
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
                logging.info("Loaded feature names from scaler")
            else:
                self.feature_names = self._get_default_feature_names()
                logging.warning("Scaler does not have feature names; using default feature names")
            
            # Create a mapping between feature names and indices
            self.feature_indices = {name: idx for idx, name in enumerate(self.feature_names)}
            
            logging.info("Loaded scaler successfully")
            
            # Compare generated feature names with expected feature names
            self._compare_feature_names()
            
        except Exception as e:
            logging.error(f"Error loading scaler: {str(e)}")
            raise

    def _compare_feature_names(self):
        """Compare generated feature names with expected feature names from the scaler."""
        generated_features = set(self._get_default_feature_names())
        expected_features = set(self.feature_names)
        
        missing_in_generated = expected_features - generated_features
        extra_in_generated = generated_features - expected_features
        
        if missing_in_generated:
            logging.warning(f"Features missing in generated features: {missing_in_generated}")
        if extra_in_generated:
            logging.warning(f"Extra features in generated features: {extra_in_generated}")
        
        return missing_in_generated, extra_in_generated

    def _get_default_feature_names(self):
        """Generate default feature names."""
        # Base stats used in training (9 stats)
        base_stats = [
            'points', 'fgp', 'tpp', 'ftp', 'totReb', 
            'assists', 'steals', 'blocks', 'turnovers'  # Removed 'plusMinus'
        ]
        
        windows = [3, 5, 10]
        feature_names = []
        
        # Add basic stat features (9 stats * 3 windows * 3 types = 81 features)
        for stat in base_stats:
            for window in windows:
                feature_names.append(f'avg_{stat}_diff_{window}')
                feature_names.append(f'avg_{stat}_home_{window}')
                feature_names.append(f'avg_{stat}_away_{window}')
        
        # Add win percentage features (3 windows * 3 types = 9 features)
        for window in windows:
            feature_names.append(f'win_pct_diff_{window}')
            feature_names.append(f'home_win_pct_{window}')
            feature_names.append(f'away_win_pct_{window}')
        
        # Add streak features (2 features)
        streak_features = ['current_streak']  # Removed 'home_streak' and 'away_streak'
        for feature in streak_features:
            feature_names.append(f'{feature}_home')
            feature_names.append(f'{feature}_away')
        
        # Add form features (6 features)
        for window in [3, 5]:  # Only using windows 3 and 5 for form
            feature_names.append(f'form_diff_{window}')
            feature_names.append(f'home_form_{window}')
            feature_names.append(f'away_form_{window}')
        
        # Add rest days and games played (5 features)
        feature_names.extend([
            'rest_days_diff',
            'home_rest_days',
            'away_rest_days',
            'home_games_played',
            'away_games_played'
        ])
        
        # Total features:
        # 81 (basic stats) + 9 (win_pct) + 2 (streak) + 6 (form) + 5 (rest/games) = 103 features
        
        return feature_names

    def _load_models(self):
        """Load all saved models."""
        try:
            # Load traditional ML models
            model_files = {
                'random_forest': 'random_forest_20241111_040330.joblib',
                'xgboost': 'xgboost_20241111_040330.joblib',
                'svm': 'svm_20241111_040330.joblib'
            }

            # Add neural network models
            nn_models = {
                'lstm': 'lstm_20241111_040330.h5',
                'gru': 'gru_20241111_040330.h5'
            }

            # Load traditional models
            for model_name, file_name in model_files.items():
                try:
                    self.models[model_name] = joblib.load(os.path.join(self.models_path, file_name))
                    logging.info(f"Loaded {model_name} model successfully")
                except Exception as e:
                    logging.error(f"Error loading {model_name} model: {str(e)}")

            # Load neural network models
            for model_name, file_name in nn_models.items():
                try:
                    model_path = os.path.join(self.models_path, file_name)
                    self.models[model_name] = tf.keras.models.load_model(model_path)
                    logging.info(f"Loaded {model_name} model successfully")
                except Exception as e:
                    logging.error(f"Error loading {model_name} model: {str(e)}")

        except Exception as e:
            logging.error(f"Error in model loading: {str(e)}")
            raise

    def prepare_features(self, home_stats: Dict, away_stats: Dict) -> np.ndarray:
        """Prepare features with validation."""
        try:
            features_dict = {}
            home_stats_obj = home_stats.get('statistics', [{}])[0]
            away_stats_obj = away_stats.get('statistics', [{}])[0]
            
            # Add default values for missing stats
            default_stats = {
                'points': 0, 'fgp': 0, 'tpp': 0, 'ftp': 0,
                'totReb': 0, 'assists': 0, 'steals': 0,
                'blocks': 0, 'turnovers': 0
            }
            
            # Update with defaults
            home_stats_obj = {**default_stats, **home_stats_obj}
            away_stats_obj = {**default_stats, **away_stats_obj}
            
            # Process features
            for feature_name in self.feature_names:
                features_dict[feature_name] = self._calculate_feature_value(
                    feature_name, home_stats_obj, away_stats_obj
                )
            
            features = np.array([[features_dict[name] for name in self.feature_names]])
            return self.scaler.transform(features)
            
        except Exception as e:
            logging.error(f"Feature preparation error: {str(e)}")
            # Return zero-filled array as fallback
            return np.zeros((1, len(self.feature_names)))

    def predict_game(self, home_stats: Dict, away_stats: Dict) -> Tuple[float, Dict[str, float]]:
        """Make prediction for a game."""
        try:
            # Use prepare_features directly instead of _prepare_feature_dict
            features = self.prepare_features(home_stats, away_stats)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                try:
                    if isinstance(model, tf.keras.Model):
                        # Reshape features for neural network models
                        reshaped_features = features.reshape((1, 1, features.shape[1]))
                        pred = model.predict(reshaped_features, verbose=0)[0][0]
                    else:
                        # For traditional ML models
                        pred = model.predict_proba(features)[0][1]
                    predictions[name] = float(pred)
                except Exception as e:
                    logging.error(f"Error getting prediction from {name} model: {str(e)}")
                    continue

            if not predictions:
                raise ValueError("No valid predictions obtained from any model")

            # Calculate weighted ensemble prediction
            ensemble_pred = sum(
                pred * self.model_weights[name]
                for name, pred in predictions.items()
            ) / sum(
                self.model_weights[name]
                for name in predictions.keys()
            )

            return ensemble_pred, predictions

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise

    def _create_feature_array(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Create numpy array from features dictionary."""
        try:
            features = np.array([[
                features_dict[feature_name] 
                for feature_name in self.feature_names
            ]])
            return features
        except Exception as e:
            logging.error(f"Error creating feature array: {str(e)}")
            raise

    def _get_model_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all models."""
        predictions = {}
        for name, model in self.models.items():
            try:
                if isinstance(model, tf.keras.Model):
                    # Reshape features for neural network models
                    reshaped_features = features.reshape((1, 1, features.shape[1]))
                    pred = model.predict(reshaped_features, verbose=0)[0][0]
                else:
                    # For traditional ML models
                    pred = model.predict_proba(features)[0][1]
                predictions[name] = float(pred)
            except Exception as e:
                logging.error(f"Error getting prediction from {name} model: {str(e)}")
                continue
        return predictions

    def _calculate_ensemble_prediction(self, predictions: Dict[str, float]) -> float:
        """Calculate weighted ensemble prediction."""
        if not predictions:
            raise ValueError("No predictions available for ensemble")
            
        weighted_sum = sum(
            pred * self.model_weights[name]
            for name, pred in predictions.items()
        )
        weight_sum = sum(
            self.model_weights[name]
            for name in predictions.keys()
        )
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.5

    def _calculate_offensive_rating(self, stats: Dict) -> float:
        """Calculate offensive rating."""
        try:
            points = float(stats.get('points', 0))
            fga = float(stats.get('fieldGoalsAttempted', 0))
            turnovers = float(stats.get('turnovers', 0))
            offReb = float(stats.get('reboundsOffensive', 0))
            
            possessions = fga - offReb + turnovers
            if possessions <= 0:
                return 0
            
            return (points * 100) / possessions
        except Exception as e:
            logging.error(f"Error calculating offensive rating: {str(e)}")
            return 0

    def _calculate_defensive_rating(self, stats: Dict) -> float:
        """Calculate defensive rating."""
        try:
            points_allowed = float(stats.get('pointsAllowed', 0))
            opp_fga = float(stats.get('opponentFieldGoalsAttempted', 0))
            opp_turnovers = float(stats.get('opponentTurnovers', 0))
            opp_offReb = float(stats.get('opponentReboundsOffensive', 0))
            
            possessions = opp_fga - opp_offReb + opp_turnovers
            if possessions <= 0:
                return 0
            
            return (points_allowed * 100) / possessions
        except Exception as e:
            logging.error(f"Error calculating defensive rating: {str(e)}")
            return 0

    def _calculate_pace_factor(self, stats: Dict) -> float:
        """Calculate pace factor."""
        try:
            team_possessions = (
                float(stats.get('fieldGoalsAttempted', 0)) +
                0.4 * float(stats.get('freeThrowsAttempted', 0)) -
                1.07 * (float(stats.get('reboundsOffensive', 0)) / 
                        (float(stats.get('reboundsOffensive', 0)) + 
                         float(stats.get('reboundsDefensive', 0)))) *
                (float(stats.get('fieldGoalsAttempted', 0)) - 
                 float(stats.get('fieldGoalsMade', 0))) +
                float(stats.get('turnovers', 0))
            )
            
            minutes_played = float(stats.get('minutes', 48))
            if minutes_played <= 0:
                return 0
                
            return 48 * (team_possessions / minutes_played)
        except Exception as e:
            logging.error(f"Error calculating pace factor: {str(e)}")
            return 0

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

    def _validate_features(self, features_dict: Dict[str, float]) -> bool:
        """Validate that all required features are present."""
        missing_features = set(self.feature_names) - set(features_dict.keys())
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
            return False
        return True

    def _check_scaler_compatibility(self):
        """Check if scaler is compatible with current scikit-learn version."""
        try:
            from sklearn import __version__ as sk_version
            
            # Get scaler attributes
            scaler_attrs = dir(self.scaler)
            
            if 'feature_names_in_' not in scaler_attrs:
                logging.warning("Scaler might be from an older scikit-learn version")
                
            return True
        except Exception as e:
            logging.error(f"Error checking scaler compatibility: {str(e)}")
            return False

    def _scale_features_safe(self, features: np.ndarray) -> np.ndarray:
        """Scale features with fallback mechanism."""
        try:
            return self.scaler.transform(features)
        except Exception as e:
            logging.warning(f"Error using scaler: {str(e)}")
            # Fallback to manual scaling if needed
            mean = self.scaler.mean_ if hasattr(self.scaler, 'mean_') else 0
            scale = self.scaler.scale_ if hasattr(self.scaler, 'scale_') else 1
            return (features - mean) / scale

    def _calculate_feature_value(
        self, 
        feature_name: str, 
        home_stats: Dict, 
        away_stats: Dict
    ) -> float:
        """Calculate value for a specific feature."""
        try:
            # Extract base stat name and window size if present
            parts = feature_name.split('_')
            
            # Handle different feature types
            if 'avg_' in feature_name:
                stat_name = parts[1]
                window = int(parts[-1])
                
                if 'home_' in feature_name:
                    return float(home_stats.get(stat_name, 0))
                elif 'away_' in feature_name:
                    return float(away_stats.get(stat_name, 0))
                else:
                    return float(home_stats.get(stat_name, 0)) - float(away_stats.get(stat_name, 0))
                    
            elif 'win_pct' in feature_name:
                window = int(parts[-1]) if parts[-1].isdigit() else 10
                home_games = float(home_stats.get('games', 1))
                away_games = float(away_stats.get('games', 1))
                
                if 'home_' in feature_name:
                    return float(home_stats.get('wins', 0)) / home_games if home_games > 0 else 0
                elif 'away_' in feature_name:
                    return float(away_stats.get('wins', 0)) / away_games if away_games > 0 else 0
                else:
                    home_pct = float(home_stats.get('wins', 0)) / home_games if home_games > 0 else 0
                    away_pct = float(away_stats.get('wins', 0)) / away_games if away_games > 0 else 0
                    return home_pct - away_pct
                    
            elif 'form' in feature_name:
                # Placeholder for form calculation
                return 0.0
                
            elif 'streak' in feature_name:
                # Placeholder for streak calculation
                return 0.0
                
            elif 'rest_days' in feature_name:
                # Placeholder for rest days calculation
                return 0.0
                
            elif 'games_played' in feature_name:
                if 'home_' in feature_name:
                    return float(home_stats.get('games', 0))
                else:
                    return float(away_stats.get('games', 0))
                    
            else:
                return float(home_stats.get(feature_name, 0)) - float(away_stats.get(feature_name, 0))
                
        except Exception as e:
            logging.warning(f"Error calculating feature {feature_name}: {str(e)}")
            return 0.0

    def _validate_feature_count(self, features: np.ndarray) -> bool:
        """Validate that the feature count matches the scaler."""
        try:
            # Get expected feature count from scaler
            if hasattr(self.scaler, 'feature_names_in_'):
                expected_features = len(self.scaler.feature_names_in_)
            elif hasattr(self.scaler, 'n_features_in_'):
                expected_features = self.scaler.n_features_in_
            else:
                # For older versions, use the scale_ attribute length
                expected_features = len(self.scaler.scale_)
            
            actual_features = features.shape[1]
            
            if actual_features != expected_features:
                logging.error(
                    f"Feature count mismatch. Expected {expected_features}, got {actual_features}. "
                    f"Feature names: {self.feature_names}"
                )
                return False
            return True
            
        except Exception as e:
            logging.error(f"Error validating feature count: {str(e)}")
            # If validation fails, return True to continue with prediction
            return True

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





