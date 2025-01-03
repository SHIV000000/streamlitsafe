# prediction_service.py

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime
import json
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier


class NBAPredictor:
    def __init__(self, models_path: str):
        self.models_path = models_path
        self.scaler = None
        self.feature_names = None
        self.load_scaler()
        self.models = {}
        self._load_models()
        self.model_weights = {
            'random_forest': 0.4,
            'xgboost': 0.4,
            'svm': 0.2
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

            # Initialize models with default parameters
            default_models = {
                'random_forest': joblib.load(os.path.join(self.models_path, 'random_forest_20241111_040330.joblib')),
                'xgboost': joblib.load(os.path.join(self.models_path, 'xgboost_20241111_040330.joblib')),
                'svm': joblib.load(os.path.join(self.models_path, 'svm_20241111_040330.joblib'))
            }

            # Load traditional models
            for model_name, file_name in model_files.items():
                try:
                    self.models[model_name] = joblib.load(os.path.join(self.models_path, file_name))
                    logging.info(f"Loaded {model_name} model successfully")
                except Exception as e:
                    logging.error(f"Error loading {model_name} model: {str(e)}")
                    # Use the default model if loading fails
                    if model_name in default_models:
                        self.models[model_name] = default_models[model_name]
                        logging.info(f"Using default {model_name} model")

            # Update model weights to remove gradient boosting
            self.model_weights = {
                'random_forest': 0.4,
                'xgboost': 0.4,
                'svm': 0.2
            }

        except Exception as e:
            logging.error(f"Error in model loading: {str(e)}")
            raise

    def prepare_features(self, home_stats: Dict, away_stats: Dict) -> np.ndarray:
        """Prepare features with validation."""
        try:
            features_dict = {}
            
            # Get base statistics
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
            
            # Calculate offensive and defensive ratings
            home_off_rating = self._calculate_offensive_rating(home_stats_obj)
            away_off_rating = self._calculate_offensive_rating(away_stats_obj)
            home_def_rating = self._calculate_defensive_rating(home_stats_obj)
            away_def_rating = self._calculate_defensive_rating(away_stats_obj)
            
            # Add ratings to stats
            home_stats_obj['off_rating'] = home_off_rating
            away_stats_obj['off_rating'] = away_off_rating
            home_stats_obj['def_rating'] = home_def_rating
            away_stats_obj['def_rating'] = away_def_rating
            
            # Process features with balanced home/away consideration
            for feature_name in self.feature_names:
                value = self._calculate_feature_value(
                    feature_name, home_stats_obj, away_stats_obj
                )
                # Normalize differential features
                if '_diff_' in feature_name:
                    value = value / 2  # Reduce the impact of differentials
                features_dict[feature_name] = value
            
            # Create feature array
            features = np.array([[features_dict[name] for name in self.feature_names]])
            
            # Apply feature scaling
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
                    # All models now use predict_proba
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
                # All models now use predict_proba
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
        """Calculate offensive rating based on team statistics."""
        try:
            points = float(stats.get('points', 0))
            fgp = float(stats.get('fgp', 0))
            tpp = float(stats.get('tpp', 0))
            assists = float(stats.get('assists', 0))
            
            # Weighted formula for offensive rating
            off_rating = (points * 0.4 + 
                         fgp * 0.3 + 
                         tpp * 0.2 + 
                         assists * 0.1)
            
            return off_rating
        except:
            return 0.0

    def _calculate_defensive_rating(self, stats: Dict) -> float:
        """Calculate defensive rating based on team statistics."""
        try:
            blocks = float(stats.get('blocks', 0))
            steals = float(stats.get('steals', 0))
            rebounds = float(stats.get('totReb', 0))
            turnovers = float(stats.get('turnovers', 0))
            
            # Weighted formula for defensive rating
            def_rating = (blocks * 0.3 + 
                         steals * 0.3 + 
                         rebounds * 0.3 - 
                         turnovers * 0.1)
            
            return def_rating
        except:
            return 0.0

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
                window = int(parts[-1]) if parts[-1].isdigit() else 5
                home_form = float(home_stats.get('recent_form', {}).get(str(window), 0.5))
                away_form = float(away_stats.get('recent_form', {}).get(str(window), 0.5))
                
                if 'home_' in feature_name:
                    return home_form
                elif 'away_' in feature_name:
                    return away_form
                else:
                    return home_form - away_form
                
            elif 'streak' in feature_name:
                if 'home_' in feature_name:
                    return float(home_stats.get('current_streak', 0))
                else:
                    return float(away_stats.get('current_streak', 0))
                
            elif 'rest_days' in feature_name:
                home_rest = float(home_stats.get('days_rest', 2))
                away_rest = float(away_stats.get('days_rest', 2))
                
                if 'home_' in feature_name:
                    return home_rest
                elif 'away_' in feature_name:
                    return away_rest
                else:
                    return home_rest - away_rest
                
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
            
            # Calculate point differential
            point_diff = points - points_allowed
            
            # Combine factors (weighted)
            strength = (win_pct * 0.6) + (point_diff * 0.002)  # Small weight for point diff
            
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
            # Get base team strengths first
            home_strength = self._calculate_base_strength(home_stats)
            away_strength = self._calculate_base_strength(away_stats)
            
            # Calculate probability based on relative strengths
            total_strength = home_strength + away_strength
            if total_strength == 0:
                base_prob = 0.5
            else:
                # Base probability from team strengths - away team perspective
                away_base_prob = away_strength / total_strength
                
                # Only apply home court advantage if home team is within 90% of away team's strength
                home_court_advantage = 0.02 if home_strength >= (away_strength * 0.9) else 0
                away_base_prob = max(0.2, min(0.8, away_base_prob - home_court_advantage))
                base_prob = 1 - away_base_prob  # Convert to home team perspective
            
            # Get team records
            home_record = self._get_team_record(home_stats)
            away_record = self._get_team_record(away_stats)
            
            # Calculate win percentage difference from away team perspective
            away_winpct = away_record['win_pct']
            home_winpct = home_record['win_pct']
            winpct_diff = (away_winpct - home_winpct) * 0.2  # 20% impact from win percentage
            
            # Calculate streak impact from away team perspective
            home_streak = float(home_stats.get('statistics', [{}])[0].get('streak', 0))
            away_streak = float(away_stats.get('statistics', [{}])[0].get('streak', 0))
            streak_diff = (away_streak - home_streak) * 0.02
            
            # Calculate final probability from away team perspective
            away_final_prob = 1 - base_prob  # Convert to away perspective
            away_final_prob += winpct_diff
            away_final_prob += streak_diff
            
            # Ensure probability is within reasonable bounds
            away_final_prob = max(0.2, min(0.8, away_final_prob))
            
            # Convert back to home team perspective
            final_prob = 1 - away_final_prob
            
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
                    'base_probability': base_prob,
                    'home_court_advantage': home_court_advantage,
                    'win_pct_adjustment': winpct_diff,
                    'streak_adjustment': streak_diff
                }
            }
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {
                'home_probability': 0.5,
                'away_probability': 0.5,
                'home_score_range': (100, 100),
                'away_score_range': (100, 100),
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
            
            # Use both scored and allowed points for range
            base_score = (avg_points + avg_points_allowed) / 2
            
            # Calculate range with smaller variation
            lower_bound = int(max(base_score * 0.9, 85))
            upper_bound = int(min(base_score * 1.1, 140))
            
            return (lower_bound, upper_bound)
        except Exception as e:
            logging.error(f"Error predicting score range: {str(e)}")
            return (95, 115)

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
