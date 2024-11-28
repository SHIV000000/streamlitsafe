# reward_system.py

import json
import os
from datetime import datetime
from typing import Dict, Optional

class RewardSystemManager:
    def __init__(self):
        self.rewards_file = "data/rewards.json"
        self.load_rewards()
        
    def load_rewards(self):
        """Load existing rewards data"""
        if os.path.exists(self.rewards_file):
            with open(self.rewards_file, 'r') as f:
                self.rewards_data = json.load(f)
        else:
            self.rewards_data = {
                'total_coins': 0,
                'boost_points': 0,
                'games_played': 0,
                'correct_predictions': 0,
                'history': {}
            }
            
    def save_rewards(self):
        """Save rewards data to file"""
        os.makedirs(os.path.dirname(self.rewards_file), exist_ok=True)
        with open(self.rewards_file, 'w') as f:
            json.dump(self.rewards_data, f, indent=4)
            
    def update_game_rewards(self, game_id: str, prediction: Dict, actual_result: Dict) -> Dict:
        """Update rewards for a single game"""
        if game_id in self.rewards_data['history']:
            return self.rewards_data['history'][game_id]
            
        rewards = self._calculate_rewards(prediction, actual_result)
        
        # Update totals
        self.rewards_data['total_coins'] += rewards['coins']
        self.rewards_data['boost_points'] += rewards['boost_points']
        self.rewards_data['games_played'] += 1
        if rewards['coins'] > 0:
            self.rewards_data['correct_predictions'] += 1
            
        # Store game rewards
        self.rewards_data['history'][game_id] = {
            'timestamp': datetime.now().isoformat(),
            'rewards': rewards,
            'prediction': prediction,
            'actual_result': actual_result
        }
        
        self.save_rewards()
        return rewards
        
    def _calculate_rewards(self, prediction: Dict, actual_result: Dict) -> Dict:
        """Calculate rewards for a single prediction"""
        rewards = {
            'coins': 0,
            'boost_points': 0,
            'accuracy': 0.0
        }
        
        # Basic win/loss prediction (1 coin)
        if prediction['prediction']['predicted_winner'] == actual_result['winner']:
            rewards['coins'] = 1
        
        # Calculate if scores are within predicted ranges
        pred_score = prediction['prediction']['score_prediction']
        actual_score = actual_result['score']
        
        # Check if actual scores fall within predicted ranges
        home_in_range = (
            pred_score['home_low'] <= actual_score['home'] <= pred_score['home_high']
        )
        away_in_range = (
            pred_score['away_low'] <= actual_score['away'] <= pred_score['away_high']
        )
        
        # Award boost point if both scores are within range
        if home_in_range and away_in_range:
            rewards['boost_points'] = 1
        
        # Calculate accuracy for display purposes only
        home_accuracy = 1 - abs(
            ((pred_score['home_high'] + pred_score['home_low']) / 2) - 
            actual_score['home']
        ) / actual_score['home']
        
        away_accuracy = 1 - abs(
            ((pred_score['away_high'] + pred_score['away_low']) / 2) - 
            actual_score['away']
        ) / actual_score['away']
        
        rewards['accuracy'] = max(0, (home_accuracy + away_accuracy) / 2)
        
        return rewards
        
    def get_progress(self) -> Dict:
        """Get current progress towards goals"""
        return {
            'total_coins': self.rewards_data['total_coins'],
            'boost_points': self.rewards_data['boost_points'],
            'games_played': self.rewards_data['games_played'],
            'accuracy': (
                self.rewards_data['correct_predictions'] / 
                max(1, self.rewards_data['games_played'])
            ) * 100,
            'coins_target': 2460,
            'boost_points_target': 2337
        }


