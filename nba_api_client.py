# nba_api_client.py

from typing import Dict, List, Optional
import logging
from datetime import datetime, timezone
import json
from nba_stats import NBA_TEAM_STATS

class NBAGameResultsFetcher:
    def __init__(self):
        """Initialize the NBA game results fetcher."""
        self.team_stats = NBA_TEAM_STATS
        
    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """Get team statistics."""
        try:
            if team_name not in self.team_stats:
                logging.error(f"Team {team_name} not found in stats")
                return None
                
            stats = self.team_stats[team_name].copy()
            stats['team_name'] = team_name
            return stats
            
        except Exception as e:
            logging.error(f"Error getting team stats: {str(e)}", exc_info=True)
            return None
            
    def get_upcoming_games(self) -> List[Dict]:
        """Get upcoming games for testing."""
        try:
            # Generate some test games using teams from our stats
            teams = list(self.team_stats.keys())
            games = [
                {
                    'teams': {
                        'home': {'name': teams[i]},
                        'away': {'name': teams[i+1]}
                    },
                    'scheduled_start': '2025-02-25 05:30 IST'
                }
                for i in range(0, len(teams)-1, 2)
            ]
            return games
            
        except Exception as e:
            logging.error(f"Error getting upcoming games: {str(e)}", exc_info=True)
            return []

    def get_game_results(self, date_str: str, home_team: str = None, away_team: str = None) -> List[Dict]:
        """Get game results for a specific date and teams."""
        try:
            results = []
            
            # If specific teams are provided, get their results
            if home_team and away_team:
                home_stats = self.get_team_stats(home_team)
                away_stats = self.get_team_stats(away_team)
                
                if home_stats and away_stats:
                    # Generate scores based on team stats
                    home_score = int(home_stats['points_per_game'])
                    away_score = int(away_stats['points_per_game'])
                    
                    # Add some randomness to scores
                    home_score = max(80, min(130, home_score + (-5 + (hash(date_str + home_team) % 10))))
                    away_score = max(80, min(130, away_score + (-5 + (hash(date_str + away_team) % 10))))
                    
                    results.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'winner': home_team if home_score > away_score else away_team,
                        'date': date_str
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error getting game results: {str(e)}", exc_info=True)
            return []
