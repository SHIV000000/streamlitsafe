# api_client.py


import requests
import logging
from typing import Dict, List, Optional, Any
from time import sleep
from datetime import datetime
import json

class EnhancedNBAApiClient:
    def __init__(self, api_key: str):
        """Initialize the NBA API client with API key and configuration."""
        self.headers = {
            'x-rapidapi-host': 'api-nba-v1.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
        self.base_url = 'https://api-nba-v1.p.rapidapi.com'
        self.current_season = '2024'
        self.previous_season = '2023'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _safe_convert_id(self, value: Any) -> str:
        """Safely convert any value to a string ID."""
        try:
            if value is None or value == '':
                return ''
            return str(value).strip()
        except Exception as e:
            logging.warning(f"Error converting ID value '{value}': {str(e)}")
            return ''

    def _validate_team_id(self, team_id: Any) -> str:
        """Validate and convert team ID to string format."""
        try:
            if team_id is None or str(team_id).strip() == '':
                raise ValueError("Team ID cannot be None or empty")
            validated_id = str(team_id).strip()
            if not validated_id.isdigit():
                raise ValueError("Team ID must be numeric")
            return validated_id
        except Exception as e:
            logging.error(f"Invalid team ID: {team_id}. Error: {str(e)}")
            raise ValueError(f"Invalid team ID: {team_id}")

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with retry logic and error handling."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method,
                    endpoint,
                    headers=self.headers,
                    params=params,
                    timeout=30  # Add timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    sleep(retry_delay)
                else:
                    raise
        return {}

    def get_live_games(self) -> List[Dict]:
        """Get current live games with validated team IDs."""
        try:
            endpoint = f"{self.base_url}/games"
            params = {
                'live': 'all',
                'league': 'standard',
                'season': self.current_season
            }
            
            response = self._make_request('GET', endpoint, params)
            all_games = response.get('response', [])
            
            live_games = [
                game for game in all_games 
                if (game.get('status', {}).get('long') == "In Play" or
                    game.get('status', {}).get('short') == 2 or
                    game.get('status', {}).get('clock', None) is not None)
            ]
            
            return self._process_live_games(live_games)
            
        except Exception as e:
            logging.error(f"Error fetching live games: {str(e)}")
            return []

    def _process_live_games(self, games: List[Dict]) -> List[Dict]:
        """Process raw live game data into required format."""
        processed_games = []
        for game in games:
            try:
                processed_game = {
                    'id': str(game.get('id')),
                    'teams': {
                        'home': {
                            'id': str(game.get('teams', {}).get('home', {}).get('id')),
                            'name': game.get('teams', {}).get('home', {}).get('name')
                        },
                        'away': {
                            'id': str(game.get('teams', {}).get('visitors', {}).get('id')),
                            'name': game.get('teams', {}).get('visitors', {}).get('name')
                        }
                    },
                    'scores': {
                        'home': game.get('scores', {}).get('home', {}).get('points', 0),
                        'away': game.get('scores', {}).get('visitors', {}).get('points', 0)
                    },
                    'status': {
                        'clock': game.get('status', {}).get('clock'),
                        'period': game.get('periods', {}).get('current', 1)
                    }
                }
                processed_games.append(processed_game)
                
            except Exception as e:
                logging.error(f"Error processing game {game.get('id')}: {str(e)}")
                continue
                
        return processed_games

    def get_team_stats(self, team_id: str) -> Dict:
        """Get team statistics with proper error handling and ID validation."""
        try:
            validated_id = self._validate_team_id(team_id)
            
            # Try current season first
            stats = self._get_season_stats(validated_id, self.current_season)
            
            # Fall back to previous season if needed
            if not stats:
                logging.info(f"No current season stats for team {validated_id}, trying previous season")
                stats = self._get_season_stats(validated_id, self.previous_season)
            
            return stats or {}
            
        except Exception as e:
            logging.error(f"Error fetching team stats for ID {team_id}: {str(e)}")
            return {}

    def _get_season_stats(self, team_id: str, season: str) -> Dict:
        """Helper method to get stats for a specific season."""
        endpoint = f"{self.base_url}/teams/statistics"
        params = {
            'id': team_id,
            'season': season
        }
        
        response = self._make_request('GET', endpoint, params)
        stats = response.get('response', [])
        
        return self.process_team_stats(stats[0]) if stats else {}

    def process_team_stats(self, stats: Dict) -> Dict:
        """Process raw team statistics into required format."""
        try:
            games = float(stats.get('games', 1))
            return {
                'statistics': [{
                    'points': float(stats.get('points', 0)) / games,
                    'fieldGoalsPercentage': float(stats.get('fgp', 0)),
                    'threePointsPercentage': float(stats.get('tpp', 0)),
                    'freeThrowsPercentage': float(stats.get('ftp', 0)),
                    'reboundsTotal': float(stats.get('totReb', 0)) / games,
                    'assists': float(stats.get('assists', 0)) / games,
                    'steals': float(stats.get('steals', 0)) / games,
                    'blocks': float(stats.get('blocks', 0)) / games,
                    'turnovers': float(stats.get('turnovers', 0)) / games,
                    'games': games,
                    'wins': float(stats.get('wins', 0))
                }]
            }
        except Exception as e:
            logging.error(f"Error processing team stats: {str(e)}")
            return {'statistics': [{}]}

    def get_game_statistics(self, game_id: str) -> Dict:
        """Get detailed game statistics."""
        endpoint = f"{self.base_url}/games/statistics"
        params = {
            'id': game_id
        }
        response = self._make_request('GET', endpoint, params)
        return response.get('response', {})

    def get_h2h(self, team1_id: str, team2_id: str, season: str = "2023") -> List[Dict]:
        """Get head-to-head matches between two teams."""
        try:
            id1 = self._validate_team_id(team1_id)
            id2 = self._validate_team_id(team2_id)
            endpoint = f"{self.base_url}/games"
            params = {
                'h2h': f"{id1}-{id2}",
                'season': season
            }
            response = self._make_request('GET', endpoint, params)
            return response.get('response', [])
        except Exception as e:
            logging.error(f"Error in h2h request: {str(e)}")
            return []

    def get_team_stats_alternative(self, team_id: str) -> Dict:
        """Alternative method to get team statistics using standings endpoint."""
        endpoint = f"{self.base_url}/standings"
        params = {
            'team': team_id,
            'season': '2024'
        }
        
        try:
            response = self._make_request('GET', endpoint, params)
            standings = response.get('response', [])
            
            if not standings:
                return {}
                
            team_stats = standings[0]
            
            # Convert standings data to our statistics format
            processed_stats = self.process_team_stats(team_stats)
            
            return processed_stats
            
        except Exception as e:
            logging.error(f"Error fetching alternative team stats for ID {team_id}: {str(e)}")
            return {}

    def get_team_injuries(self, team_id: str) -> List[Dict]:
        """Get current team injuries."""
        try:
            validated_id = self._validate_team_id(team_id)
            endpoint = f"{self.base_url}/injuries"
            params = {
                'team': validated_id,
                'league': 'standard'
            }
            
            response = self._make_request('GET', endpoint, params)
            injuries = response.get('response', [])
            
            return [{
                'player': injury.get('player', {}).get('name'),
                'status': injury.get('status'),
                'reason': injury.get('reason'),
                'date': injury.get('date')
            } for injury in injuries]
        except Exception as e:
            logging.error(f"Error fetching injuries for team {team_id}: {str(e)}")
            return []

    def get_team_info(self, team_id: str) -> Dict:
        """Get detailed team information with validated ID."""
        try:
            validated_id = self._validate_team_id(team_id)
            endpoint = f"{self.base_url}/teams"
            params = {'id': validated_id}
            
            response = self._make_request('GET', endpoint, params)
            team_data = response.get('response', [{}])[0]
            
            return {
                'id': validated_id,  # Include the validated ID
                'name': team_data.get('name', ''),
                'code': team_data.get('code', ''),
                'city': team_data.get('city', ''),
                'conference': team_data.get('leagues', {}).get('standard', {}).get('conference', ''),
                'division': team_data.get('leagues', {}).get('standard', {}).get('division', '')
            }
        except Exception as e:
            logging.error(f"Error fetching team info for ID {team_id}: {str(e)}")
            return {}

    def get_team_players(self, team_id: str) -> List[Dict]:
        """Get detailed player information for a team."""
        endpoint = f"{self.base_url}/players"
        params = {
            'team': team_id,
            'season': '2024'
        }
        response = self._make_request('GET', endpoint, params)
        return response.get('response', [])

    def get_h2h_detailed(self, team1_id: str, team2_id: str) -> Dict:
        """Get detailed head-to-head statistics."""
        endpoint = f"{self.base_url}/games/h2h"
        params = {
            'h2h': f"{team1_id}-{team2_id}",
            'season': '2024'
        }
        response = self._make_request('GET', endpoint, params)
        return response.get('response', [])

    def get_team_stats_with_retry(self, team_id: str, max_retries: int = 3) -> Dict:
        """Get team stats with improved retry logic and fallback."""
        for attempt in range(max_retries):
            try:
                # Try primary endpoint
                stats = self.get_team_stats(team_id)
                if stats and stats.get('statistics'):
                    return stats
                
                # Try alternative endpoint
                stats = self.get_team_stats_alternative(team_id)
                if stats and stats.get('statistics'):
                    return stats
                
                # Try backup data source
                stats = self.get_team_standings(team_id)
                if stats:
                    return self.convert_standings_to_stats(stats)
                    
                sleep(1)  # Short delay between retries
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for team ID {team_id}: {str(e)}")
                
        return {'statistics': [{}]}  # Return empty stats if all attempts fail

    def get_team_injuries_with_fallback(self, team_id: str) -> List[Dict]:
        """Get team injuries with fallback options."""
        try:
            # Try primary injuries endpoint
            injuries = self.get_team_injuries(team_id)
            if injuries:
                return injuries
                
            # Try alternative source
            roster = self.get_team_players(team_id)
            if roster:
                return self.extract_injuries_from_roster(roster)
                
            return []
            
        except Exception as e:
            logging.error(f"Error getting injuries for team {team_id}: {str(e)}")
            return []

    def extract_injuries_from_roster(self, roster: List[Dict]) -> List[Dict]:
        """Extract injury information from roster data."""
        injuries = []
        for player in roster:
            if player.get('status') in ['Out', 'Questionable', 'Doubtful']:
                injuries.append({
                    'player': player.get('name'),
                    'status': player.get('status'),
                    'reason': player.get('injury', {}).get('description', 'Unknown')
                })
        return injuries


