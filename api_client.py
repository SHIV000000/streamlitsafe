# api_client.py


import requests
import logging
from typing import Dict, List, Optional, Any
from time import sleep
from datetime import datetime, timedelta
import json
import os

class EnhancedNBAApiClient:
    def __init__(self, api_key: str):
        """Initialize the NBA API client with API key and configuration."""
        self.api_key = api_key
        self.base_url = "https://v2.nba.api-sports.io"
        self.football_base_url = "https://api-football-v1.p.rapidapi.com/v3"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }
        self.nba_headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "v2.nba.api-sports.io"
        }
        self.current_season = '2024'
        self.previous_season = '2023'
        
        # Team name standardization mapping
        self.team_name_mapping = {
            'LA Clippers': ['Los Angeles Clippers', 'LAC', 'LA Clippers'],
            'Los Angeles Lakers': ['LA Lakers', 'LAL', 'Lakers'],
            'Brooklyn Nets': ['BKN', 'Nets'],
            'New York Knicks': ['NY Knicks', 'NYK'],
            'Philadelphia 76ers': ['PHI', 'Sixers', '76ers'],
            'Toronto Raptors': ['TOR'],
            'Chicago Bulls': ['CHI'],
            'Cleveland Cavaliers': ['CLE', 'Cavs'],
            'Detroit Pistons': ['DET'],
            'Indiana Pacers': ['IND'],
            'Milwaukee Bucks': ['MIL'],
            'Atlanta Hawks': ['ATL'],
            'Charlotte Hornets': ['CHA'],
            'Miami Heat': ['MIA'],
            'Orlando Magic': ['ORL'],
            'Washington Wizards': ['WAS'],
            'Denver Nuggets': ['DEN'],
            'Minnesota Timberwolves': ['MIN', 'Wolves'],
            'Oklahoma City Thunder': ['OKC'],
            'Portland Trail Blazers': ['POR', 'Blazers'],
            'Utah Jazz': ['UTA'],
            'Golden State Warriors': ['GSW', 'Warriors'],
            'Phoenix Suns': ['PHX'],
            'Sacramento Kings': ['SAC'],
            'Dallas Mavericks': ['DAL', 'Mavs'],
            'Houston Rockets': ['HOU'],
            'Memphis Grizzlies': ['MEM'],
            'New Orleans Pelicans': ['NOP', 'Pels'],
            'San Antonio Spurs': ['SAS'],
            'Boston Celtics': ['BOS']
        }
        
        # Configure logging
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging with error handling
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(os.path.join(log_dir, "nba_api.log"), mode='a'),
                    logging.StreamHandler()
                ]
            )
        except Exception as e:
            # Fallback to console-only logging if file logging fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            logging.warning(f"Failed to set up file logging: {str(e)}. Falling back to console logging only.")

    def standardize_team_name(self, team_name: str) -> str:
        """Standardize team name to a consistent format."""
        if not team_name:
            return team_name
            
        # Check if it's already a standard name
        if team_name in self.team_name_mapping:
            return team_name
            
        # Look for variations
        for standard_name, variations in self.team_name_mapping.items():
            if team_name in variations:
                return standard_name
                
        # If no match found, log warning and return original
        logging.warning(f"Unknown team name format: {team_name}")
        return team_name

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
                    timeout=30
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

    def get_upcoming_games(self) -> List[Dict]:
        """Get upcoming games for today and tomorrow."""
        try:
            endpoint = f"{self.base_url}/games"
            games = []
            
            # Get today's and tomorrow's games
            for day_offset in [0, 1]:
                date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                params = {
                    'date': date,
                    'league': 'standard',
                    'season': self.current_season
                }
                
                logging.info(f"Fetching games for date: {date}")
                response = self._make_request('GET', endpoint, params)
                day_games = response.get('response', [])
                
                # Process each game
                for game in day_games:
                    try:
                        # Only include scheduled or not started games
                        status = game.get('status', {})
                        if status.get('long') in ['Scheduled', 'Not Started'] or status.get('short') in ['1', 'NS']:
                            processed_game = {
                                'id': game.get('id'),
                                'date': {
                                    'start': game.get('date', {}).get('start')
                                },
                                'teams': {
                                    'home': {
                                        'id': game.get('teams', {}).get('home', {}).get('id'),
                                        'name': self.standardize_team_name(game.get('teams', {}).get('home', {}).get('name'))
                                    },
                                    'away': {
                                        'id': game.get('teams', {}).get('visitors', {}).get('id'),
                                        'name': self.standardize_team_name(game.get('teams', {}).get('visitors', {}).get('name'))
                                    }
                                },
                                'status': status
                            }
                            games.append(processed_game)
                            logging.info(f"Added game: {processed_game['teams']['home']['name']} vs {processed_game['teams']['away']['name']}")
                    except Exception as e:
                        logging.error(f"Error processing game {game.get('id')}: {str(e)}")
                        continue
            
            logging.info(f"Found {len(games)} upcoming games")
            return games
            
        except Exception as e:
            logging.error(f"Error fetching upcoming games: {str(e)}")
            return []

    def get_live_games(self) -> List[Dict]:
        """Get current live games with enhanced status checking."""
        try:
            # First attempt: Try getting games directly
            endpoint = f"{self.base_url}/games"
            params = {
                'live': 'all',
                'league': 'standard',
                'season': self.current_season
            }
            
            logging.info(f"Attempting to fetch live games with params: {params}")
            response = self._make_request('GET', endpoint, params)
            all_games = response.get('response', [])
            
            # If no games found, try getting today's games
            if not all_games:
                today = datetime.now().strftime("%Y-%m-%d")
                params = {
                    'date': today,
                    'league': 'standard',
                    'season': self.current_season
                }
                logging.info(f"Attempting to fetch today's games with params: {params}")
                response = self._make_request('GET', endpoint, params)
                all_games = response.get('response', [])
            
            logging.debug(f"Found {len(all_games)} games in response")
            
            # Enhanced live game detection
            live_games = []
            for game in all_games:
                try:
                    status = game.get('status', {})
                    periods = game.get('periods', {})
                    scores = game.get('scores', {})
                    
                    # Log individual game status
                    logging.debug(f"Processing game: {game.get('id')}")
                    logging.debug(f"Game status: {json.dumps(status, indent=2)}")
                    
                    # Check if game is live
                    if self._is_game_live(status, periods, scores):
                        processed_game = self._process_game_data(game)
                        if processed_game:
                            live_games.append(processed_game)
                            logging.info(f"Found live game: {self.debug_game_status(game)}")
                
                except Exception as e:
                    logging.error(f"Error processing game {game.get('id')}: {str(e)}")
                    continue
            
            if not live_games:
                logging.info("No live games found")
            else:
                logging.info(f"Found {len(live_games)} live games")
                
            return live_games
            
        except Exception as e:
            logging.error(f"Error fetching live games: {str(e)}")
            return []

    def _is_game_live(self, status: Dict, periods: Dict, scores: Dict) -> bool:
        """Helper method to determine if a game is live."""
        try:
            status_long = status.get('long', '').lower()
            status_short = str(status.get('short', ''))
            current_period = periods.get('current')
            
            # Debug logging for status checking
            logging.debug(f"Checking game status - Long: {status_long}, Short: {status_short}, Period: {current_period}")
            
            # First check: Explicitly handle finished games
            if status_short == '3' or status_long == 'finished':
                logging.debug("Game is finished")
                return False
            
            # Second check: Handle scheduled games
            if status_short == '1' or status_long == 'scheduled':
                logging.debug("Game is scheduled")
                return False
            
            # Third check: Handle quarter-specific status
            quarter_status = {
                'q1': current_period == 1,
                'q2': current_period == 2,
                'q3': current_period == 3,
                'q4': current_period == 4,
                'ot': current_period > 4
            }
            
            # Fourth check: Game state indicators
            game_state = {
                'has_clock': status.get('clock') is not None,
                'has_scores': all(scores.get(team, {}).get('points', 0) > 0 
                                for team in ['home', 'visitors']),
                'valid_period': current_period is not None and current_period > 0,
                'is_playing': status_short == '2' or status_long in ['in play', 'live'],
                'is_halftime': status.get('halftime', False) is True,
                'in_quarter': any(quarter_status.values())
            }
            
            logging.debug(f"Quarter status: {quarter_status}")
            logging.debug(f"Game state: {game_state}")
            
            # Game is considered live if:
            # 1. It's in a valid quarter/period
            # 2. Has either a clock or valid scores
            # 3. Is not finished
            is_live = (
                game_state['valid_period'] and
                (game_state['has_clock'] or game_state['has_scores']) and
                (game_state['is_playing'] or game_state['is_halftime'] or game_state['in_quarter'])
            )
            
            if is_live:
                logging.info(f"Game is live - Period: {current_period}, Status: {status_long}")
            
            return is_live
            
        except Exception as e:
            logging.error(f"Error in _is_game_live: {str(e)}")
            return False

    def _process_game_data(self, game: Dict) -> Optional[Dict]:
        """Process raw game data into standardized format."""
        try:
            # Extract and validate team data
            teams = game.get('teams', {})
            home_team = teams.get('home', {})
            away_team = teams.get('visitors', {})  # API uses 'visitors' for away team
            
            # Debug logging for team data
            logging.debug(f"Raw home team data: {json.dumps(home_team, indent=2)}")
            logging.debug(f"Raw away team data: {json.dumps(away_team, indent=2)}")
            
            # Extract team IDs with proper validation
            home_id = home_team.get('id')
            away_id = away_team.get('id')
            
            if not home_id or not away_id:
                logging.error(f"Missing team ID - Home: {home_id}, Away: {away_id}")
                return None
            
            # Convert IDs to strings
            home_id_str = str(home_id)
            away_id_str = str(away_id)
            
            # Extract scores with proper path
            scores = game.get('scores', {})
            home_score = scores.get('home', {}).get('points', 0)
            away_score = scores.get('visitors', {}).get('points', 0)
            
            return {
                'id': str(game.get('id')),
                'teams': {
                    'home': {
                        'id': home_id_str,
                        'name': self.standardize_team_name(home_team.get('name', '')),
                        'nickname': home_team.get('nickname', ''),
                        'code': home_team.get('code', '')
                    },
                    'visitors': {  # Changed from 'away' to 'visitors'
                        'id': away_id_str,
                        'name': self.standardize_team_name(away_team.get('name', '')),
                        'nickname': away_team.get('nickname', ''),
                        'code': away_team.get('code', '')
                    }
                },
                'scores': {
                    'home': {
                        'points': int(home_score) if home_score is not None else 0,
                        'linescore': scores.get('home', {}).get('linescore', [])
                    },
                    'visitors': {  # Changed from 'away' to 'visitors'
                        'points': int(away_score) if away_score is not None else 0,
                        'linescore': scores.get('visitors', {}).get('linescore', [])
                    }
                },
                'status': {
                    'clock': game.get('status', {}).get('clock'),
                    'period': game.get('periods', {}).get('current', 1),
                    'halftime': game.get('status', {}).get('halftime', False),
                    'short': game.get('status', {}).get('short'),
                    'long': game.get('status', {}).get('long')
                },
                'periods': game.get('periods', {}),
                'arena': game.get('arena', {}),
                'date': {
                    'start': game.get('date', {}).get('start'),
                    'end': game.get('date', {}).get('end')
                }
            }
        except Exception as e:
            logging.error(f"Error processing game data: {str(e)}")
            return None

    def debug_game_status(self, game: Dict) -> str:
        """Helper function to debug game status"""
        return (f"Game ID: {game.get('id')} - "
                f"Status: {game.get('status', {}).get('long')} - "
                f"Clock: {game.get('status', {}).get('clock')} - "
                f"Period: {game.get('periods', {}).get('current')} - "
                f"Home: {self.standardize_team_name(game.get('teams', {}).get('home', {}).get('name'))} "
                f"({game.get('scores', {}).get('home', {}).get('points', 0)}) - "
                f"Away: {self.standardize_team_name(game.get('teams', {}).get('visitors', {}).get('name'))} "
                f"({game.get('scores', {}).get('visitors', {}).get('points', 0)})")

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
                            'name': self.standardize_team_name(game.get('teams', {}).get('home', {}).get('name'))
                        },
                        'away': {
                            'id': str(game.get('teams', {}).get('visitors', {}).get('id')),
                            'name': self.standardize_team_name(game.get('teams', {}).get('visitors', {}).get('name'))
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
            endpoint = f"{self.base_url}/teams/statistics"
            params = {
                'id': validated_id,
                'season': self.current_season
            }
            
            response = self._make_request('GET', endpoint, params)
            stats = response.get('response', [])
            
            if not stats:
                # Try previous season if current season stats not available
                params['season'] = self.previous_season
                response = self._make_request('GET', endpoint, params)
                stats = response.get('response', [])
            
            if stats and len(stats) > 0:
                return self.process_team_stats(stats[0])
            
            # Return default stats structure if no data found
            return {
                'statistics': [{
                    'points': 0.0,
                    'fieldGoalsPercentage': 0.0,
                    'threePointsPercentage': 0.0,
                    'freeThrowsPercentage': 0.0,
                    'reboundsTotal': 0.0,
                    'assists': 0.0,
                    'steals': 0.0,
                    'blocks': 0.0,
                    'turnovers': 0.0,
                    'games': 0,
                    'wins': 0
                }]
            }
            
        except Exception as e:
            logging.error(f"Error fetching team stats for ID {team_id}: {str(e)}")
            return {'statistics': [{}]}

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
            if games == 0:
                games = 1
            
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
            return {
                'statistics': [{
                    'points': 0.0,
                    'fieldGoalsPercentage': 0.0,
                    'threePointsPercentage': 0.0,
                    'freeThrowsPercentage': 0.0,
                    'reboundsTotal': 0.0,
                    'assists': 0.0,
                    'steals': 0.0,
                    'blocks': 0.0,
                    'turnovers': 0.0,
                    'games': 0,
                    'wins': 0
                }]
            }

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

    def get_game_results(self, date: str) -> List[Dict]:
        """
        Get game results for a specific date.
        Args:
            date (str): Date in format YYYY-MM-DD
        Returns:
            List[Dict]: List of game results
        """
        endpoint = f"{self.football_base_url}/fixtures"
        params = {
            "date": date,
            "league": "115",  # NBA league ID in API-FOOTBALL
            "season": self.current_season
        }
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('response'):
                logging.warning(f"No game results found for date: {date}")
                return []
            
            results = []
            for game in data['response']:
                result = {
                    'home_team': self.standardize_team_name(game['teams']['home']['name']),
                    'away_team': self.standardize_team_name(game['teams']['away']['name']),
                    'home_score': game['score']['fulltime']['home'],
                    'away_score': game['score']['fulltime']['away'],
                    'status': game['fixture']['status']['long'],
                    'date': game['fixture']['date'],
                    'id': game['fixture']['id']
                }
                results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching game results: {str(e)}")
            return []

    def update_game_results(self, predictions: List[Dict]) -> List[Dict]:
        """
        Update predictions with actual game results.
        Args:
            predictions (List[Dict]): List of predictions
        Returns:
            List[Dict]: Updated predictions with results
        """
        for pred in predictions:
            game_date = datetime.fromisoformat(pred['scheduled_start'].replace('Z', '+00:00')).strftime('%Y-%m-%d')
            results = self.get_game_results(game_date)
            
            # Find matching game result
            for result in results:
                if (result['home_team'] == self.standardize_team_name(pred['home_team']) and 
                    result['away_team'] == self.standardize_team_name(pred['away_team'])):
                    pred['actual_home_score'] = result['home_score']
                    pred['actual_away_score'] = result['away_score']
                    pred['game_status'] = result['status']
                    
                    # Determine if prediction was correct
                    if result['status'] == 'Match Finished':
                        actual_winner = (result['home_team'] if result['home_score'] > result['away_score']
                                      else result['away_team'] if result['away_score'] > result['home_score']
                                      else 'Tie')
                        pred['prediction_correct'] = (actual_winner == pred['predicted_winner'])
                    
                    break
            
            # If no result found, mark as pending
            if 'game_status' not in pred:
                pred['game_status'] = 'Scheduled'
        
        return predictions

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
                'name': self.standardize_team_name(team_data.get('name', '')),
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
