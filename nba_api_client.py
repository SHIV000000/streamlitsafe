# nba_api_client.py

from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreTraditionalV2, Scoreboard
from nba_api.stats.static import teams
from nba_api.stats.library.parameters import SeasonType
from datetime import datetime, timedelta
import pandas as pd
import logging
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from typing import Dict

class NBAGameResultsFetcher:
    def __init__(self):
        self.cache = {}
        self.headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://stats.nba.com/',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true'
        }
        
        # Configure retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        
    def get_game_results(self, date):
        """Fetch game results for a specific date"""
        logging.info(f"Fetching results for date: {date}")
        
        try:
            time.sleep(1)
            
            # Standardize date handling
            if isinstance(date, datetime):
                date = date.date()
            
            # Format date for API request (MM/DD/YYYY)
            date_str = date.strftime('%m/%d/%Y')
            current_date = datetime.now().date()
            
            cache_key = date_str
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            results = {}
            
            # For today's games
            if date == current_date:
                try:
                    from nba_api.live.nba.endpoints import scoreboard
                    games = scoreboard.ScoreBoard()
                    games_dict = games.get_dict()
                    
                    if 'games' in games_dict:
                        for game in games_dict['games']:
                            if game['gameStatus'] >= 3:  # Game is finished
                                game_id = str(game['gameId'])
                                home_team = self._normalize_team_name(game['homeTeam']['teamName'])
                                away_team = self._normalize_team_name(game['awayTeam']['teamName'])
                                home_score = int(game['homeTeam']['score'])
                                away_score = int(game['awayTeam']['score'])
                                
                                results[game_id] = {
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'winner': home_team if home_score > away_score else away_team,
                                    'game_date': date_str
                                }
                except Exception as e:
                    logging.error(f"Error fetching today's games: {str(e)}")
            
            # If no results from live API or not today's games, use LeagueGameFinder
            if not results:
                game_finder = LeagueGameFinder(
                    date_from_nullable=date_str,
                    date_to_nullable=date_str,
                    league_id_nullable='00',
                    season_type_nullable=SeasonType.regular,
                    player_or_team_abbreviation='T',
                    headers=self.headers,
                    timeout=60
                )
                
                games_df = game_finder.get_data_frames()[0]
                
                if not games_df.empty:
                    results = self._process_games_data(games_df)
            
            if results:
                self.cache[cache_key] = results
                return results
            
            logging.warning(f"No valid games found for date {date_str}")
            return {}
            
        except Exception as e:
            logging.error(f"Error fetching results for {date}: {str(e)}")
            return {}
    
    def _process_games_data(self, games_df):
        """Process the games dataframe and extract relevant information"""
        results = {}
        try:
            # Print columns for debugging
            logging.debug(f"Available columns: {games_df.columns.tolist()}")
            
            for game_id, group in games_df.groupby('GAME_ID'):
                try:
                    # Filter for home and away games
                    home_games = group[group['MATCHUP'].str.contains('vs', case=False, na=False)]
                    away_games = group[group['MATCHUP'].str.contains('@', case=False, na=False)]
                    
                    if len(home_games) > 0 and len(away_games) > 0:
                        home_game = home_games.iloc[0]
                        away_game = away_games.iloc[0]
                        
                        # Determine winner
                        home_won = home_game['WL'] == 'W'
                        winner = home_game['TEAM_NAME'] if home_won else away_game['TEAM_NAME']
                        
                        results[str(game_id)] = {
                            'home_team': home_game['TEAM_NAME'],
                            'away_team': away_game['TEAM_NAME'],
                            'home_score': int(home_game['PTS']),
                            'away_score': int(away_game['PTS']),
                            'winner': winner,
                            'game_date': home_game['GAME_DATE']
                        }
                    
                except Exception as e:
                    logging.warning(f"Error processing game {game_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing games data: {str(e)}")
            
        return results
    
    def _print_debug_info(self, games_df):
        """Helper method to print debug information about the API response"""
        logging.debug("Available columns: " + str(games_df.columns.tolist()))
        logging.debug("Sample data:")
        logging.debug(games_df.head().to_string())
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names to match between different API responses"""
        replacements = {
            'LA ': 'Los Angeles ',
            'Blazers': 'Trail Blazers',
            'Sixers': '76ers'
        }
        
        normalized = team_name
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized
    
    def get_team_stats(self, team_name: str) -> Dict:
        """Get team statistics from NBA.com."""
        try:
            # Get team ID
            team_id = None
            all_teams = teams.get_teams()
            for team in all_teams:
                if team['full_name'].lower() == team_name.lower() or team['nickname'].lower() == team_name.lower():
                    team_id = team['id']
                    break
            
            if not team_id:
                logging.error(f"Could not find team ID for {team_name}")
                return None
            
            # Get team game logs
            game_finder = LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable='2023-24',
                league_id_nullable='00',
                season_type_nullable=SeasonType.regular
            )
            
            games_df = game_finder.get_data_frames()[0]
            if games_df.empty:
                logging.error(f"No games found for team {team_name}")
                return None
            
            # Calculate statistics
            total_games = len(games_df)
            wins = len(games_df[games_df['WL'] == 'W'])
            losses = total_games - wins
            
            # Calculate points per game and points allowed
            points_scored = games_df['PTS'].mean()
            points_allowed = games_df['PTS'].mean() - games_df['PLUS_MINUS'].mean()
            
            # Get last 10 games
            last_10 = games_df.head(10)
            last_10_wins = len(last_10[last_10['WL'] == 'W'])
            
            # Calculate win streak
            streak = 0
            for wl in games_df['WL']:
                if wl == games_df['WL'].iloc[0]:
                    streak += 1
                else:
                    break
            
            # Get home/away records
            home_games = games_df[games_df['MATCHUP'].str.contains('vs.')]
            away_games = games_df[games_df['MATCHUP'].str.contains('@')]
            
            home_wins = len(home_games[home_games['WL'] == 'W'])
            away_wins = len(away_games[away_games['WL'] == 'W'])
            
            # Calculate shooting percentages
            fg_pct = games_df['FG_PCT'].mean() * 100
            fg3_pct = games_df['FG3_PCT'].mean() * 100
            
            stats = {
                'wins': wins,
                'losses': losses,
                'points_per_game': float(points_scored),
                'points_allowed': float(points_allowed),
                'field_goal_pct': float(fg_pct),
                'three_point_pct': float(fg3_pct),
                'win_streak': int(streak if games_df['WL'].iloc[0] == 'W' else 0),
                'last_ten': {'wins': last_10_wins, 'losses': 10 - last_10_wins},
                'home_record': {'wins': home_wins, 'losses': len(home_games) - home_wins},
                'away_record': {'wins': away_wins, 'losses': len(away_games) - away_wins}
            }
            
            logging.info(f"Calculated stats for {team_name}: {json.dumps(stats, indent=2)}")
            return stats
            
        except Exception as e:
            logging.error(f"Error getting team stats for {team_name}: {str(e)}")
            return None
