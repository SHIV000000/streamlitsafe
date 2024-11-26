# nba_api_client.py

from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreTraditionalV2
from datetime import datetime, timedelta
import pandas as pd
import logging
from nba_api.stats.static import teams
from nba_api.stats.library.parameters import SeasonType
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
            # Add delay between requests
            time.sleep(1)
            
            # Convert date to string format
            date_str = date.strftime('%m/%d/%Y')
            cache_key = date_str
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Multiple attempts with increasing delays
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
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
                    if attempt < max_attempts - 1:
                        wait_time = (attempt + 1) * 2
                        logging.warning(f"Attempt {attempt + 1} failed. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise e
                        
        except Exception as e:
            logging.error(f"Error fetching results for {date}: {str(e)}")
            return None
    
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



