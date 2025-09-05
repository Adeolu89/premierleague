import pandas as pd
from typing import Dict, Tuple

def preprocess_match_data(csv_path: str, output_dir: str = "data/02_preprocessed/") -> Dict[str, pd.DataFrame]:
    """
    Preprocess raw match data and split into seasons.
    
    Args:
        csv_path: Path to the raw CSV file
        output_dir: Directory to save processed files
    
    Returns:
        Dictionary of season DataFrames
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Reorder columns
    column_order = ['date', 'time', 'round','team', 'opponent', 'venue' ,'gf', 'ga', 
                   'result', 'formation', 'opp formation', 'poss','xg', 'xga', 'sh', 
                   'sot', 'dist', 'season']
    df = df[column_order]
    
    # Standardize team names
    team = list(df['team'].unique())
    team.sort()
    opponent = list(df['opponent'].unique())
    opponent.sort()
    team_names = dict(zip(team, opponent))
    df['team'] = df['team'].apply(lambda x: team_names[x])
    
    # Split into home and away
    df_home = df[df['venue'] == 'Home'].sort_values(by=['date','time']).reset_index(drop=True)
    df_away = df[df['venue'] == 'Away'].sort_values(by=['date','time']).reset_index(drop=True)
    
    # Rename columns for away data
    df_away.rename(columns={
        "team": "away_team", "opponent": "home_team", "formation": "away_formation",
        "opp formation": "home_formation", "gf": "away_goals", "ga": "home_goals",
        "poss": "away_poss", "xg": "away_xg", "xga": "home_xg", "sh": "away_sh",
        "sot": "away_shot_on_target", "dist": "away_dist_covered"
    }, inplace=True)
    
    # Rename columns for home data
    df_home.rename(columns={
        "team": "home_team", "opponent": "away_team", "formation": "home_formation",
        "opp formation": "away_formation", "gf": "home_goals", "ga": "away_goals",
        "poss": "home_poss", "xg": "home_xg", "xga": "away_xg", "sh": "home_sh",
        "sot": "home_shot_on_target", "dist": "home_dist_covered"
    }, inplace=True)
    
    # Merge home and away data
    away_stats_to_add = df_away[['date', 'home_team', 'away_team', 'away_sh', 
                                'away_shot_on_target', 'away_dist_covered', 'away_poss']].copy()
    fixtures_df = pd.merge(df_home, away_stats_to_add, on=['date', 'home_team', 'away_team'])
    
    # Reorder final columns
    new_order = ['date', 'time', 'round', 'home_team', 'away_team', 'venue', 'result',
                'home_goals', 'away_goals', 'home_poss', 'away_poss', 'home_xg', 'away_xg',
                'home_sh', 'away_sh', 'home_shot_on_target', 'away_shot_on_target', 'season']
    fixtures_df = fixtures_df[new_order]
    
    # Set date as index
    fixtures_df.index = fixtures_df['date']
    fixtures_df.drop(columns=['date'], inplace=True)
    
    # Split by seasons and save
    seasons = {
        '2020-2021': fixtures_df[fixtures_df['season'] == 2021],
        '2021-2022': fixtures_df[fixtures_df['season'] == 2022],
        '2022-2023': fixtures_df[fixtures_df['season'] == 2023],
        '2023-2024': fixtures_df[fixtures_df['season'] == 2024],
        '2024-2025': fixtures_df[fixtures_df['season'] == 2025]
    }
    
    # Save to CSV files
    for season_name, season_df in seasons.items():
        season_df.to_csv(f"{output_dir}{season_name}.csv")
    
    return seasons

if __name__ == "__main__":
    # Run the preprocessing
    seasons_data = preprocess_match_data("data/01_raw/final_matches.csv")
    print("Data preprocessing completed!")