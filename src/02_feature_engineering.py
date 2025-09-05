import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data and prepare initial preprocessing
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Prepared DataFrame with date column converted and result mapped to integers
    """
    # Load data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Map results to integers
    results_map = {"W": 1, "D": 0, "L": -1}  # W,D,L all in the context of the home team
    df['result'] = df['result'].apply(lambda x: results_map[x])
    
    return df

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling features for each team based on last 5 games (regardless of venue)
    
    Args:
        df: DataFrame with match data
        
    Returns:
        DataFrame with rolling features added
    """
    df_prep = df.copy()
    
    # Last 5 games performance for each team (regardless of venue)
    for team in df_prep['home_team'].unique():
        # Get all games for this team (both home and away)
        team_home_games = df_prep[df_prep['home_team'] == team].copy()
        team_away_games = df_prep[df_prep['away_team'] == team].copy()
        
        # Flip results for away games (from team's perspective)
        team_away_games['result'] = team_away_games['result'] * -1
        
        # Create standardized columns for goals, xg, etc. from team perspective
        team_away_games['team_goals'] = team_away_games['away_goals']  
        team_away_games['team_xg'] = team_away_games['away_xg']
        team_away_games['team_shots'] = team_away_games['away_sh']
        team_away_games['team_shots_on_target'] = team_away_games['away_shot_on_target']
        team_away_games['team_poss'] = team_away_games['away_poss']
        team_away_games['team_goals_conceded'] = team_away_games['home_goals']
        team_away_games['team_xg_conceded'] = team_away_games['home_xg']
        
        team_home_games['team_goals'] = team_home_games['home_goals']
        team_home_games['team_xg'] = team_home_games['home_xg'] 
        team_home_games['team_shots'] = team_home_games['home_sh']
        team_home_games['team_shots_on_target'] = team_home_games['home_shot_on_target']
        team_home_games['team_poss'] = team_home_games['home_poss']
        team_home_games['team_goals_conceded'] = team_home_games['away_goals']
        team_home_games['team_xg_conceded'] = team_home_games['away_xg']
        
        # Combine and sort by date
        all_team_games = pd.concat([team_home_games, team_away_games]).sort_values('date')
        
        # Calculate rolling stats (shift by 1 to avoid data leakage)
        all_team_games['form_last_5'] = all_team_games['result'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_goals_last_5'] = all_team_games['team_goals'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_goals_conceded_last_5'] = all_team_games['team_goals_conceded'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_xg_last_5'] = all_team_games['team_xg'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_xg_conceded_last_5'] = all_team_games['team_xg_conceded'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_poss_last_5'] = all_team_games['team_poss'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_shots_last_5'] = all_team_games['team_shots'].rolling(5, min_periods=1).mean().shift(1)
        all_team_games['avg_shots_on_target_last_5'] = all_team_games['team_shots_on_target'].rolling(5, min_periods=1).mean().shift(1)

        # Map back to original DataFrame - For home team
        home_matches = df_prep[df_prep['home_team'] == team]
        for idx in home_matches.index:
            match_date = df_prep.loc[idx, 'date']
            team_stats = all_team_games[all_team_games['date'] == match_date]
            if not team_stats.empty:
                df_prep.loc[idx, 'home_form_last_5'] = team_stats['form_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_goals_last_5'] = team_stats['avg_goals_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_goals_conceded_last_5'] = team_stats['avg_goals_conceded_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_xg_last_5'] = team_stats['avg_xg_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_xg_conceded_last_5'] = team_stats['avg_xg_conceded_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_poss_last_5'] = team_stats['avg_poss_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_shots_last_5'] = team_stats['avg_shots_last_5'].iloc[0]
                df_prep.loc[idx, 'home_avg_shots_on_target_last_5'] = team_stats['avg_shots_on_target_last_5'].iloc[0]
        
        # Map back to original DataFrame - For away team
        away_matches = df_prep[df_prep['away_team'] == team]
        for idx in away_matches.index:
            match_date = df_prep.loc[idx, 'date']
            team_stats = all_team_games[all_team_games['date'] == match_date]
            if not team_stats.empty:
                df_prep.loc[idx, 'away_form_last_5'] = team_stats['form_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_goals_last_5'] = team_stats['avg_goals_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_goals_conceded_last_5'] = team_stats['avg_goals_conceded_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_xg_last_5'] = team_stats['avg_xg_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_xg_conceded_last_5'] = team_stats['avg_xg_conceded_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_poss_last_5'] = team_stats['avg_poss_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_shots_last_5'] = team_stats['avg_shots_last_5'].iloc[0]
                df_prep.loc[idx, 'away_avg_shots_on_target_last_5'] = team_stats['avg_shots_on_target_last_5'].iloc[0]

    return df_prep

def encode_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode team names
    
    Args:
        df: DataFrame with team columns
        
    Returns:
        DataFrame with one-hot encoded teams
    """
    # Create dummy variables for the team columns separately
    dummies = pd.get_dummies(df[['home_team', 'away_team']], prefix=['home', 'away'], dtype=int)

    # Concatenate the new dummy columns with the original DataFrame
    df = pd.concat([df, dummies], axis=1)

    return df

def create_comparison_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison features between home and away teams
    
    Args:
        df: DataFrame with home and away rolling features
        
    Returns:
        DataFrame with comparison features added
    """
    df_with_comparisons = df.copy()
    
    # Form comparison
    df_with_comparisons['form_difference'] = df['home_form_last_5'] - df['away_form_last_5']

    # Performance comparisons  
    df_with_comparisons['goals_difference'] = df['home_avg_goals_last_5'] - df['away_avg_goals_last_5']
    df_with_comparisons['xg_difference'] = df['home_avg_xg_last_5'] - df['away_avg_xg_last_5']
    df_with_comparisons['poss_difference'] = df['home_avg_poss_last_5'] - df['away_avg_poss_last_5']

    # Defensive comparison
    df_with_comparisons['defensive_difference'] = df['away_avg_goals_conceded_last_5'] - df['home_avg_goals_conceded_last_5']
    
    return df_with_comparisons

def clean_data(df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Clean the dataset by dropping unnecessary columns
    
    Args:
        df: DataFrame to clean
        columns_to_drop: List of column names to drop
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns_to_drop is None:
        columns_to_drop = ['season', 'home_formation', 'away_formaation']
    
    # Drop specified columns if they exist
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    return df_clean

def process_season_data(filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for a single season
    
    Args:
        filepath: Path to the raw season CSV file
        output_path: Optional path to save the processed data
        
    Returns:
        Fully processed DataFrame ready for modeling
    """
    print("Loading and preparing data...")
    df = load_and_prepare_data(filepath)
    
    print("Creating rolling features...")
    df_with_rolling = create_rolling_features(df)
    
    print("Encoding teams...")
    df_encoded = encode_teams(df_with_rolling)
    
    print("Creating comparison features...")
    df_with_comparisons = create_comparison_features(df_encoded)
    
    print("Cleaning data...")
    df_final = clean_data(df_with_comparisons)
    
    if output_path:
        print(f"Saving processed data to {output_path}")
        df_final.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete! Dataset shape: {df_final.shape}")
    
    return df_final

def main():
    """
    Example usage of the feature engineering pipeline.
    Processes all CSV files in the preprocessed directory.
    """
    # Set pandas display options
    pd.set_option('display.max_columns', None)
    
    # Define directories
    input_dir = "data/02_preprocessed/"
    output_dir = "data/03_feature_engineered/"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # List all csv files in the input directory
        files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if not files_to_process:
            print(f"No CSV files found in '{input_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # Process each file
    for filename in files_to_process:
        input_file = os.path.join(input_dir, filename)
        
        # Create a corresponding output filename
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_engineered{extension}"
        output_file = os.path.join(output_dir, output_filename)
        
        print(f"\n{'='*20} PROCESSING: {filename} {'='*20}")
        
        try:
            df_processed = process_season_data(input_file, output_file)
            
            # Display basic info about the processed data
            print("\n--- DATA SUMMARY ---")
            print(f"Shape: {df_processed.shape}")
            print(f"Missing values: {df_processed.isnull().sum().sum()}")
            print(f"Output saved to: {output_file}")
            
        except Exception as e:
            print(f"!!! Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()