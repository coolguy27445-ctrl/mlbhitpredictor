import streamlit as st
import pandas as pd
import numpy as np
import datetime
import statsapi

st.set_page_config(page_title="MLB Daily Hit Predictor", page_icon="⚾", layout="wide")

# ==========================================
# 1. CORE ALGORITHM
# ==========================================
def logodds(p):
    return np.log(p / (1 - p))

def probability(odds):
    return 1 / (1 + np.exp(-odds))

def predict_daily_hit_probability(player_data, matchup_data, environment_data):
    base_ba = player_data["batting_average"]
    contact_mod = player_data["contact_rate"] - player_data["whiff_rate"]
    
    # Scale H/AB to an approximate H/PA baseline
    base_h_per_pa = base_ba * 0.90 + (contact_mod * 0.05)
    
    # Check bounds safety before logodds conversion
    base_h_per_pa = max(0.01, min(0.99, base_h_per_pa))
    current_log_odds = logodds(base_h_per_pa)
    
    # Platoon Hand Adjustments
    pitcher_hand = matchup_data["pitcher_hand"]
    split_ba = player_data["vs_LHP"] if pitcher_hand == "L" else player_data["vs_RHP"]
    current_log_odds += ((split_ba - base_ba) * 2.5)
    
    # Context Factors
    current_log_odds += np.log(environment_data["park_factor"])
    current_log_odds += (environment_data["temperature"] - 70) * 0.0015
    
    if environment_data["wind_direction"] == "out": current_log_odds += 0.05
    elif environment_data["wind_direction"] == "in": current_log_odds -= 0.05
    
    adjusted_pa_hit_prob = probability(current_log_odds)
    
    projected_pa = matchup_data["projected_pa"]
    if projected_pa < 3:
        return 0.0
        
    prob_zero_hits = (1 - adjusted_pa_hit_prob) ** projected_pa
    return round((1 - prob_zero_hits) * 100, 1)

# ==========================================
# 2. LIVE PRODUCTION DATA & STATS ENGINE
# ==========================================
def fetch_real_player_stats(player_id):
    """Fetches real-time season batting averages and volume stats."""
    try:
        # Pull live season stats directly from the active wrapper (defaults to current season)
        stats = statsapi.player_stat_data(player_id, group="hitting", type="season")
        
        ba = 0.250
        games_played = 0
        at_bats = 0
        
        # Safely parse stats and extract playing time volume
        if stats and 'stats' in stats and len(stats['stats']) > 0:
            hitting_stats = stats['stats'][0].get('stats', {})
            avg_str = hitting_stats.get('avg', '0.250')
            try:
                ba = float(avg_str)
            except ValueError:
                ba = 0.250 
                
            games_played = int(hitting_stats.get('gamesPlayed', 0))
            at_bats = int(hitting_stats.get('atBats', 0))
        
        # Default splits to overall BA
        vs_LHP = ba
        vs_RHP = ba
        
        # Query true splits
        try:
            splits_data = statsapi.get('people', {
                'personIds': player_id, 
                'hydrate': 'stats(group=[hitting],type=[statSplits])'
            })
            
            person_data = splits_data.get('people', [{}])[0]
            for stat_block in person_data.get('stats', []):
                if stat_block.get('type', {}).get('displayName') == 'statSplits':
                    for split in stat_block.get('splits', []):
                        code = split.get('split', {}).get('code', '')
                        split_avg_str = split.get('stat', {}).get('avg', str(ba))
                        
                        try:
                            split_avg = float(split_avg_str)
                        except ValueError:
                            split_avg = ba
                            
                        if code == 'vl': vs_LHP = split_avg
                        elif code == 'vr': vs_RHP = split_avg
        except Exception:
            pass 

        return {
            "batting_average": ba,
            "contact_rate": 0.78, 
            "whiff_rate": 0.22,  
            "vs_LHP": vs_LHP,
            "vs_RHP": vs_RHP,
            "games_played": games_played,
            "at_bats": at_bats
        }
    except Exception:
        # Final fallback
        return {
            "batting_average": 0.250, "contact_rate": 0.75, "whiff_rate": 0.25, 
            "vs_LHP": 0.250, "vs_RHP": 0.250, "games_played": 0, "at_bats": 0
        }

@st.cache_data(ttl=1800) 
def load_live_predictions(date_str):
    schedule_data = statsapi.get('schedule', {'date': date_str, 'sportId': 1, 'hydrate': 'probablePitcher,lineups'})
    master_predictions = []

    for date_obj in schedule_data.get('dates', []):
        for game in date_obj.get('games', []):
            if game.get('status', {}).get('abstractGameState') == 'Finalized' or 'Postponed' in game.get('status', {}).get('detailedState', ''):
                continue
                
            game_id = game['gamePk']
            live_feed = statsapi.get('game', {'gamePk': game_id})
            
            weather_info = live_feed.get('gameData', {}).get('weather', {})
            temp = int(weather_info.get('temp', 70))
            wind_str = weather_info.get('wind', '0 mph').lower()
            
            wind_direction = "neutral"
            if "out" in wind_str: wind_direction = "out"
            elif "in" in wind_str: wind_direction = "in"
            
            environment = {"park_factor": 1.00, "temperature": temp, "wind_direction": wind_direction}
            
            teams_data = live_feed.get('gameData', {}).get('probablePitchers', {})
            pitcher_hands = {'home': 'R', 'away': 'R'}
            for side in ['home', 'away']:
                p_id = teams_data.get(side, {}).get('id')
                if p_id:
                    p_info = statsapi.get('person', {'personId': p_id})
                    pitcher_hands[side] = p_info.get('people', [{}])[0].get('pitchHand', {}).get('code', 'R')

            lineups = live_feed.get('liveData', {}).get('boxscore', {}).get('teams', {})
            
            for team_side, opp_pitcher_hand in [('away', pitcher_hands['home']), ('home', pitcher_hands['away'])]:
                batting_order = lineups.get(team_side, {}).get('battingOrder', [])
                
                for index, player_id in enumerate(batting_order):
                    # 1. FILTER: Must be a starter (Top 9 in the batting order)
                    if index >= 9: 
                        continue 
                    
                    player_info = statsapi.get('person', {'personId': player_id})
                    if not player_info.get('people'): continue
                    player_name = player_info['people'][0]['fullName']
                    
                    player_stats = fetch_real_player_stats(player_id)
                    
                    # 2. FILTER: Must average at least 3 At-Bats per game this season
                    games = player_stats.get("games_played", 0)
                    abs_total = player_stats.get("at_bats", 0)
                    
                    if games == 0:
                        continue # Exclude players with no appearances
                        
                    ab_per_game = abs_total / games
                    if ab_per_game < 3.0:
                        continue # Exclude rotational players / defensive replacements
                        
                    matchup = {"pitcher_hand": opp_pitcher_hand, "projected_pa": 5 if index < 4 else 4}
                    prob = predict_daily_hit_probability(player_stats, matchup, environment)
                    
                    master_predictions.append({
                        "Order": index + 1,
                        "Player": player_name,
                        "Team": game['teams'][team_side]['team']['name'],
                        "Opp Pitcher Hand": opp_pitcher_hand,
                        "Proj PA": matchup["projected_pa"],
                        "Season BA": player_stats["batting_average"],
                        "AB/Game": round(ab_per_game, 1),
                        "Hit Probability (%)": prob
                    })
                    
    return pd.DataFrame(master_predictions)

# ==========================================
# 3. STREAMLIT USER INTERFACE
# ==========================================
st.title("⚾ MLB Daily Hit Probability Dashboard")
st.markdown("This tracker displays precise calculations strictly for **starting players** who average **at least 3.0 At-Bats per game** this season.")

selected_date = st.sidebar.date_input("Select Games Date", datetime.date.today())
date_formatted = selected_date.strftime("%m/%d/%Y")
min_prob = st.sidebar.slider("Minimum Hit Probability (%)", 0, 100, 50)

with st.spinner("Connecting to MLB network data streams..."):
    df_predictions = load_live_predictions(date_formatted)

if not df_predictions.empty:
    filtered_df = df_predictions[df_predictions["Hit Probability (%)"] >= min_prob]
    filtered_df = filtered_df.sort_values(by="Hit Probability (%)", ascending=False).reset_index(drop=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Qualified Starters Tracked", len(df_predictions))
    col2.metric("Highest Hit Chance Today", f"{filtered_df['Hit Probability (%)'].max()}%" if not filtered_df.empty else "N/A")
    col3.metric("Starters Meeting Threshold", len(filtered_df))
        
    st.markdown("---")
    
    search_query = st.text_input("🔍 Search Hitter or Franchise:")
    if search_query:
        filtered_df = filtered_df[
            filtered_df['Player'].str.contains(search_query, case=False) | 
            filtered_df['Team'].str.contains(search_query, case=False)
        ]

    st.subheader(f"Hit Projections for {date_formatted}")
    st.dataframe(
        filtered_df,
        column_config={
            "Hit Probability (%)": st.column_config.ProgressColumn(
                "Hit Probability (%)",
                format="%f%%",
                min_value=0,
                max_value=100,
            ),
            "Season BA": st.column_config.NumberColumn("Season BA", format="%.3f"),
            "AB/Game": st.column_config.NumberColumn("AB/Game", format="%.1f")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("No dynamic games scheduled or official active lineups have not been submitted to the feed yet.")