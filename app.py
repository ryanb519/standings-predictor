import streamlit as st
import pandas as pd
import io
from typing import List
from datetime import datetime
import requests

def color_metric_diverging(series, higher_is_better=True):
    """
    Creates Green → White → Red gradient for any metric.
    
    - higher_is_better=True  → high values = green, low = red
    - higher_is_better=False → low values = green, high = red
    """
    
    # Reverse values when "lower is better" so we can reuse one scale
    values = series.copy()
    if not higher_is_better:
        values = values.max() - values  # invert scale

    max_val = values.max()
    min_val = values.min()
    mid_val = (max_val + min_val) / 2

    colors = []
    for v in values:
        if v <= mid_val:
            # Green → White
            scale = (v - min_val) / (mid_val - min_val)
            r = int(255 * scale)
            g = 255
            b = int(255 * scale)
        else:
            # White → Red
            scale = (v - mid_val) / (max_val - mid_val)
            r = 255
            g = int(255 * (1 - scale))
            b = int(255 * (1 - scale))

        colors.append(f"background-color: rgba({r}, {g}, {b}, 0.6);")

    return colors
    
def format_standings_df(df):
    # Whole-number columns
    whole_cols = [
        "R", "R_Rank", "HR", "HR_Rank", "RBI", "RBI_Rank", "SB", "SB_Rank",
        "Grand_Total_Score", "Hitter_Score", "Pitcher_Score",
        "W", "W_Rank", "SO", "SO_Rank", "SV", "SV_Rank"
    ]
    for col in whole_cols:
        if col in df.columns:
            df[col] = df[col].round(0).astype("Int64")

    # ERA / WHIP → 2 decimals
    for col in ["ERA", "WHIP"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # AVG → 3 decimals, no leading zero
    if "AVG" in df.columns:
        df["AVG"] = df["AVG"].apply(lambda x: f"{x:.3f}".lstrip("0") if pd.notnull(x) else x)

    return df

def getAggregateHitterProj(list):
  df = pd.DataFrame()
  for i in list:
    data = "https://raw.githubusercontent.com/ryanb519/standings-predictor/main/data/"+i+"_hitters.csv"
    projDF = pd.read_csv(data)
    projDF['#Proj'] = 1
    df = pd.concat([df,projDF])

  aggDF = df.groupby(by=['xMLBAMID','playerid','PlayerName'], as_index=False).agg({
    'PA':'mean','AB':'mean','AVG':'mean','R':'mean','HR':'mean','RBI':'mean','SB':'mean',
    'G':'mean','H':'mean','1B':'mean','2B':'mean','3B':'mean','BB':'mean','IBB':'mean','SO':'mean','HBP':'mean','SF':'mean',
    'SH':'mean','GDP':'mean','CS':'mean','OBP':'mean','SLG':'mean','OPS':'mean','wOBA':'mean','BB%':'mean','K%':'mean',
    'ISO':'mean','BABIP':'mean','wRC+':'mean','WAR':'mean','Team':'min','#Proj':'sum'})#,'Source':'transform(lambda x: '',''.join(x))'})
  aggDF.rename(columns={'playerid':'FanGraphsID','PlayerName':'Name','xMLBAMID':'MLBAMID'},inplace=True)
  return aggDF

def getAggregatePitcherProj(list):
  df = pd.DataFrame()
  for i in list:
    data = "https://raw.githubusercontent.com/ryanb519/standings-predictor/main/data/"+i+"_pitchers.csv"
    projDF = pd.read_csv(data)
    projDF['#Proj'] = 1
    df = pd.concat([df,projDF])

  aggDF = df.groupby(by=['xMLBAMID','playerid','PlayerName'], as_index=False).agg({
    'IP':'mean','W':'mean','ERA':'mean','WHIP':'mean','SO':'mean','SV':'mean','L':'mean','GS':'mean','G':'mean',
    'HLD':'mean','BS':'mean','TBF':'mean','H':'mean','R':'mean','ER':'mean','HR':'mean','BB':'mean','IBB':'mean',
    'HBP':'mean','HR/9':'mean','K%':'mean','BB%':'mean','K-BB%':'mean','GB%':'mean','AVG':'mean','BABIP':'mean',
    'LOB%':'mean','FIP':'mean','WAR':'mean','QS':'mean','Team':'min','#Proj':'sum'})#,'Source':'transform(lambda x: '',''.join(x))'})'Source':'mean',
  aggDF.rename(columns={'playerid':'FanGraphsID','PlayerName':'Name','xMLBAMID':'MLBAMID'},inplace=True)
  return aggDF

def calculate_standings(picks_df, projections, starters):

  # 1. Reads in the dataframes
  hitter_df = getAggregateHitterProj(projections)
  pitcher_df = getAggregatePitcherProj(projections)
  #picks_df = pd.read_csv(picks, sep='\t')

  # Function to transform 'Firstname Lastname' to 'Lastname, Firstname'
  def transform_player_name(name):
    if not isinstance(name, str) or name.strip() == "":
        return name  # return as-is for blank/invalid input

    suffixes = {"Jr.", "Sr.", "II", "III", "IV", "V"}

    parts = name.strip().split()

    # No transformation possible
    if len(parts) == 1:
        return name

    # Check if last token is a suffix
    if parts[-1] in suffixes:
        last_name = " ".join(parts[-2:])      # e.g. "Witt Jr."
        first_name = " ".join(parts[:-2])     # e.g. "Bobby"
    else:
        last_name = parts[-1]
        first_name = " ".join(parts[:-1])

    return f"{last_name}, {first_name}"

  # Apply the transformation to the projection dataframes
  hitter_df['Player'] = hitter_df['Name'].apply(transform_player_name)
  pitcher_df['Player'] = pitcher_df['Name'].apply(transform_player_name)

  # 2. Merges projection data with picks.tsv based on the player name
  hitter_picks_df = picks_df[picks_df['Position'] != 'P'].merge(
      hitter_df, on='Player', how='inner', suffixes=('_picks', '_proj')
  ).rename(columns={'Team_picks': 'DraftTeam'})

  pitcher_picks_df = picks_df[picks_df['Position'] == 'P'].merge(
      pitcher_df, on='Player', how='inner', suffixes=('_picks', '_proj')
  ).rename(columns={'Team_picks': 'DraftTeam'})


  # --- 2.5. Apply player limits based on draft order (Pick) ---
  # Hitter Limit: First 14 non-pitchers
  if starters:
    hitter_picks_df = hitter_picks_df.sort_values(by=['DraftTeam', 'Pick'])
    hitter_picks_df['Hitter_Count'] = hitter_picks_df.groupby('DraftTeam').cumcount() + 1
    hitter_picks_df = hitter_picks_df[hitter_picks_df['Hitter_Count'] <= 14]
    hitter_picks_df = hitter_picks_df.drop(columns=['Hitter_Count'])

    # Pitcher Limit: First 9 pitchers
    pitcher_picks_df = pitcher_picks_df.sort_values(by=['DraftTeam', 'Pick'])
    pitcher_picks_df['Pitcher_Count'] = pitcher_picks_df.groupby('DraftTeam').cumcount() + 1
    pitcher_picks_df = pitcher_picks_df[pitcher_picks_df['Pitcher_Count'] <= 9]
    pitcher_picks_df = pitcher_picks_df.drop(columns=['Pitcher_Count'])


  # 3. Calculates projected statistics for each Team (HITTER)
  hitter_stats_df = hitter_picks_df.groupby('DraftTeam').agg(
      R=('R', 'sum'), RBI=('RBI', 'sum'), SB=('SB', 'sum'), HR=('HR', 'sum'),
      H=('H', 'sum'), AB=('AB', 'sum')
  ).reset_index()

  # Calculate AVG (weighted average)
  hitter_stats_df['AVG'] = hitter_stats_df['H'] / hitter_stats_df['AB']
  hitter_stats_df = hitter_stats_df.drop(columns=['H', 'AB'])

  # 4. Creates projected fantasy baseball standings (HITTER)
  hitter_standings = hitter_stats_df[['DraftTeam']].copy()
  hitter_cols = ['R', 'RBI', 'SB', 'HR', 'AVG']

  # Rank (highest is best)
  for col in hitter_cols:
      hitter_standings[f'{col}_Rank'] = hitter_stats_df[col].rank(method='max', ascending=True)

  # Calculate Total Score
  hitter_standings['Hitter_Score'] = hitter_standings.filter(like='_Rank').sum(axis=1)

  # Combine Hitter Stats and Ranks
  hitter_combined_df = pd.merge(hitter_stats_df, hitter_standings, on='DraftTeam', how='inner')


  # 5. Calculates projected statistics for each Team (PITCHER)
  pitcher_stats_df = pitcher_picks_df.groupby('DraftTeam').agg(
      W=('W', 'sum'), SV=('SV', 'sum'), SO=('SO', 'sum'),
      ER=('ER', 'sum'), IP=('IP', 'sum'), H=('H', 'sum'), BB=('BB', 'sum')
  ).reset_index()

  # Calculate ERA and WHIP (weighted averages)
  pitcher_stats_df['ERA'] = (pitcher_stats_df['ER'] / pitcher_stats_df['IP']) * 9
  pitcher_stats_df['WHIP'] = (pitcher_stats_df['H'] + pitcher_stats_df['BB']) / pitcher_stats_df['IP']

  # 6. Creates projected fantasy baseball standings (PITCHER)
  pitcher_standings = pitcher_stats_df[['DraftTeam']].copy()
  pitcher_cols = ['W', 'SV', 'SO', 'ERA', 'WHIP']

  # Rank W, SV, SO (highest is best)
  for col in ['W', 'SV', 'SO']:
      pitcher_standings[f'{col}_Rank'] = pitcher_stats_df[col].rank(method='min', ascending=True)

  # Rank ERA, WHIP (lowest is best)
  for col in ['ERA', 'WHIP']:
      pitcher_standings[f'{col}_Rank'] = pitcher_stats_df[col].rank(method='min', ascending=False)

  # Calculate Total Score
  pitcher_standings['Pitcher_Score'] = pitcher_standings.filter(like='_Rank').sum(axis=1)

  # Combine Pitcher Stats and Ranks
  pitcher_combined_df = pd.merge(pitcher_stats_df, pitcher_standings, on='DraftTeam', how='inner')


  # 7. Dumps all projected standings in a final dataframe including raw totals
  final_standings = pd.merge(
      hitter_combined_df,
      pitcher_combined_df.drop(columns=['ER', 'IP', 'H', 'BB']), # Drop intermediate columns
      on='DraftTeam',
      how='outer'
  )

  # Calculate Grand Total Score and Rank
  final_standings['Grand_Total_Score'] = final_standings['Hitter_Score'] + final_standings['Pitcher_Score']
  final_standings['Overall_Rank'] = final_standings['Grand_Total_Score'].rank(method='min', ascending=False).astype(int)

  # Define and apply column order
  output_cols = ['Overall_Rank', 'DraftTeam', 'Grand_Total_Score', 'Hitter_Score', 'Pitcher_Score']
  category_cols = ['R', 'RBI', 'SB', 'HR', 'AVG', 'W', 'SV', 'SO', 'ERA', 'WHIP']
  for col in category_cols:
      output_cols.append(col)
      output_cols.append(f'{col}_Rank')

  final_standings = final_standings[output_cols]

  # Sort by Overall Rank and save to a new file
  final_standings = final_standings.sort_values(by='Overall_Rank').reset_index(drop=True)
  final_standings.to_csv('projected_fantasy_standings_with_totals.csv', index=False)
  return final_standings

def get_hitters(picks_df, projections, starters):

  # 1. Reads in the dataframes
  hitter_df = getAggregateHitterProj(projections)

  # Function to transform 'Firstname Lastname' to 'Lastname, Firstname'
  def transform_player_name(name):
    if not isinstance(name, str) or name.strip() == "":
        return name  # return as-is for blank/invalid input

    suffixes = {"Jr.", "Sr.", "II", "III", "IV", "V"}

    parts = name.strip().split()

    # No transformation possible
    if len(parts) == 1:
        return name

    # Check if last token is a suffix
    if parts[-1] in suffixes:
        last_name = " ".join(parts[-2:])      # e.g. "Witt Jr."
        first_name = " ".join(parts[:-2])     # e.g. "Bobby"
    else:
        last_name = parts[-1]
        first_name = " ".join(parts[:-1])

    return f"{last_name}, {first_name}"

  # Apply the transformation to the projection dataframes
  hitter_df['Player'] = hitter_df['Name'].apply(transform_player_name)

  # 2. Merges projection data with picks.tsv based on the player name
  hitter_picks_df = picks_df[picks_df['Position'] != 'P'].merge(
      hitter_df, on='Player', how='inner', suffixes=('_picks', '_proj')
  ).rename(columns={'Team_picks': 'DraftTeam'})

  # --- 2.5. Apply player limits based on draft order (Pick) ---
  # Hitter Limit: First 14 non-pitchers
  if starters:
    hitter_picks_df = hitter_picks_df.sort_values(by=['DraftTeam', 'Pick'])
    hitter_picks_df['Hitter_Count'] = hitter_picks_df.groupby('DraftTeam').cumcount() + 1
    hitter_picks_df = hitter_picks_df[hitter_picks_df['Hitter_Count'] <= 14]
    hitter_picks_df = hitter_picks_df.drop(columns=['Hitter_Count'])

  return hitter_picks_df[['DraftTeam','Round','Pick','Player','Position','PA','R','HR','RBI','AVG','SB']].sort_values(by='Pick')

def get_pitchers(picks_df, projections, starters):

  # 1. Reads in the dataframes
  pitcher_df = getAggregatePitcherProj(projections)

  # Function to transform 'Firstname Lastname' to 'Lastname, Firstname'
  def transform_player_name(name):
    if not isinstance(name, str) or name.strip() == "":
        return name  # return as-is for blank/invalid input

    suffixes = {"Jr.", "Sr.", "II", "III", "IV", "V"}

    parts = name.strip().split()

    # No transformation possible
    if len(parts) == 1:
        return name

    # Check if last token is a suffix
    if parts[-1] in suffixes:
        last_name = " ".join(parts[-2:])      # e.g. "Witt Jr."
        first_name = " ".join(parts[:-2])     # e.g. "Bobby"
    else:
        last_name = parts[-1]
        first_name = " ".join(parts[:-1])

    return f"{last_name}, {first_name}"

  # Apply the transformation to the projection dataframes
  pitcher_df['Player'] = pitcher_df['Name'].apply(transform_player_name)

  # 2. Merges projection data with picks.tsv based on the player name
  pitcher_picks_df = picks_df[picks_df['Position'] == 'P'].merge(
      pitcher_df, on='Player', how='inner', suffixes=('_picks', '_proj')
  ).rename(columns={'Team_picks': 'DraftTeam'})


  # --- 2.5. Apply player limits based on draft order (Pick) ---
  if starters:
    # Pitcher Limit: First 9 pitchers
    pitcher_picks_df = pitcher_picks_df.sort_values(by=['DraftTeam', 'Pick'])
    pitcher_picks_df['Pitcher_Count'] = pitcher_picks_df.groupby('DraftTeam').cumcount() + 1
    pitcher_picks_df = pitcher_picks_df[pitcher_picks_df['Pitcher_Count'] <= 9]
    pitcher_picks_df = pitcher_picks_df.drop(columns=['Pitcher_Count'])

  return pitcher_picks_df[['DraftTeam','Round','Pick','Player','Position','IP','W','SV','ERA','WHIP','SO']].sort_values(by='Pick')
  
st.set_page_config(page_title="Fantasy Baseball Standings (2026)", layout="wide")

# -----------------------
# UI
# -----------------------
st.title("Fantasy Baseball — Projected Standings (2026)")
st.markdown(
    """
Upload your draft picks (.tsv), select which projection systems to use, choose whether to score
entire rosters or starters-only, then click **Calculate Projected Standings**.
"""
)

with st.expander("Instructions / Notes", expanded=False):
    st.write("""
    - Upload a draft picks file in **TSV** format (tab-separated). The UI will preview the first 10 rows.
    - Select one or more projection systems using the checkboxes.
    - Choose whether to calculate the standings using the entire roster or only starters
      (starters defined as first 9 pitchers, first 14 hitters).
    - Click **Calculate Projected Standings** to run the engine.
    """)

# Layout: left column for inputs, right column for outputs
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Inputs")

    uploaded = st.file_uploader("Upload Draft Picks (.tsv)", type=["tsv"], accept_multiple_files=False)
    picks_df = None
    if uploaded is not None:
        # Validate extension (extra check)
        filename = uploaded.name.lower()
        if not filename.endswith(".tsv"):
            st.error("Please upload a file with a .tsv extension.")
        else:
            try:
                # read as TSV
                uploaded.seek(0)
                picks_df = pd.read_csv(uploaded, sep="\t", dtype=str)
                picks_df = picks_df[picks_df['Player'] != '-'] #Remove picks without players yet
                st.success(f"Loaded `{uploaded.name}` — {picks_df.shape[0]:,} rows, {picks_df.shape[1]:,} columns.")
                st.write("Preview (first 10 rows):")
                st.dataframe(picks_df.head(10))
            except Exception as e:
                st.error(f"Could not read the TSV file: {e}")

    st.markdown("---")
    st.subheader("Projection Systems")
    # Six individual checkboxes as requested
    st.write("Select projection systems to include:")
    col_a, col_b = st.columns(2)
    with col_a:
        use_steamer = st.checkbox("Steamer", value=False)
        use_fgdc = st.checkbox("FanGraphs Depth Charts", value=True)
        use_thebat = st.checkbox("The Bat", value=False)
    with col_b:
        use_thebatx = st.checkbox("The Bat X", value=False)
        use_oopsy = st.checkbox("OOPSY", value=False)
        use_atc = st.checkbox("ATC", value=False)

    selected_systems = []
    if use_steamer: selected_systems.append("steamer")
    if use_fgdc: selected_systems.append("fangraphsdc")
    if use_thebat: selected_systems.append("thebat")
    if use_thebatx: selected_systems.append("thebatx")
    if use_oopsy: selected_systems.append("oopsy")
    if use_atc: selected_systems.append("atc")

    st.markdown("---")
    st.subheader("Roster Option")
    roster_option = st.radio(
        "Calculate standings based on:",
        ("Starters only (first 9 pitchers, 14 hitters)","Entire roster"),
        index=0
    )
    starters_only = True
    starters_only = roster_option.startswith("Starters")

    st.markdown("---")
    st.write("Calculation control")
    # The Calculate button
    calculate_button = st.button("Calculate Projected Standings")

    st.markdown("---")

with right_col:
    st.header("Projected Standings")
    results_placeholder=st.empty()

    if calculate_button:
        if uploaded is None:
            st.warning("Please upload a TSV draft file first.")
        elif len(selected_systems) == 0:
            st.warning("Please select at least one projection system.")
        else:
            standings_df = calculate_standings(picks_df, selected_systems, starters_only)
            hitter_picks_df = get_hitters(picks_df, selected_systems, starters_only)
            pitcher_picks_df = get_pitchers(picks_df, selected_systems, starters_only)

            standings_df = format_standings_df(standings_df)

            st.session_state["standings_df"] = standings_df
            st.session_state["hitter_picks_df"] = hitter_picks_df
            st.session_state["pitcher_picks_df"] = pitcher_picks_df

            if "standings_df" in st.session_state:
                standings_df = st.session_state["standings_df"]
                hitter_picks_df = st.session_state["hitter_picks_df"]
                pitcher_picks_df = st.session_state["pitcher_picks_df"]
    
                # -----------------------
                # APPLY DISPLAY ROUNDING
                # -----------------------
                whole_number_columns = [
                    "R", "R_Rank",
                    "HR", "HR_Rank",
                    "RBI", "RBI_Rank",
                    "SB", "SB_Rank",
                    "Grand_Total_Score",
                    "Hitter_Score",
                    "Pitcher_Score",
                    "W", "W_Rank",
                    "SO", "SO_Rank",
                    "SV", "SV_Rank"
                ]

                # Round whole-number columns
                for col in whole_number_columns:
                    if col in standings_df.columns:
                        standings_df[col] = pd.to_numeric(standings_df[col], errors="coerce").round(0).astype("Int64")

                # Round ERA & WHIP to 2 decimals
                if "ERA" in standings_df.columns:
                    standings_df["ERA"] = (
                        pd.to_numeric(standings_df["ERA"], errors="coerce")
                        .round(2)
                    )

                if "WHIP" in standings_df.columns:
                    standings_df["WHIP"] = (
                        pd.to_numeric(standings_df["WHIP"], errors="coerce")
                        .round(2)
                    )

                # Format AVG to 3 decimals and remove leading zero
                if "AVG" in standings_df.columns:
                    standings_df["AVG"] = (
                        pd.to_numeric(standings_df["AVG"], errors="coerce")
                        .round(3)
                        .map(lambda x: f"{x:.3f}".lstrip("0") if pd.notnull(x) else "")
                    )
                    
                # Apply red-green color formatting to Standings
                rank_cols = ["R", "HR", "RBI", "SB", "W", "SO", "SV"]
                standings_df = standings_df.style.apply(color_metric_diverging, higher_is_better=True,subset=rank_cols)
                standings_df = standings_df.style.apply(color_metric_diverging, higher_is_better=False,subset=["ERA", "WHIP"])
                
                # -----------------------
                # DISPLAY THE TABLE
                # -----------------------
                standings_df = standings_df.drop(columns=['R_Rank','RBI_Rank','HR_Rank','SB_Rank','AVG_Rank','ERA_Rank','WHIP_Rank','SO_Rank','W_Rank','SV_Rank'])
                standings_df = standings_df.rename(columns={'Overall_Rank':'Rank','DraftTeam':'Team','Grand_Total_Score':'Total Points','Hitter_Score':'Hitters','Pitcher_Score':'Pitchers'})

                results_placeholder.dataframe(standings_df, use_container_width=True, hide_index=True, height=575)
    
                # -----------------------
                # TEAM DETAIL SECTION
                # -----------------------
                st.subheader("Team Detail")
    
                teams = sorted(standings_df["Team"].unique())
                selected_team = st.selectbox("Select a Team", teams)
    
                # Filter team-specific DFS
                team_hitters = hitter_picks_df[hitter_picks_df["DraftTeam"] == selected_team].copy()
                team_pitchers = pitcher_picks_df[pitcher_picks_df["DraftTeam"] == selected_team].copy()
    
                # Format hitter AVG
                if "AVG" in team_hitters.columns:
                    team_hitters["AVG"] = team_hitters["AVG"].apply(
                        lambda x: f"{x:.3f}".lstrip("0") if pd.notnull(x) else x
                    )
    
                # HITTERS TABLE
                st.markdown("### Hitters")
                st.dataframe(team_hitters, hide_index=True, use_container_width=True)
    
                # PITCHERS TABLE
                st.markdown("### Pitchers")
                st.dataframe(team_pitchers, hide_index=True, use_container_width=True)
    
'''
# -----------------------
# Perform calculation when button pressed
# -----------------------
if calculate_button:
    if uploaded is None or picks_df is None:
        st.warning("Please upload a valid .tsv draft picks file before calculating.")
    elif len(selected_systems) == 0:
        st.warning("Please select at least one projection system.")
    else:
        with st.spinner("Calculating projected standings..."):
            try:
                standings_df = calculate_standings(picks_df, selected_systems, starters_only)
                hitter_picks_df = get_hitters(picks_df, selected_systems, starters_only)
                pitcher_picks_df = get_pitchers(picks_df, selected_systems, starters_only)

                if not isinstance(standings_df, pd.DataFrame):
                    raise ValueError("calculate_standings did not return a pandas DataFrame.")

                # -----------------------
                # APPLY DISPLAY ROUNDING
                # -----------------------
                whole_number_columns = [
                    "R", "R_Rank",
                    "HR", "HR_Rank",
                    "RBI", "RBI_Rank",
                    "SB", "SB_Rank",
                    "Grand_Total_Score",
                    "Hitter_Score",
                    "Pitcher_Score",
                    "W", "W_Rank",
                    "SO", "SO_Rank",
                    "SV", "SV_Rank"
                ]

                # Round whole-number columns
                for col in whole_number_columns:
                    if col in standings_df.columns:
                        standings_df[col] = pd.to_numeric(standings_df[col], errors="coerce").round(0).astype("Int64")

                # Round ERA & WHIP to 2 decimals
                if "ERA" in standings_df.columns:
                    standings_df["ERA"] = (
                        pd.to_numeric(standings_df["ERA"], errors="coerce")
                        .round(2)
                    )

                if "WHIP" in standings_df.columns:
                    standings_df["WHIP"] = (
                        pd.to_numeric(standings_df["WHIP"], errors="coerce")
                        .round(2)
                    )

                # Format AVG to 3 decimals and remove leading zero
                if "AVG" in standings_df.columns:
                    standings_df["AVG"] = (
                        pd.to_numeric(standings_df["AVG"], errors="coerce")
                        .round(3)
                        .map(lambda x: f"{x:.3f}".lstrip("0") if pd.notnull(x) else "")
                    )

              

                # -----------------------
                # DISPLAY THE TABLE
                # -----------------------
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                last_run_info.markdown(
                    f"**Last calculated:** {now} — using: {', '.join(selected_systems)}"
                )
                standings_df = standings_df.drop(columns=['R_Rank','RBI_Rank','HR_Rank','SB_Rank','AVG_Rank','ERA_Rank','WHIP_Rank','SO_Rank','W_Rank','SV_Rank'])
                standings_df = standings_df.rename(columns={'Overall_Rank':'Rank','DraftTeam':'Team','Grand_Total_Score':'Total Points','Hitter_Score':'Hitters','Pitcher_Score':'Pitchers'})

                results_placeholder.dataframe(standings_df, use_container_width=True, hide_index=True, height=600)

                # -----------------------
                # DOWNLOAD BUTTON
                # -----------------------
                csv_bytes = standings_df.to_csv(index=False).encode("utf-8")
                download_placeholder.download_button(
                    label="Download Standings as CSV",
                    data=csv_bytes,
                    file_name="projected_standings.csv",
                    mime="text/csv"
                )

                # ---------------------------------------------------------
                # TEAM DETAIL SECTION (right column, below standings)
                # ---------------------------------------------------------
                
                # All teams from both dataframes
                teams = sorted(
                    set(hitter_picks_df["DraftTeam"].unique().tolist() +
                        pitcher_picks_df["DraftTeam"].unique().tolist())
                )
                
                team_selected = st.selectbox("Select Team", teams)
                
                # =========================================================
                #                     HITTER DETAILS
                # =========================================================
                
                st.markdown("### 🥎 Hitters")
                
                hitters = hitter_picks_df[hitter_picks_df["DraftTeam"] == team_selected].copy()
                
                h_cols = ["Pick", "Player", "Position", "PA", "R", "HR", "AVG", "RBI", "SB"]
                df_hitters = hitters[h_cols].copy()
                
                # Format AVG (3 decimal places, no leading zero)
                df_hitters["AVG"] = df_hitters["AVG"].round(3).astype(str)
                df_hitters["AVG"] = df_hitters["AVG"].str.replace("0.", ".", regex=False)
                
                # Round counting stats
                for col in ["PA", "R", "HR", "RBI", "SB"]:
                    df_hitters[col] = df_hitters[col].round(0).astype("Int64")
                
                # Totals row
                h_totals = df_hitters.drop(columns=["Pick", "Player", "Position"], errors="ignore") \
                                     .apply(pd.to_numeric, errors="coerce").sum()
                
                h_total_row = {col: "" for col in df_hitters.columns}
                for col in h_totals.index:
                    h_total_row[col] = h_totals[col]
                
                # Team AVG mean
                team_avg = hitters["AVG"].mean()
                h_total_row["AVG"] = "." + str(round(team_avg, 3))[2:]
                
                df_hitters = pd.concat([df_hitters, pd.DataFrame([h_total_row])], ignore_index=True)
                
                hitter_placeholder.dataframe(df_hitters, use_container_width=True)
                
                
                # =========================================================
                #                     PITCHER DETAILS
                # =========================================================
                
                st.markdown("### ⚾ Pitchers")
                
                pitchers = pitcher_picks_df[pitcher_picks_df["DraftTeam"] == team_selected].copy()
                
                p_cols = ["Pick", "Player", "Position", "ERA", "WHIP", "SO", "W", "SV"]
                df_pitchers = pitchers[p_cols].copy()
                
                # Format ERA/WHIP
                df_pitchers["ERA"] = df_pitchers["ERA"].round(2)
                df_pitchers["WHIP"] = df_pitchers["WHIP"].round(2)
                
                # Round counting stats
                for col in ["SO", "W", "SV"]:
                    df_pitchers[col] = df_pitchers[col].round(0).astype("Int64")
                
                # Totals row
                p_totals = df_pitchers.drop(columns=["Pick", "Player", "Position"], errors="ignore") \
                                       .apply(pd.to_numeric, errors="coerce").sum()
                
                p_total_row = {col: "" for col in df_pitchers.columns}
                for col in p_totals.index:
                    p_total_row[col] = p_totals[col]
                
                # Append totals
                df_pitchers = pd.concat([df_pitchers, pd.DataFrame([p_total_row])], ignore_index=True)
                
                pitcher_placeholder.dataframe(df_pitchers, use_container_width=True)


            except Exception as e:
                st.exception(f"Calculation failed: {e}")
'''
