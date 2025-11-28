import streamlit as st
import pandas as pd
import io
from typing import List
from datetime import datetime
import requests

def getAggregateHitterProj(list):
  df = pd.DataFrame()
  for i in list:
    data = requests.get("https://www.fangraphs.com/api/projections?pos=all&stats=bat&type="+i).json()
    #data = requests.get("https://www.fangraphs.com/api/projections?type=steamerr&stats=bat&pos=all&team=0&players=0&lg=all&z=1744109615&pageitems=30&statgroup=dashboard&fantasypreset=dashboard")
    projDF = pd.DataFrame(data)
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
    data = requests.get("https://www.fangraphs.com/api/projections?pos=all&stats=pit&type="+i).json()
    projDF = pd.DataFrame(data)
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
      if pd.isna(name) or ', ' in name:
          return name
      parts = name.split()
      if len(parts) > 1:
          return f"{parts[-1]}, {' '.join(parts[:-1])}"
      return name

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
    st.subheader("Projected Standings")
    # Placeholder for results area
    results_placeholder = st.empty()
    download_placeholder = st.empty()
    last_run_info = st.empty()

# -----------------------
# Perform calculation when button pressed
# -----------------------
if calculate_button:
    if uploaded is None or picks_df is None:
        st.warning("Please upload a valid .tsv draft picks file before calculating.")
    elif len(selected_systems) == 0:
        st.warning("Please select at least one projection system.")
    else:
        # Run the calculation
        with st.spinner("Calculating projected standings..."):
            try:
                standings_df = calculate_standings(picks_df, selected_systems, starters_only)

                if not isinstance(standings_df, pd.DataFrame):
                    raise ValueError("calculate_standings did not return a pandas DataFrame.")

                # Display timestamp
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                last_run_info.markdown(f"**Last calculated:** {now} — using: {', '.join(selected_systems)}")

                # -----------------------
                # EXTRA UI: Highlight Controls
                # -----------------------
                st.markdown("### 🔍 Highlight Options")

                # Team dropdown (populate after calculation)
                team_list = standings_df["Team"].dropna().unique().tolist()
                highlight_team = st.selectbox(
                    "Highlight a team (optional):",
                    options=["None"] + team_list,
                    index=0
                )

                # Category highlight dropdown
                numeric_cols = [
                    col for col in standings_df.columns
                    if standings_df[col].dtype != "object" and col not in ("Rank")
                ]

                highlight_stat = st.selectbox(
                    "Highlight top team in a category (optional):",
                    options=["None"] + numeric_cols,
                    index=0
                )

                # Color intensity
                intensity = st.slider(
                    "Highlight intensity:",
                    min_value=0.1, max_value=1.0, value=0.3, step=0.05
                )

                # -----------------------
                # APPLY HIGHLIGHTING
                # -----------------------
                styled_df = standings_df.copy()

                def highlight_rows(row):
                    base = ""
                    # Team highlight
                    if highlight_team != "None" and row["Team"] == highlight_team:
                        base = f"background-color: rgba(255, 255, 0, {intensity});"

                    # Category highlight
                    if highlight_stat != "None":
                        max_val = standings_df[highlight_stat].max()
                        if row[highlight_stat] == max_val:
                            base = f"background-color: rgba(0, 255, 0, {intensity});"

                    return [base] * len(row)

                styled_output = standings_df.style.apply(highlight_rows, axis=1)

                # -----------------------
                # DISPLAY THE TABLE
                # -----------------------
                results_placeholder.dataframe(styled_output, use_container_width=True)

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

            except Exception as e:
                st.exception(f"Calculation failed: {e}")


# -----------------------
# Footer / helpful tips
# -----------------------
st.markdown("---")
st.caption(
    "This interface implements: 1) TSV draft file upload, 2) six projection-system checkboxes, "
    "3) roster option (entire roster vs starters only), 4) a Calculate button, and 5) "
    "a sortable standings table with CSV download. Replace the placeholder `projection_engine.calculate_standings` "
    "import with your existing function to plug in your projection engine."
)
