import pandas as pd
from github import Github
import os
import requests

def upload_df_to_github(df, repo_name, file_path, commit_message, github_token):
    """Uploads a Pandas DataFrame to GitHub as a CSV."""
    csv_data = df.to_csv(index=False)

    g = Github(github_token)
    repo = g.get_repo(repo_name)

    try:
        contents = repo.get_contents(file_path)
        repo.update_file(
            path=file_path,
            message=commit_message,
            content=csv_data,
            sha=contents.sha
        )
        print(f"Updated {file_path}")
    except Exception:
        repo.create_file(
            path=file_path,
            message=commit_message,
            content=csv_data
        )
        print(f"Created {file_path}")


# 1. Load GitHub token
github_token = os.getenv("TOKEN_PRIVATE")  # stored securely in GitHub Actions secrets
repo_name = "ryanb519/standings-predictor"
    
# 2. Get Projections and upload into GitHub #NOTE added 'r' to projections for in-season ROS. Remove 'r' for preseason proj.
# For example 'steamerr' is 'steamer', 'ratcdc' should be 'atc', etc.
projList = ['steamerr','rfangraphsdc','rthebat','rthebatx','roopsydc','ratcdc']

#PROJECTIONS
for i in projList:
  data = requests.get("https://www.fangraphs.com/api/projections?pos=all&stats=bat&type="+i).json()
  df = pd.DataFrame(data)
  df.to_csv(i+'_hitters.csv', index=False)
  upload_df_to_github(df, repo_name, "data/"+i+"_hitters.csv", "Daily update: hitters", github_token)

  data = requests.get("https://www.fangraphs.com/api/projections?pos=all&stats=pit&type="+i).json()
  df = pd.DataFrame(data)
  df.to_csv(i+'_pitchers.csv', index=False)
  upload_df_to_github(df, repo_name, "data/"+i+"_pitchers.csv", "Daily update: pitchers", github_token)

#SCHEDULE
data = requests.get("https://www.fangraphs.com/api/roster-resource/schedule-grid/data").json()
df = pd.DataFrame(data)
upload_df_to_github(df, repo_name, "data/fangraphs/schedule.json", "Daily update: hitters", github_token)

#PROBABLES
data = requests.get("https://www.fangraphs.com/api/roster-resource/probables-grid/data").json()
df = pd.DataFrame(data)
upload_df_to_github(df, repo_name, "data/fangraphs/probables.json", "Daily update: hitters", github_token)

#2026 HITTER STATS
data = requests.get("https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=bat&lg=all&qual=5&season=2026&season1=2026&startdate=2026-03-01&enddate=2026-11-01&month=0&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR").json()
data = data['data']
data = pd.DataFrame(data)
data['IDfg'] = data['playerid']
upload_df_to_github(df, repo_name, "data/fangraphs/fgb26.json", "Daily update: hitters", github_token)

#2026 PITCHER STATS
data = requests.get("https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&qual=5&season=2026&season1=2026&startdate=2026-03-01&enddate=2026-11-01&month=0&hand=&team=0&pageitems=2000000000&pagenum=1&ind=0&rost=0&players=&type=8&postseason=&sortdir=default&sortstat=WAR").json()
data = data['data']
data = pd.DataFrame(data)
data['IDfg'] = data['playerid']
upload_df_to_github(df, repo_name, "data/fangraphs/fgp26.json", "Daily update: pitchers", github_token)

#TEAM LEVEL
ytdData = requests.get('https://www.fangraphs.com/api/leaders/major-league/data?pos=all&stats=bat&lg=all&qual=y&season=2026&season1=2026&ind=0&team=0%2Cts&type=8&month=0').json()
vLData = requests.get("https://www.fangraphs.com/api/leaders/major-league/data?pos=all&stats=bat&lg=all&qual=y&season=2026&season1=2026&ind=0&team=0%2Cts&type=8&month=13").json()
vRData = requests.get("https://www.fangraphs.com/api/leaders/major-league/data?pos=all&stats=bat&lg=all&qual=y&season=2026&season1=2026&ind=0&team=0%2Cts&type=8&month=14").json()
l30Data = requests.get("https://www.fangraphs.com/api/leaders/major-league/data?pos=all&stats=bat&lg=all&qual=y&season=2026&season1=2026&ind=0&team=0%2Cts&type=8&month=3").json()

ytdData = ytdData['data']
vLdata = vLData['data']
vRdata = vRData['data']
l30Data = l30Data['data']

ytd = pd.DataFrame(ytdData)
ytd['SBA'] = (ytd['SB'] + ytd['CS']) / (ytd['1B'] + ytd['BB'] + ytd['IBB'] + ytd['HBP'])
vL = pd.DataFrame(vLdata)
vR = pd.DataFrame(vRdata)
l30 = pd.DataFrame(l30Data)

upload_df_to_github(ytd, repo_name, "data/fangraphs/month0.json", "Daily update: pitchers", github_token)
upload_df_to_github(vL, repo_name, "data/fangraphs/month13.json", "Daily update: pitchers", github_token)
upload_df_to_github(vR, repo_name, "data/fangraphs/month13.json", "Daily update: pitchers", github_token)
upload_df_to_github(l30, repo_name, "data/fangraphs/month3.json", "Daily update: pitchers", github_token)
