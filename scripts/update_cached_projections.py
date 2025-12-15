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
    
# 2. Get Projections and upload into GitHub
projList = ['steamer','fangraphsdc','thebat','thebatx','oopsy','atc']

for i in projList:
  data = requests.get("https://www.fangraphs.com/api/projections?pos=all&stats=bat&type="+i).json()
  df = pd.DataFrame(data)
  df.to_csv(i+'_hitters.csv', index=False)
  upload_df_to_github(df, repo_name, "data/"+i+"_hitters.csv", "Daily update: hitters", github_token)

  data = requests.get("https://www.fangraphs.com/api/projections?pos=all&stats=pit&type="+i).json()
  df = pd.DataFrame(data)
  df.to_csv(i+'_pitchers.csv', index=False)
  upload_df_to_github(df, repo_name, "data/"+i+"_pitchers.csv", "Daily update: pitchers", github_token)


