import pandas as pd
from src.data.team_names import canonicalize

r = pd.read_csv('data/processed/team_ratings.csv', parse_dates=['rating_date'])
r['team_name'] = r['team_name'].map(canonicalize)
latest = r.sort_values('rating_date').groupby('team_name')['elo_rating'].last()

u = pd.read_csv('data/unified.csv')
u['home_team'] = u['home_team'].map(canonicalize)
u['away_team'] = u['away_team'].map(canonicalize)

teams = ['United States','Paraguay','Mexico','Canada','Brazil','Spain','Japan','Argentina','Uruguay','Morocco']
print("=== ELO + PARTIDOS ===")
for t in teams:
    elo = latest.get(t, float('nan'))
    n = int(((u.home_team == t) | (u.away_team == t)).sum())
    print(f"{t:18} Elo={elo:7.0f}   {n} partidos")
    