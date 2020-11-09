# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib as plt
import seaborn as sns


# %%
df = pd.read_csv(r'../data/games.csv')
details_df = pd.read_csv(r'../data/games_details.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
details_df.columns = details_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df.dropna(inplace=True)
details_df.drop(columns='comment', inplace=True)
details_df.dropna(inplace=True)

# %%
df.head()


# %%
df.info()

# %% [markdown]
# - about 100 missing values for games in 2003, consider dropping rows

# %%
details_df.head()


# %%
details_df.info()

# %% [markdown]
# - almost 100k nan values due to listed players that did not play and therefore have no stats. planning to drop
# %% [markdown]
# ## approach, for now anyway
# %% [markdown]
# 
# consider grouping game details metrics for [these values](https://i.imgur.com/WqiiDfg.png) and also adding to main table
# 
# - group by game id (and team_id?????) to create columns for both maybe thisll work
#     - [reference](https://jamesrledoux.com/code/group-by-aggregate-pandas)
#     - details_df.groupby(['game_id', 'team_id'])                                            #.agg({'Age': ['mean', 'min', 'max']}) - multi aggregation optino - I think this might be tedious for my case with so many features. Instead I'll try to do one df for sum and another for mean and grab what I need from each
#         - sum_details_df = details_df.groupby(['game_id', 'team_id']).sum()
#         - mean_details_df = details_df.groupby(['game_id', 'team_id']).mean()
#     - may need to do something about indexes, either reset or move shit around so I can make the join
#     - home team
#         - sum of fga - field goal attempts to understand context as denominator
#         - mean fielg goal pct
#         - sum of free throw attempts for context
#         - mean free throw pct
#         - sum of three point shot attempts - context
#         - sum of off rebound
#         - sum of def rebound
#         - sum of steals
#         - sum of blocks
#         - sum of turn overs
#         - sum of personal fouls
#         - max points??? to show the highest single player point contributer?
#         - max assist??? to show the highest single player assist contributer?
#     - away team
#         - sum of fga - field goal attempts to understand context as denominator
#         - mean fielg goal pct
#         - sum of free throw attempts for context
#         - mean free throw pct
#         - sum of three point shot attempts - context
#         - sum of off rebound
#         - sum of def rebound
#         - sum of steals
#         - sum of blocks
#         - sum of turn overs
#         - sum of personal fouls
#         - max points??? to show the highest single player point contributer?
#         - max assist??? to show the highest single player assist contributer?
# - join to main with something like 
#     - pd.merge(main_df, details_df, on=['game_id','home_team_id'], suffixes='_home') 
#     - pd.merge(main_df, details_df, on=['game_id','away_team_id'], suffixes='_away') 
# 
#         
# 
# if time allows I'd love to learn more about this and incorporate it in some way
# 
# https://fivethirtyeight.com/features/how-our-raptor-metric-works/
# 
# consider grouping modern and historical estimates of raptor metrics for average by team and joining home and away metrics to main table
# 
# - "RAPTOR is a plus-minus statistic that measures the number of points a player contributes to his team’s offense and defense per 100 possessions, relative to a league-average player"
# - "RAPTOR does not account for coaching, systems or synergies between teammates"
# - "RAPTOR ratings for players with at least 1,000 minutes played in a season"


# %%
df.columns
# %%
sum_details_df = details_df.groupby(['game_id', 'team_id']).sum()

avg_details_df = details_df.groupby(['game_id', 'team_id']).mean()

# %%
columns_from_sum_to_drop = ['player_id','fg_pct','fg3_pct','ft_pct','plus_minus','pts','reb','ast']
columns_from_avg_to_keep = ['fg_pct','fg3_pct','ft_pct']
sum_details_df = sum_details_df.drop(columns=columns_from_sum_to_drop)
# avg_details_df = avg_details_df[columns_from_avg_to_keep]

home_sum_details_df = sum_details_df
away_sum_details_df = sum_details_df.copy()
# home_avg_details_df = avg_details_df
# away_avg_details_df = avg_details_df.copy()
# %%
# home_sum_details_df == away_sum_details_df
# %%
home_sum_details_df.columns = [f'sum_of_{col}_home' for col in home_sum_details_df.columns]
away_sum_details_df.columns = [f'sum_of_{col}_away' for col in away_sum_details_df.columns]
# home_avg_details_df.columns = [f'avg_of_{col}_home' for col in home_avg_details_df.columns]
# away_avg_details_df.columns = [f'avg_of_{col}_away' for col in away_avg_details_df.columns]
# %%
# sum_details_df.index.names[0]
home_sum_details_df = home_sum_details_df.reset_index()
away_sum_details_df = away_sum_details_df.reset_index()
# home_avg_details_df = home_avg_details_df.reset_index()
# away_avg_details_df = away_avg_details_df.reset_index()

# %%
# home_details_df.rename(columns={"sum_of_game_id_home": "game_id", "sum_of_team_id_home": "team_id"}, inplace=True)
# %%
away_sum_details_df
# %%
# df = df.merge(home_sum_details_df, left_on=['game_id','home_team_id'], right_on=['game_id', 'team_id'])
df = df.merge(home_sum_details_df, left_on=['game_id','home_team_id'], right_on=['game_id', 'team_id'])
df = df.merge(away_sum_details_df, left_on=['game_id','visitor_team_id'], right_on=['game_id', 'team_id'])
# df = df.merge(home_avg_details_df, left_on=['game_id','home_team_id'], right_on=['game_id', 'team_id'])
# df = df.merge(away_avg_details_df, left_on=['game_id','visitor_team_id'], right_on=['game_id', 'team_id'])


# %%
df.columns
# %%
df.head()


# %%
df.head()
# %%
df.to_csv('first merge test.csv')