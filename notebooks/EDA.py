# %%
import pandas as pd
import matplotlib as plt
import seaborn as sns
import pickle


# %%
df = pd.read_pickle('../data/main_df.pkl')
# %% [markdown]
#best overall team by wins
#   - home
#   - away
#best overall by conference?
#defense wins games - pressure busts pipes -  analyze impact
#pressure makes diamonds - look into highest scoring individuals
#   - overall
#   - by conference
# look into relationship between home team win pct at arenas vs centers vs other for funsies
# 
# look into raptor rating
# 




# %%
# plt.figure(figsize=(20,15))
# df.hist(figsize=(20,15));
# %%
df.columns
# %%
home_history_totals = df.groupby(by='home_team_id').sum()[['home_team_wins','sum_of_fgm_home',
       'sum_of_fga_home', 'sum_of_fg3m_home', 'sum_of_fg3a_home',
       'sum_of_ftm_home', 'sum_of_fta_home', 'sum_of_oreb_home',
       'sum_of_dreb_home', 'sum_of_stl_home', 'sum_of_blk_home',
       'sum_of_to_home', 'sum_of_pf_home']]

home_history_totals.reset_index()


# %%
home_history_totals = pd.merge(home_history_totals, df[['home_team_id','home_nickname', 'home_yearfounded','home_arena', 'home_city']], on=['home_team_id'])
# %%
home_history_totals.drop_duplicates(inplace=True)
home_history_totals.reset_index(inplace=True)
# %%
home_history_totals.sort_values(by='home_team_wins', ascending=False, inplace=True)
# %%
# plt.figure(figsize=(20,15))
sns.barplot(x='home_city', y='home_team_wins', data=home_history_totals, palette='rocket_r')
# %%
