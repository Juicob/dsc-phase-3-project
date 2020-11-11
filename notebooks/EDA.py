# %%
from IPython.display import display
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib as plt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import pickle


# %%
df = pd.read_pickle('../data/main_df.pkl')
# %% [markdown]
#~~best overall team by wins~~
#   - ~~home~~
#   - ~~away~~
# !
# ~~most losses~~
# - ~~home~~
# - ~~away~~
# !
# wins by year plots by year all teams
#~~best overall by conference?~~
#defense wins games - pressure busts pipes -  analyze impact
#pressure makes diamonds - look into highest scoring individuals
#   - overall
#   - by conference
# time series of scoring?
# look into relationship between home team win pct at arenas vs centers vs other for funsies
# 
# look into raptor rating
# 




# %%
# plt.figure(figsize=(20,15))
# df.hist(figsize=(20,15));
# %%
df['away_team_wins'] = df['home_team_wins'].apply(lambda x: 1.0 if x == 0 else 0)
df['home_team_losses'] = df['away_team_wins'].apply(lambda x: 1.0 if x == 1 else 0)
df['away_team_losses'] = df['home_team_wins'].apply(lambda x: 1.0 if x == 1 else 0)
# %%
# %%
home_history_totals = df.groupby(by='home_team_id').sum()[['home_team_wins','home_team_losses','sum_of_fgm_home',
       'sum_of_fga_home', 'sum_of_fg3m_home', 'sum_of_fg3a_home',
       'sum_of_ftm_home', 'sum_of_fta_home', 'sum_of_oreb_home',
       'sum_of_dreb_home', 'sum_of_stl_home', 'sum_of_blk_home',
       'sum_of_to_home', 'sum_of_pf_home']]

away_history_totals = df.copy().groupby(by='visitor_team_id').sum()[['away_team_wins','away_team_losses','sum_of_fgm_away',
       'sum_of_fga_away', 'sum_of_fg3m_away', 'sum_of_fg3a_away',
       'sum_of_ftm_away', 'sum_of_fta_away', 'sum_of_oreb_away',
       'sum_of_dreb_away', 'sum_of_stl_away', 'sum_of_blk_away',
       'sum_of_to_away', 'sum_of_pf_away']]

home_history_totals.reset_index(inplace=True)
away_history_totals.reset_index(inplace=True)


# %%
home_history_totals = pd.merge(home_history_totals, df[['home_team_id','home_nickname', 'home_yearfounded','home_arena', 'home_city', 'home_conference']], on=['home_team_id'])


away_history_totals = away_history_totals.merge(df[['visitor_team_id','away_nickname', 'away_yearfounded','away_arena', 'away_city', 'away_conference']],on=['visitor_team_id'])


# %%
home_history_totals.drop_duplicates(inplace=True)
away_history_totals.drop_duplicates(inplace=True)
home_history_totals.reset_index(drop=True, inplace=True)
away_history_totals.reset_index(drop=True, inplace=True)
# %%
display(home_history_totals.head())
display(away_history_totals.head())

# %%
home_history_totals.sort_values(by='home_team_wins', ascending=False, inplace=True)
away_history_totals.sort_values(by='away_team_wins', ascending=False, inplace=True)
# %%
# plt.figure(figsize=(20,15))
display(sns.barplot(x='home_city', y='home_team_wins', data=home_history_totals, palette='rocket_r'))


# %%
display(sns.barplot(x='away_city', y='away_team_wins', data=away_history_totals, palette='rocket_r'))

# %%
display(home_history_totals.columns)
display(away_history_totals.columns)
# %%
fig = go.Figure(data=[
    go.Bar(name='Home Wins', x=home_history_totals.home_city, y=home_history_totals.home_team_wins),
    go.Bar(name='Away Wins', x=away_history_totals.away_city, y=away_history_totals.away_team_wins),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()
# %% [markdown]
### consider doing both grouped and stacked charts
# %%

# %%
fig = go.Figure(data=[
    go.Bar(name='Home Wins', x=home_history_totals.home_nickname, y=home_history_totals.home_team_wins, marker_color='burlywood'),
    go.Bar(name='Away Wins', x=away_history_totals.away_nickname, y=away_history_totals.away_team_wins, marker_color='#b36360'),
])
fig.update_layout(title=dict(text='Wins Per Team by Travel',
                             y=0.9,x=0.5,
                             xanchor='auto', 
                             yanchor='middle'),
                  barmode='group',
                  plot_bgcolor='lightslategrey',
                  xaxis_title="Team Name",
                  yaxis_title="Total Wins",
                  bargap=0.25,
                  bargroupgap=0.15,
                  height=500,
                  width=1000)
fig.show()
# %%
home_history_totals.sort_values(by='home_team_losses', ascending=False, inplace=True)
away_history_totals.sort_values(by='away_team_losses', ascending=False, inplace=True)
fig = go.Figure(data=[
    go.Bar(name='Home Losses', x=home_history_totals.home_nickname, y=home_history_totals.home_team_losses, marker_color='burlywood'),
    go.Bar(name='Away Losses', x=away_history_totals.away_nickname, y=away_history_totals.away_team_losses, marker_color='#b36360'),
])
fig.update_layout(title=dict(text='Losses Per Team by Travel',
                             y=0.9,x=0.5,
                             xanchor='auto', 
                             yanchor='middle'),
                  barmode='group',
                  plot_bgcolor='lightslategrey',
                  xaxis_title="Team Name",
                  yaxis_title="Total Wins",
                  bargap=0.25,
                  bargroupgap=0.15,
                  height=500,
                  width=1000)
fig.show()
# %%
season_df = df.copy().groupby(by=['season','home_team_id']).sum()[['home_team_wins','home_team_losses','away_team_wins','away_team_losses']]
season_df.reset_index(inplace=True)
season_df['total_wins'] = season_df.home_team_wins + season_df.away

# %%
df.head()
# %%
fig = make_subplots(rows=5, cols=1)

for i in range(1,5):
    fig.add_trace(
        go.Bar(name='Home Losses', x=home_history_totals.home_nickname, y=home_history_totals.home_team_losses, marker_color='burlywood'),
        row=i, col=1)

fig.update_layout(height=800, width=800)

# %%
fig = px.bar(home_history_totals, 
             x='home_nickname', 
             y='home_team_wins',
             hover_data=['home_team_wins',], 
             color='home_conference',
             height=400)

fig.update_layout(plot_bgcolor='lightslategrey',
                  paper_bgcolor='ivory', )

fig.update_traces(marker=dict(line=dict(width=2, 
                                        color='seashell')),
                  selector=dict(type='bar'))
fig.show()
# %%
fig = go.Figure(data=[go.Bar(
    x=home_history_totals.home_city,
    y=home_history_totals.home_team_wins,
    marker_color='burlywood' # marker color can be a single color value or an iterable
)])
fig.update_layout(title_text='Home Team Wins')
# %%
df.columns
# %%
