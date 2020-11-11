# %%
import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, fbeta_score, f1_score, precision_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import plotly
import plotly.graph_objects as go
# %%
def evaluate_test_data(classifier, df_test, features_for_modeling, target):
    # df_test = transform_dataframe(df_test)
    X = df_test[features_for_modeling]
    y = df_test[target]
    score = logreg.score(X, y)
    print(f"Score of model = {score}")
    y_pred = classifier.predict(X)
    plot_confusion_matrix(classifier, y, y_pred, cmap='bone')
# %%
df = pd.read_pickle('../data/main_df.pkl')
df.shape
# %%
# splitting data 
df_train, df_test = train_test_split(df, test_size=0.10)
# %%
df_train.columns
# %%
features_for_modeling = ['fg_pct_home',
       'ft_pct_home', 'fg3_pct_home', 'ast_home', 'reb_home', 'fg_pct_away', 'ft_pct_away', 'fg3_pct_away', 'ast_away',
       'reb_away','sum_of_fga_home', 'sum_of_fg3m_home', 'sum_of_fg3a_home',
       'sum_of_ftm_home', 'sum_of_fta_home', 'sum_of_oreb_home',
       'sum_of_dreb_home', 'sum_of_stl_home', 'sum_of_blk_home',
       'sum_of_to_home', 'sum_of_pf_home','sum_of_fgm_away',
       'sum_of_fga_away', 'sum_of_fg3m_away', 'sum_of_fg3a_away',
       'sum_of_ftm_away', 'sum_of_fta_away', 'sum_of_oreb_away',
       'sum_of_dreb_away', 'sum_of_stl_away', 'sum_of_blk_away',
       'sum_of_to_away', 'sum_of_pf_away']
# %%
X = df_train[features_for_modeling]
y = df_train['home_team_wins']
X.shape, y.shape
# %%
# splitting training set as a train test
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.50)
# %%
logreg = LogisticRegression()
# %%
logreg.fit(X_train, y_train)
# %%
logreg.score(X_train, y_train), logreg.score(X_test, y_test)
# %%
y_test_pred = logreg.predict(X_test)
# %%
plot_confusion_matrix(logreg, X_test, y_test, cmap='bone', )
# plt.show()
# %%
evaluate_test_data(logreg, df_test, features_for_modeling, 'home_team_wins')
# %%

# %%

# %%

# %%

# %%

# %%
