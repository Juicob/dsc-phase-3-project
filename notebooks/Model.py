# %%
from IPython.display import display
# from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline





# %%
df = pd.read_pickle('../data/main_df.pkl')
df.shape
# %%
# splitting data 
df_train, df_validation = train_test_split(df, test_size=0.10, random_state=42)
# %%
df_train.columns
# %%
# base model resuled in ~92
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
# base model resulted in ~82
features_for_modeling_pct = ['fg_pct_home',
       'ft_pct_home','fg3_pct_home','fg_pct_away', 'ft_pct_away', 'fg3_pct_away']
# %%
# base model resulted in ~82
features_for_modeling_secondary = ['ast_home', 'reb_home','ast_away',
       'reb_away','sum_of_oreb_home',
       'sum_of_dreb_home', 'sum_of_stl_home', 'sum_of_blk_home',
       'sum_of_to_home', 'sum_of_pf_home','sum_of_oreb_away',
       'sum_of_dreb_away', 'sum_of_stl_away', 'sum_of_blk_away',
       'sum_of_to_away', 'sum_of_pf_away']
# %%
X = df_train[features_for_modeling_secondary]
y = df_train['home_team_wins']
X.shape, y.shape
# %%
# splitting training set as a train test
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# %%
classifiers = [KNeighborsClassifier(),
               RandomForestClassifier(), 
            #    SVC(gamma="auto"), 
               AdaBoostClassifier(), 
               GradientBoostingClassifier()]
# Build a for loop to instaniate the classification models and loop through the classifiers
for classifier in classifiers:
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(X_train, y_train)
    print(f'{classifier} Score: {round(pipe.score(X_test, y_test), 4)}')
# %%
gboost_param_grid={'max_depth': [3],
            'max_leaf_nodes': [2, 3],
            'min_samples_leaf': [1, 2],
            'n_estimators': [450],
            'subsample': [0.25,0.5],
            'verbose': [1],
            'random_state':[42]
}
# %%
gridsearch = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=gboost_param_grid, 
                          scoring='precision', cv=5, verbose=1,n_jobs=-1)
gridsearch.fit(X_train, y_train)
display(gridsearch.best_estimator_)
display(gridsearch.best_score_)
gridbest = gridsearch.best_estimator_
 # %%
display(gridbest.fit(X_train, y_train))
display(gridbest.score(X_train, y_train), gridbest.score(X_test, y_test))
# %%
display(plot_confusion_matrix(gridbest, X_train, y_train, cmap='bone'))
display(plot_confusion_matrix(gridbest, X_test, y_test, cmap='bone'))

# %%
X_valid = df_validation[features_for_modeling_secondary]
y_valid = df_validation['home_team_wins']
display(gridbest.score(X_valid, y_valid))
display(plot_confusion_matrix(gridbest, X_valid, y_valid, cmap='bone'))
# %%
px.bar(x=X_train[features_for_modeling_secondary].columns,
       y=sorted(pipe[0].feature_importances_, reverse=True),
       title='Most Important Features',
       labels={'x':'Features',
               'y':'Importance'})


# %%
# %%
# y_pred = gridbest.predict(X_test)
# %%
y_scores = gridbest.decision_function(X_train)
# y_preds = gridbest.predict(X_train)
fpr, tpr, thresh = roc_curve(y_train, y_scores)

# Calculate the ROC (Reciever Operating Characteristic) AUC (Area Under the Curve)
rocauc = auc(fpr, tpr)
print('Train ROC AUC Score: ', rocauc)
# %%
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()

# %%