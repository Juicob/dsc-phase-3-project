# %%
import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, fbeta_score, f1_score, precision_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from IPython.display import Image

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import plotly
import plotly.graph_objects as go

from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# %%
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='0.4g', cmap='bone')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    pass



def evaluate_test_data(logreg, df_test, feature_for_modeling, target='y'):
    # df_test = transform_dataframe(df_test)
    X = df_test[feature_for_modeling]
    y = df_test[target]
    score = logreg.score(X, y)
    print(f"Score of model = {score}")
    y_pred = logreg.predict(X)
    plot_confusion(y, y_pred)
    pass
# %%
df = pd.read_pickle('../data/main_df.pkl')
df.shape
# %%
# splitting data 
df_train, df_validation = train_test_split(df, test_size=0.10)
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
features_for_modeling_pct = ['fg_pct_home',
       'ft_pct_home', 'fg3_pct_home','fg_pct_away', 'ft_pct_away', 'fg3_pct_away']
# %%
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
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.20)
# %%
classifiers = [KNeighborsClassifier(),
               RandomForestClassifier(), 
               SVC(gamma="auto"), 
               AdaBoostClassifier(), 
               GradientBoostingClassifier()]
# Build a for loop to instaniate the classification models and loop through the classifiers
for classifier in classifiers:
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(X_train, y_train)
    print(f'{classifier} Score: {round(pipe.score(X_test, y_test), 4)}')
# %%

grid = GradientBoostingClassifier(n_estimators=150)
display(grid.fit(X_train, y_train))
display(grid.score(X_train, y_train), grid.score(X_test, y_test))
# %%
             param_grid={'classifier__max_depth': [3],
                         'classifier__max_leaf_nodes': [2, 3],
                         'classifier__min_samples_leaf': [1, 2],
                         'classifier__n_estimators': [450],
                         'classifier__verbose': [1]},
# %%
gridsearch = GridSearchCV(estimator=pipe, param_grid=param_grid, 
                          scoring='accuracy', cv=5, verbose=1,n_jobs=-1)
# %%
gridsearch.fit(X_train, y_train)
# %%
gridsearch.best_estimator_

# %%
gridbest = gridsearch.best_estimator_
gridsearch.best_score_
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
testgrad = GradientBoostingClassifier()
testgrad.fit(X_train, y_train)
testgrad.score(X_train, y_train)

first_tree = testgrad.estimators_[0]
#  plot_tree(first_tree, feature_names=X_train.columns, class_names=df['home_team_wins'].unique)

sub_tree_42 = testgrad.estimators_[42, 0]

# Visualization. Install graphviz in your system
dot_data = export_graphviz(
    sub_tree_42,
    out_file=None, filled=True, rounded=True,
    special_characters=True,
    proportion=False, impurity=False, # enable them if you want
)
graph = graph_from_dot_data(dot_data)
Image(graph.create_png())
# %%

# %%
y_scores = gridbest.decision_function(X_train)
# %%
y_pred = gridbest.predict(X_test)
# %%
fpr, tpr, thresh = roc_curve(y_train, y_scores)

# Calculate the ROC (Reciever Operating Characteristic) AUC (Area Under the Curve)
rocauc = auc(fpr, tpr)

print('Train FPR: ', fpr)
print('Train TPR: ', tpr)
print('Train ROC AUC Score: ', rocauc)
# %%

# Calculate the False Positive Rate (FPR), True Posititve
# Rate (TPR), and Threshold for the testing data and testing predictions
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

# print('Training Root Mean Square Error', np.sqrt(mean_squared_error(y_resampled_train, y_scores)))
print('Testing Root Mean Square Error', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Test FPR: ', false_positive_rate)
print('Test TPR: ', true_positive_rate)
print('Test ROC AUC Score: ', roc_auc)
# %%

# %%

# %%
