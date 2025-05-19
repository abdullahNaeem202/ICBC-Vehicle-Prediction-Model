import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.parse.earleychart import EARLEY_FEATURE_STRATEGY
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy import stats, np_minversion
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate

# Read data
raw_data = pd.read_csv('cleaned_vehicle_data.csv')

import pandas as pd

# Step 1: Filter by target values
yes_df = raw_data[raw_data['Electric_Vehicle_Indicator'] == 'Yes']
no_df = raw_data[raw_data['Electric_Vehicle_Indicator'] == 'No']

# Step 2: Randomly sample from the majority class to match the minority count
no_sampled_df = no_df.sample(n=len(yes_df), random_state=100)

# Step 3: Concatenate to form a balanced dataset
data = pd.concat([yes_df, no_sampled_df]).sample(frac=1, random_state=100).reset_index(drop=True)

# function for contingency table generation
def print_cross_tab(s):
    out = pd.crosstab(s, data['Electric_Vehicle_Indicator'])
    print(f"\nContingency Table: {s.name}")
    print(tabulate(out, headers='keys', tablefmt='fancy_grid'))
    return out



data['Make'] = data['Make'].astype('category')
data['Model.Year'] = data['Model.Year'].astype('category')
data['Body.Style'] = data['Body.Style'].astype('category')
data['Anti.Theft.Device.Indicator'] = data['Anti.Theft.Device.Indicator'].astype('category')
data['Electric_Vehicle_Indicator'] = data['Electric_Vehicle_Indicator'].astype('category')

chi_1 = print_cross_tab(data['Make'])
chi_2 = print_cross_tab(data['Model.Year'])
chi_3 = print_cross_tab(data['Body.Style'])
chi_4 = print_cross_tab(data['Anti.Theft.Device.Indicator'])

data = data[["Make", "Model.Year", "Body.Style", "Anti.Theft.Device.Indicator", "Electric_Vehicle_Indicator"]]


# Chi-squared test
chi_test1 = stats.chi2_contingency(chi_1)
print('Chi - squared p value for variable "Make": ',chi_test1.pvalue)

chi_test2 = stats.chi2_contingency(chi_2)
print('Chi - squared p value for variable "Model Year": ',chi_test2.pvalue)

chi_test3 = stats.chi2_contingency(chi_3)
print('Chi - squared p value for variable "Body Style": ',chi_test3.pvalue)

chi_test4 = stats.chi2_contingency(chi_4)
print('Chi - squared p value for variable "Anti-Theft": ',chi_test4.pvalue)

def cat_to_int(v):
    num_levels = v.nunique()
    encoding_dictionary = {}
    unique_values = v.unique()

    for i, value in enumerate(unique_values):
        encoding_dictionary[value] = i

    return encoding_dictionary

def numeric_features_category(dt):
    copy_data = dt.copy()
    copy_data['Make'] = copy_data['Make'].map(cat_to_int(data['Make']))
    copy_data['Body.Style'] = copy_data['Body.Style'].map(cat_to_int(data['Body.Style']))
    copy_data['Model.Year'] = copy_data['Model.Year'].map(cat_to_int(data['Model.Year']))
    copy_data['Anti.Theft.Device.Indicator'] = copy_data['Anti.Theft.Device.Indicator'].map(cat_to_int(data['Anti.Theft.Device.Indicator']))
    return copy_data

def plot_roc_curves(models_probs, true_y):
    plt.figure(figsize=(8, 6))
    auc_scores = []

    for model_name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(true_y, y_prob, pos_label='Yes')
        auc = roc_auc_score(true_y, y_prob)
        auc_scores.append(auc)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('ROC_Comparison.png')

    return auc_scores
# Balancing the dataset (Oversampling)

new_data = numeric_features_category(data)
print(new_data)

# Defining our X and Y (Features and Target)

X = new_data[['Make', 'Model.Year', 'Body.Style', 'Anti.Theft.Device.Indicator']]
Y = new_data['Electric_Vehicle_Indicator']

# Splitting data into training and validating sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y)
print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)

# Decision Trees model

model_dtree = DecisionTreeClassifier(max_depth= 8, min_samples_leaf= 8)
model_dtree.fit(X_train, Y_train)

Y_predicted = model_dtree.predict(X_valid)


# Random Forest model

model_rf = RandomForestClassifier(n_estimators= 75
        , min_samples_leaf=10)

model_rf.fit(X_train, Y_train)
Y_predicted_rf = model_rf.predict(X_valid)

## Gradient Boosting classifier

model_gb = GradientBoostingClassifier(n_estimators= 325, max_depth = 5, min_samples_leaf= 0.2)
model_gb.fit(X_train, Y_train)

##  NN classifier

model_nn = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500)
model_nn.fit(X_train, Y_train)



# Confusion Matrix for the Best model based on Accuracy Scores
confusion_matrix = metrics.confusion_matrix(Y_valid, Y_predicted_rf)
sns.set(font_scale= 1.5)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix, annot = True, fmt = 'd', xticklabels = model_rf.classes_, yticklabels = model_rf.classes_)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for RF')
plt.show()
plt.savefig('confusion_matrix.png')

# Extract values
tn, fp, fn, tp = confusion_matrix.ravel()

# Calculate Hit Rate (Recall)
hit_rate = tp / (tp + fn)
print("Hit Rate for Random Forest: ",hit_rate)

# Create a ROC curve to compare model performances

Y_predicted_proba_dt = model_dtree.predict_proba(X_valid)[:, 1]
Y_predicted_proba_rf = model_rf.predict_proba(X_valid)[:, 1]
Y_predicted_proba_gb = model_gb.predict_proba(X_valid)[:, 1]
Y_predicted_proba_nn = model_nn.predict_proba(X_valid)[:, 1]

# Dictionary to store probabilities of different models
models_probs = {
    'Decision Tree': Y_predicted_proba_dt,
    'Random Forest': Y_predicted_proba_rf,
    'Gradient Boosting': Y_predicted_proba_gb,
    'Neural Network': Y_predicted_proba_nn
}


auc_scores = plot_roc_curves(models_probs, Y_valid)


# Create dataframe to store Model accuracy scores

scores_df = pd.DataFrame(columns=['Model Name', 'Accuracy Score','AUC Score'])
scores_df['Model Name'] = ['Decision Tree', 'Random Forest', 'Gradient Boosting','Neural Network']
scores_df['Accuracy Score'] = [model_dtree.score(X_valid, Y_valid), model_rf.score(X_valid, Y_valid), model_gb.score(X_valid, Y_valid),
                               model_nn.score(X_valid, Y_valid)]
scores_df['AUC Score'] = auc_scores

print("Model Performance Comparison:\n")
print(tabulate(scores_df, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".2f"))