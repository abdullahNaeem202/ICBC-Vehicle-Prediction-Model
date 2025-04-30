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

# Read data
raw_data = pd.read_csv('cleaned_vehicle_data.csv')

import pandas as pd

# Assuming your DataFrame is named df
# Step 1: Filter by target values
yes_df = raw_data[raw_data['Electric_Vehicle_Indicator'] == 'Yes']
no_df = raw_data[raw_data['Electric_Vehicle_Indicator'] == 'No']

# Step 2: Randomly sample from the majority class to match the minority count
no_sampled_df = no_df.sample(n=len(yes_df), random_state=100)

# Step 3: Concatenate to form a balanced dataset
data = pd.concat([yes_df, no_sampled_df]).sample(frac=1, random_state=100).reset_index(drop=True)

# Optional: check the distribution
print(data['Electric_Vehicle_Indicator'].value_counts())
# function for contingency table generation

def print_cross_tab(s):
    out = pd.crosstab(s, data['Electric_Vehicle_Indicator'])
    return out


data['Make'] = data['Make'].astype('category')
data['Model.Year'] = data['Model.Year'].astype('category')
data['Body.Style'] = data['Body.Style'].astype('category')
data['Anti.Theft.Device.Indicator'] = data['Anti.Theft.Device.Indicator'].astype('category')
data['Electric_Vehicle_Indicator'] = data['Electric_Vehicle_Indicator'].astype('category')

# Tells us data types of variables
print(data.dtypes)

# Call Crosstab function on each pair of Variables
print(print_cross_tab(data['Make']))
print(print_cross_tab(data['Model.Year']))
print(print_cross_tab(data['Body.Style']))
print(print_cross_tab(data['Anti.Theft.Device.Indicator']))

data = data[["Make", "Model.Year", "Body.Style", "Anti.Theft.Device.Indicator", "Electric_Vehicle_Indicator"]]

print(data.shape)

# Chi-squared test
chi_1 = stats.chi2_contingency(print_cross_tab(data['Make']))
print(chi_1.pvalue)

chi_2 = stats.chi2_contingency(print_cross_tab(data['Model.Year']))
print(chi_2.pvalue)

chi_3 = stats.chi2_contingency(print_cross_tab(data['Body.Style']))
print(chi_3.pvalue)

chi_4 = stats.chi2_contingency(print_cross_tab(data['Anti.Theft.Device.Indicator']))
print(chi_4.pvalue)


print(data['Body.Style'].nunique())



#unique_values = data['Make'].unique()
#print(unique_values)


def cat_to_int(v):
    num_levels = v.nunique()  # Number of unique categories in the column
    encoding_dictionary = {}
    unique_values = v.unique()  # Get unique values from the column

    for i, value in enumerate(unique_values):
        encoding_dictionary[value] = i  # Add key-value pair to the dictionary

    return encoding_dictionary

# print(cat_to_int(data['Make']))


def numeric_features_category(dt):
    copy_data = dt.copy()
    copy_data['Make'] = copy_data['Make'].map(cat_to_int(data['Make']))
    copy_data['Body.Style'] = copy_data['Body.Style'].map(cat_to_int(data['Body.Style']))
    copy_data['Model.Year'] = copy_data['Model.Year'].map(cat_to_int(data['Model.Year']))
    copy_data['Anti.Theft.Device.Indicator'] = copy_data['Anti.Theft.Device.Indicator'].map(cat_to_int(data['Anti.Theft.Device.Indicator']))
    return copy_data

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

new_data = numeric_features_category(data)
print(new_data)
print(new_data['Make'].nunique())
print(new_data['Body.Style'].nunique())
print(new_data['Model.Year'].nunique())
print(new_data['Anti.Theft.Device.Indicator'].nunique())




# Defining our X and Y.

X = new_data[['Make', 'Model.Year', 'Body.Style', 'Anti.Theft.Device.Indicator']]
Y = new_data['Electric_Vehicle_Indicator']

# Splitting data into training and validating sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y)
print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)

# Using Decision Trees model

model_dtree = DecisionTreeClassifier(max_depth= 8, min_samples_leaf= 8)
model_dtree.fit(X_train, Y_train)

Y_predicted = model_dtree.predict(X_valid)
print("Training Model Score for Decision Tree",model_dtree.score(X_train, Y_train))
print("Validation Model Score for Decision Tree",model_dtree.score(X_valid, Y_valid))
#print(Y_predicted, Y_valid)
compare = pd.DataFrame({
    "Y_predicted": Y_predicted,
    "Y_valid": Y_valid
})
print(compare.head(10))
print(compare.shape)

'''
confusion_matrix = metrics.confusion_matrix(Y_valid, Y_predicted)
sns.set(font_scale= 1.5)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix, annot = True, fmt = 'd', xticklabels = model_dtree.classes_, yticklabels = model_dtree.classes_)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
#plt.savefig('confusion_matrix.png')
plt.show()
#print(confusion_matrix)
'''

# Using Random Forest model

model_rf = RandomForestClassifier(n_estimators= 75
        , min_samples_leaf=10)

model_rf.fit(X_train, Y_train)
print("Training Model Score for Random Forest",model_rf.score(X_train, Y_train))
print("Validation Model Score for Random Forest ",model_rf.score(X_valid, Y_valid))
Y_predicted_rf = model_rf.predict(X_valid)

print(data.head())

## Using Gradient Boosting classifier

model_gb = GradientBoostingClassifier(n_estimators= 325, max_depth = 5, min_samples_leaf= 0.2)
model_gb.fit(X_train, Y_train)
print("Training Model Score for Gradient Boosting",model_gb.score(X_train, Y_train))
print("Validation Model Score for Gradient Boosting ",model_gb.score(X_valid, Y_valid))

## Using NN classifier (tried but wasnt helpful)

model_nn = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500)
model_nn.fit(X_train, Y_train)
print("Training Model Score for Neural Network",model_nn.score(X_train, Y_train))
print("Validation Model Score for Neural Network",model_nn.score(X_valid, Y_valid))

# Create dataframe to store Model accuracy scores

print('Dataframe for scores:')

scores_df = pd.DataFrame(columns=['Model_name', 'Accuracy_Score'])
scores_df['Model_name'] = ['Decision Tree', 'Random Forest', 'Gradient Boosting','Neural Network']
scores_df['Accuracy_Score'] = [model_dtree.score(X_valid, Y_valid), model_rf.score(X_valid, Y_valid), model_gb.score(X_valid, Y_valid),
                                model_nn.score(X_valid, Y_valid)]
print(scores_df)
'''
# Create an ROC curve to compare model performance
plot_roc_curve(Y_valid,Y_predicted_rf )
print(f'model 1 AUC score: {roc_auc_score(Y_valid, Y_predicted_rf)}')
'''

# Confusion Matrix for the Best model based on Accuracy Scores
confusion_matrix = metrics.confusion_matrix(Y_valid, Y_predicted_rf)
sns.set(font_scale= 1.5)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix, annot = True, fmt = 'd', xticklabels = model_rf.classes_, yticklabels = model_rf.classes_)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for RF')
plt.savefig('confusion_matrix.png')
plt.show()

# Extract values
tn, fp, fn, tp = confusion_matrix.ravel()
print(tn, fp, fn, tp)
# Calculate Hit Rate (Recall)
hit_rate = tp / (tp + fn)
print("Hit Rate for Random Forest: ",hit_rate)

# TODO: create a lift chart or ROC curve to compare model performances