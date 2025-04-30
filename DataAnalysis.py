import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from scipy import stats, np_minversion
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Read data
data = pd.read_csv('cleaned_vehicle_data')

# Printing top 5 rows
print(data.head())


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

model_dtree = DecisionTreeClassifier(max_depth = 4)
model_dtree.fit(X_train, Y_train)

Y_predicted = model_dtree.predict(X_valid)
print(model_dtree.score(X_train, Y_train))
print(model_dtree.score(X_valid, Y_valid))
#print(Y_predicted, Y_valid)
compare = pd.DataFrame({
    "Y_predicted": Y_predicted,
    "Y_valid": Y_valid
})
print(compare.head(10))
print(compare.shape)

# Using Random Forest

model_rf = RandomForestClassifier(n_estimators= 50,
        max_depth=4, min_samples_leaf=10)
model_rf.fit(X_train, Y_train)
print(model_rf.score(X_train, Y_train))
print(model_rf.score(X_valid, Y_valid))