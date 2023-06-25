"""
| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| PassengerId    | Unique key                                                                            |
| Pclass         | Ticket class                                                                          |
| Name           | Passenger's name                                                                      |
| Sex            | Gender                                                                                |
| Age            | Age in years                                                                          |
| SibS           | # of siblings / spouses aboard the Titanic	                                           |
| Parch          | # of parents / children aboard the Titanic	                                           |
| Ticket         | Ticket number	                                                                       |
| Fare           | Passenger fare                                                                        |
| Cabin          | Cabin number                                                                          |
| Embarked       | Port of Embarkation                                                                   |
| Survived       | Survival status                                                                       |
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import re
import scipy
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Section 1: Data loading ---------------------------------------------------------------------------------------

df_titanic = pd.read_csv('titanic.csv', sep = ',')

  # Identify variables' data type and convert them if necessary

df_titanic.info()
df_titanic.describe()

"""
Input:
    * PassengerId: Numerical variable (Discrete)
    * Pclass: Categorical variable (Ordinal)
    * Name: Text variable
    * Sex: Categorical variable (Nominal)
    * Age: Numerical variable (Discrete)
    * SibSp: Numerical variable (Discrete)
    * Parch: Numerical variable (Discrete)
    * Ticket: Text variable
    * Fare: Numerical variable (Continuous)
    * Cabin: Text variable
    * Embarked: Categorical variable (Nominal)
  
Output:
    * Survived: Categorical variable (Nominal)
"""

# Section 2: Data Preprocessing ---------------------------------------------------------------------------------------

  # Identify duplicated variables and erase if necessary
df_titanic.duplicated().sum()

  # Remove duplicated rows
print('Before:', len(df_titanic))
df_titanic.drop_duplicates(inplace = True)
print('After:', len(df_titanic))

  # Identify null variables and erase if necessary

df_titanic.isnull().sum()

  # Add values in null cells
df_titanic[['Sex','Age']].groupby('Sex').median()
median_M = df_titanic[['Sex','Age']].groupby('Sex').median().iloc[1,0]
median_F = df_titanic[['Sex','Age']].groupby('Sex').median().iloc[0,0]

nan_age_male = df_titanic[(df_titanic['Sex'] == 'male')&(df_titanic['Age'].isnull())].index
nan_age_female = df_titanic[(df_titanic['Sex'] == 'female')&(df_titanic['Age'].isnull())].index

df_titanic.loc[nan_age_male, 'Age'] = median_M
df_titanic.loc[nan_age_female, 'Age'] = median_F

mode_embarked = df_titanic['Embarked'].mode()[0]
nan_embarked_index = df_titanic[df_titanic['Embarked'].isnull()].index
df_titanic.loc[nan_embarked_index, 'Embarked'] = mode_embarked

df_titanic['Embarked'].unique()

  # Erase Cabin column due to numerous null values & Name + Ticket columns due to their little contribution

df_titanic.drop('Cabin', axis = 1, inplace = True)
df_titanic.drop('Name', axis = 1, inplace = True)
df_titanic.drop('Ticket', axis = 1, inplace = True)

df_titanic.info()

# Apply Feature Engineer to convert the categorical data types into numbers

  # gender (male: 1, female: 0)
df_titanic['Sex_encoder'] = df_titanic['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df_titanic[['Sex','Sex_encoder']]

  # Embarked ('S': 0, 'C': 1, 'Q': 2)
terms_type_mapping = {'S': 0, 'C': 1, 'Q': 2}
label_encoder_s1 = preprocessing.LabelEncoder()
df_titanic['Embarked_encoder'] = df_titanic['Embarked'].map(terms_type_mapping)
df_titanic['Embarked_encoder'] = label_encoder_s1.fit_transform(df_titanic['Embarked_encoder'])
df_titanic[['Embarked','Embarked_encoder']]

  # Convert data types

df_titanic['PassengerId'] = df_titanic['PassengerId'].astype('object')
df_titanic['Embarked'] = df_titanic['Embarked'].astype('object')
df_titanic['Age'] = df_titanic[['Age']].astype(int)
df_titanic['Fare'] = df_titanic[['Fare']].astype(int)
df_titanic['SibSp'] = df_titanic[['SibSp']].astype(int)
df_titanic['Parch'] = df_titanic[['Parch']].astype(int)
df_titanic['Survived'] = df_titanic[['Survived']].astype('category')
df_titanic['Pclass'] = df_titanic[['Pclass']].astype('category')
df_titanic['Sex_encoder'] = df_titanic[['Sex_encoder']].astype('category')
df_titanic['Embarked_encoder'] = df_titanic[['Embarked_encoder']].astype('category')

# Reconstructure data

df_titanic = df_titanic[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex_encoder', 'Embarked_encoder','Survived']]
df_titanic.head()

# Section 3: Univariate Analysis ---------------------------------------------------------------------------------------
  # Categorical Variables

cat_cols = df_titanic.select_dtypes('category').columns
cat_cols = cat_cols.tolist()

for column in cat_cols:
  print('\n* Column:', column)
  print(len(df_titanic[column].unique()), 'unique values')

def univariate_analysis_categorical_variable(df, group_by_col):
    print(df[group_by_col].value_counts())
    df[group_by_col].value_counts().plot.bar(figsize=(5, 6),rot=0)
    plt.show()

for cat in cat_cols:
  print('Variable: ', cat)
  univariate_analysis_categorical_variable(df_titanic, cat)
  print()

"""
Comment:

  * P-class: Most passengers were from the 3rd class, while 2nd class occupied a very small number compared to others
  * Sex_encoder: Male gender dominates the porportion
  * Embarked_encoder: Significant number of passengers were on board in Southampton
  * Survived: the decreased is nearly double the number of the survival
"""  

  # Numerical Variables

num_cols = df_titanic.select_dtypes('number').columns
num_cols = num_cols.tolist()

for column in num_cols:
  print('\n* Column:', column)
  print(len(df_titanic[column].unique()), 'unique values')

def univariate_analysis_continuous_variable(df, feature):
    print("Describe:")
    print(feature.describe(include='all'))
    print("Mode:", feature.mode())
    print("Range:", feature.values.ptp())
    print("IQR:", scipy.stats.iqr(feature))
    print("Var:", feature.var())
    print("Std:", feature.std())
    print("Skew:", feature.skew())
    print("Kurtosis:", feature.kurtosis())

def check_outlier(df, feature):
    plt.boxplot(feature)
    plt.show()
    Q1 = np.percentile(feature, 25)
    Q3 = np.percentile(feature, 75)
    n_O_upper = df[feature > (Q3 + 1.5*scipy.stats.iqr(feature))].shape[0]
    print("Number of upper outliers:", n_O_upper)
    n_O_lower = df[feature < (Q1 - 1.5*scipy.stats.iqr(feature))].shape[0]
    print("Number of lower outliers:", n_O_lower)
    # Percentage of ouliers
    outliers_per = (n_O_lower + n_O_upper)/df.shape[0]
    print("Percentage of ouliers:", outliers_per)
    return Q1, Q3, n_O_upper, n_O_lower, outliers_per

def univariate_visualization_analysis_continuous_variable_new(feature):
    # Histogram
    feature.plot.kde()
    plt.show()
    feature.plot.hist()
    plt.show()

for con in num_cols:
  print('Variable: ', con)
  univariate_analysis_continuous_variable(df_titanic, df_titanic[con])
  check_outlier(df_titanic, df_titanic[con])
  univariate_visualization_analysis_continuous_variable_new(df_titanic[con])
  print()

"""
Comment:

  * age: a few outliers (~ 42 outliers), most common value is 29 year old, positive Kurtosis and mildly symmetric skewness (a bit like Normal Distribution)
  * Sibsb: a few outliers (~ 46 outliers), most common value is 0, highly positive Kurtosis with 2 tops and right-sided skewness
  * Parch: a lot of outliers (~ 213 outliers), most common value is 0, positive Kurtosis and right-sided skewness
  * Fare: a lot of outliers (~ 114 outliers), most common value is 7, highly positive Kurtosis and right-sided skewness
"""

# Section 4: Bivariate Analysis: Input - Output ---------------------------------------------------------------------------------------
  # Numerical - Categorical

output = 'Survived'
cat_cols.remove('Survived')

# ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols

def variables_cont_cat(df, col1, col2):
    df_sub = df[[col1, col2]]
    plt.figure(figsize=(5,6))
    sns.boxplot(x=col1, y=col2, data=df_sub, palette="Set3")
    plt.show()
    chuoi = str(col2)+' ~ '+str(col1)
    model = ols(chuoi, data=df_sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('ANOVA table: ', anova_table)

col1 = 'Survived'
for i in range(0, len(num_cols)):
    col2 = num_cols[i]
    print('2 variables:', col1, 'and', col2)
    variables_cont_cat(df_titanic, col1, col2)
    print()

"""
Comment:

  * survived and age: suggests that there's an influence because the P-value (0.02) < 0.05
  * survived and SibSp: suggests that there's no influence because the P-value (0.29) > 0.05
  * survived and parch: suggests that there's an influence because the P-value (0.01) < 0.05
  * survived and fare: suggests that there's an influence because the P-value (5.85e-15) < 0.05

There's a weak relationship between input(sibsp) and output(survived)
"""

# z-test

from statsmodels.stats.weightstats import ztest

"""
Check the Numerical variables in 2 group (No: 0, Yes: 1) of survived

*  H0: There is no mean difference in turn of Numerical variables between No: 0, Yes: 1
*  H1: There is mean difference in turn of Numerical variables between No: 0, Yes: 1
"""

def z_test_loop(data, group_column, value_columns, alpha):

    results = {}
    for column in value_columns:
        group1_data = data[data[group_column] == 0][column]
        group2_data = data[data[group_column] == 1][column]
        z_score, p_value = ztest(group1_data, group2_data, value=group1_data.mean())
        if p_value > alpha:
            result = "Accept the null hypothesis that the means are equal."
        else:
            result = "Reject the null hypothesis that the means are equal."
        results[column] = result
    return results

group_column = 'Survived'
alpha = 0.05

for i in range(len(num_cols)):
    value_columns = [num_cols[i]]
    results = z_test_loop(df_titanic, group_column, value_columns, alpha)
    for column, result in results.items():
        print("Column: {}".format(column))
        print(result)
        print()
'''
Comment:

Since p-value of all variables is less than 0.05, we have enough evidence to reject hypothesis H0 
  => there is a relationship between numerical variables and the outcome variable
'''

  # Categorical - Categorical

col2 = 'Survived'
lst = []

def categorical_categorical(feature1, feature2):
    # Contingency table
    table_FB = pd.crosstab(feature1, feature2)
    print(table_FB)
    table_FB.plot(kind='bar', stacked=True, figsize=(5, 6),rot=0)
    plt.show()
    table_FB.plot.bar(figsize=(5, 6))
    plt.show()

    # Chi-Square Test
    stat, p, dof, expected = chi2_contingency(table_FB)
    print('dof=%d' % dof)
    print('p=', p)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
        x1 = feature1.name
        x2 = feature2.name
        chuoi = x1 + ' and ' + x2
        return chuoi
    else:
        print('Independent (fail to reject H0)')

for i in range(0, len(cat_cols)):
    col1 = cat_cols[i]
    print('2 variables:', col1, 'and', col2)
    chuoi = categorical_categorical(df_titanic[col1], df_titanic[col2])
    lst.append(chuoi)
    print()

"""
Comment:

  * Pclass and Survived: p-value < alpha (0.05) => reject the null hypothesis => 2 dependent variables
  * Sex_encoder and Survived: p-value < alpha (0.05) => reject the null hypothesis => 2 dependent variables
  * Embarked_encoder and Survived: p-value < alpha (0.05) => reject the null hypothesis => 2 dependent variables
"""

# Section 4: Bivariate Analysis: Input - Input ---------------------------------------------------------------------------------------
  # Numerical - Numerical

for i in range(0, len(num_cols)):
    col1 = num_cols[i]
    for j in range(i+1, len(num_cols)):
        col2 = num_cols[j]
        print('Correlation between 2 variables:', col1, 'and', col2)
        print(df_titanic[[col1, col2]].corr())
        print('Pearson Correlation between 2 variables:', col1, 'and', col2)
        print(stats.pearsonr(df_titanic[col1], df_titanic[col2]))
        print('Spearman Correlation between 2 variables:', col1, 'and', col2)
        print(stats.spearmanr(df_titanic[col1], df_titanic[col2]))
        sns.pairplot(df_titanic[[col1, col2]])
        plt.show()
        print()

"""
Comment:

  * Age and SibSp: moderate negative linear relationship and moderate negative monotonic relationship
  * Age and Parch: moderate negative linear relationship and moderate negative monotonic relationship
  * Age and Fare: almost no linear relationship and almost no monotonic relationship
  * SibSp and Parch: almost no linear relationship and almost no monotonic relationship
  * SibSp and Fare: almost no linear relationship and almost no monotonic relationship
  * Parch and Fare: almost no linear relationship and almost no monotonic relationship
"""

# Section 4: Bivariate Analysis: Input - Input
  # Categorical - Categorical

lst = []
for i in range (0, len(cat_cols)):
  col1 = cat_cols[i]
  for j in range (i+1, len(cat_cols)):
    col2 = cat_cols[j]
    print('2 variables:', col1, 'and', col2)
    chuoi = categorical_categorical(df_titanic[col1], df_titanic[col2])
    lst.append(chuoi)
    print()

"""
Comment:

  * Pclass and Sex_encoder: p-value < alpha (0.05) => Reject the null hypothesis => 2 dependent variables
  * Pclass and Embarked_encoder: p-value < alpha (0.05) => Reject the null hypothesis => 2 dependent variables
  * Sex_encoder and Embarked_encoder: p-value (0.002) < alpha (0.05) => Reject the null hypothesis => 2 dependent variables  
"""

# Section 4: Bivariate Analysis: Input - Input
  # Numerical - Categorical

for i in range(0, len(cat_cols)):
    col1 = cat_cols[i]
    for j in range(0, len(num_cols)):
        col2 = num_cols[j]
        print('2 variables:', col1, 'and', col2)
        variables_cont_cat(df_titanic, col1, col2)
        print()

"""
Comment:

  * Pclass and Age: suggests that there's an influence because the P-value (1.06e-24) < 0.05
  * Pclass and SibSp: suggests that there's an influence because the P-value (0.02) < 0.05
  * Pclass and Parch: suggests that there's no influence because the P-value (0.85) > 0.05
  * Pclass and Fare: suggests that there's an influence because the P-value (8.77e-85) < 0.05
---
  * Sex_encoder and Age: suggests that there's an influence because the P-value (0.004) < 0.05
  * Sex_encoder and SibSp: suggests that there's an influence because the P-value (0.0006) < 0.05
  * Sex_encoder and Parch: suggests that there's an influence because the P-value (1.07e-13) < 0.05
  * Sex_encoder and Fare: suggests that there's an influence because the P-value (4.23e-08) < 0.05
---
  * Embarked_encoder and Age: suggests that there's no influence because the P-value (0.4) > 0.05
  * Embarked_encoder and SibSp: suggests that there's no influence because the P-value (0.11) > 0.05
  * Embarked_encoder and Parch: suggests that there's no influence because the P-value (0.04) > 0.05
  * Embarked_encoder and Fare: suggests that there's an influence because the P-value (1.53e-16) < 0.05

Recommend the following:
  Erase outliers:
  * Age: There are a few outliers (~42 outliers).
  * SibSp: There are a few outliers (~46 outliers).
  * Parch: There are a lot of outliers (~213 outliers).
  * Fare: There are a lot of outliers (~114 outliers).

Based on the statistical tests and analysis:

  * All other variables (Age, Parch, Fare) show a significant influence on the target variable (Survived), 
as indicated by the ANOVA tests with p-values of 0.02, 0.01, and 5.85e-15, respectively (< 0.05).

For categorical variables:

  * Pclass, Sex_encoder, and Embarked_encoder show a significant relationship with the target variable (Survived), 
as indicated by the Chi-Square tests with p-values less than 0.05.
"""

# Section 5: Data Prediction ---------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Testing the accuracy score by applying 6 different Algorithms:

  * Logistic Regression
  * Linear Discriminant Analysis
  * K Nearest Neighbors
  * Decision Tree
  * Naive Bayes
  * Support Vector Machine
"""

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd

  # Separate the input variables (features) and output variable (target)
X = df_titanic[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex_encoder', 'Embarked_encoder']]
y = df_titanic['Survived']

  # Create a list of models to evaluate
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K Nearest Neighbors', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Support Vector Machine', SVC()))

  # Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
seed = 6

best_model = None
best_score = 0.0

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean()} ({cv_results.std()})"
    print(msg)

    score_mean = cv_results.mean()
    if score_mean > best_score:
        best_score = score_mean
        best_model = model

selected_model = best_model

"""
Comment:

The Logistic Regression, Linear Discriminant Analysis, K-nearest Neighbors, Decision Tree, and Naive Bayes all 
give an accuracy score of around 78 - 80% (except Support Vector Machine with a very low precision score)

  => Choosing Logistic Regression to predict new values due to its highest precision (80%)
"""

print(selected_model)

selected_model.fit(X, y)  # Fit the selected model with the training data

  # Predict new value to see which group it belongs to
fare_by_pclass = df_titanic.groupby('Pclass')['Fare'].mean()
fare_by_pclass

x_new_jack = pd.DataFrame({
    'Age': [df_titanic['Age'].unique()[0]],
    'SibSp': [df_titanic['SibSp'].unique()[1]],
    'Parch': [df_titanic['Parch'].unique()[0]],
    'Fare': [fare_by_pclass[3]],
    'Pclass': [df_titanic['Pclass'].unique()[0]],
    'Sex_encoder': [df_titanic['Sex_encoder'].unique()[0]],
    'Embarked_encoder': [df_titanic['Embarked_encoder'].unique()[0]]
})

y_pred_jack = selected_model.predict(x_new_jack)
print(y_pred_jack)

x_new_rose = pd.DataFrame({
    'Age': [df_titanic['Age'].unique()[32]],
    'SibSp': [df_titanic['SibSp'].unique()[0]],
    'Parch': [df_titanic['Parch'].unique()[1]],
    'Fare': [fare_by_pclass[1]],
    'Pclass': [df_titanic['Pclass'].unique()[1]],
    'Sex_encoder': [df_titanic['Sex_encoder'].unique()[1]],
    'Embarked_encoder': [df_titanic['Embarked_encoder'].unique()[0]]
})

y_pred_rose = selected_model.predict(x_new_rose)
print(y_pred_rose)

x_new_family = pd.DataFrame({
    'Age': [df_titanic['Age'].unique()[12]],
    'SibSp': [df_titanic['SibSp'].unique()[1]],
    'Parch': [df_titanic['Parch'].unique()[2]],
    'Fare': [fare_by_pclass[2]],
    'Pclass': [df_titanic['Pclass'].unique()[2]],
    'Sex_encoder': [df_titanic['Sex_encoder'].unique()[1]],
    'Embarked_encoder': [df_titanic['Embarked_encoder'].unique()[0]]
})

y_pred_family = selected_model.predict(x_new_family)
print(y_pred_family)

"""
Comment:

Jack: Perished
Rose: Survived
Family: Survived
"""