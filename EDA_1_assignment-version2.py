import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# load preprepared merged file for EDA1 assignment focus 
df = pd.read_csv('data/Merged_bank_data.csv')
df.head()

df.drop('ID', axis=1, inplace=True)

# transform head read so view cols at one glance
df.head().T

# list name of all cols
df.columns

# shape of the df
df.shape

# view unique values in each feature of df
for i in df.columns:
    print(i)
    print(df[i].unique())
    print('---'*20)

# list of numerical cols
numerical_cols = list(df.select_dtypes(exclude=['object']))
numerical_cols

# list of categorical cols
categorical_cols = list(df.select_dtypes(include=['object']))
categorical_cols

print("\n>> Dtypes:\n{}".format(df.dtypes))

df.describe()

print(df["Y"].value_counts())

# Numerical data analysis
plt.figure(figsize=(10,8))
sns.distplot(df["nr.employed"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="nr.employed")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["euribor3m"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="euribor3m")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["cons.conf.idx"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="cons.conf.idx")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["cons.price.idx"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="cons.price.idx")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["emp.var.rate"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="emp.var.rate")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["previous"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="previous")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["campaign"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="campaign")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["duration"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="duration")
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df["age"])

get_ipython().run_line_magic('matplotlib', 'inline')
sns.boxplot(data=df, x="Y", y="age")
plt.show()

# Categorical data analysis

categorical_variables = ['job', 'marital', 'education', 'default', 'loan', 'housing', 'contact', 'month', 'day_of_week', 'poutcome', 'Y']
for col in categorical_variables:
    plt.figure(figsize=(10,4))
    sns.barplot( df[col].value_counts().values,  df[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()
    plt.legend

categorical_variables = ['job', 'marital', 'education', 'default', 'loan', 'housing', 'contact', 'month', 'day_of_week', 'poutcome', 'Y']
for col in categorical_variables:
    plt.figure(figsize=(10,4))
    #Returns counts of unique values for each outcome for each feature.
    pos_counts = df.loc[df.Y.values == 'yes', col].value_counts() 
    neg_counts = df.loc[df.Y.values == 'no', col].value_counts()
    
    all_counts = list(set(list(pos_counts.index) + list(neg_counts.index)))
    
    #Counts of how often each outcome was recorded.
    freq_pos = (df.Y.values == 'yes').sum()
    freq_neg = (df.Y.values == 'no').sum()
    
    pos_counts = pos_counts.to_dict()
    neg_counts = neg_counts.to_dict()
    
    all_index = list(all_counts)
    all_counts = [pos_counts.get(k, 0) / freq_pos - neg_counts.get(k, 0) / freq_neg for k in all_counts]

    sns.barplot(all_counts, all_index)
    plt.title(col)
    plt.tight_layout()


# Correlation Matrix
#From the heatmap we can see that there are some numerical features which share a high correlation between them, 
# e.g nr.employed and euribor3m these features share a correlation value of 0.95, and euribor3m and emp.var.rate 
# share a correlation of 0.97, which is very high compared to the other features that we see in the heatmap.
get_ipython().run_line_magic('matplotlib', 'inline')
corr = df.corr()

f, ax = plt.subplots(figsize=(10,12))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

_ = sns.heatmap(corr, cmap="YlGn", square=True, ax=ax, annot=True, linewidth=0.1)

plt.title("Pearson correlation of Features", y=1.05, size=15)

# Missing / Unknown Value
df.default.replace('unknown','no',inplace=True)

df['default'] = df.default.replace('unknown',df.default.mode([0]))

df.housing.replace('unknown',df.housing.mode()[0],inplace=True)

df['loan'] = df.default.replace('unknown',df.loan.mode()[0])

df.loc[(df['age']>60) & (df['job']=='unknown'), 'job'] = 'retired'
df.loc[(df['education']=='unknown') & (df['job']=='management'), 'education'] = 'university.degree'
df.loc[(df['education']=='unknown') & (df['job']=='services'), 'education'] = 'high.school'
df.loc[(df['education']=='unknown') & (df['job']=='housemaid'), 'education'] = 'basic.4y'
df.loc[(df['job'] == 'unknown') & (df['education']=='basic.4y'), 'job'] = 'blue-collar'
df.loc[(df['job'] == 'unknown') & (df['education']=='basic.6y'), 'job'] = 'blue-collar'
df.loc[(df['job'] == 'unknown') & (df['education']=='basic.9y'), 'job'] = 'blue-collar'
df.loc[(df['job']=='unknown') & (df['education']=='professional.course'), 'job'] = 'technician'

df['pdays'] = np.where(df['pdays'] == 999,df[df['pdays'] < 999]['pdays'].mean(),df['pdays'])

df['Y'].replace({'no':0,'yes':1},inplace=True)

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(), annot=True,cmap='viridis',linewidths=0.5,ax=ax)

df['duration'] = df['duration'].apply(lambda n:n/60).round(2)

duration_campaign = sns.scatterplot(x='duration', y='campaign',data = df,
                     hue = 'Y')
plt.axis([0,65,0,65])
plt.ylabel('Number of Calls')
plt.xlabel('Duration of Calls (Minutes)')
plt.title('The Relationship between the Number and Duration of Calls')
# Annotation
plt.show()

df = df.drop(df[df.duration < 10/60].index, axis = 0, inplace = False)

df.loc[df["age"] < 30,  'age'] = 20
df.loc[(df["age"] >= 30) & (df["age"] <= 39), 'age'] = 30
df.loc[(df["age"] >= 40) & (df["age"] <= 49), 'age'] = 40
df.loc[(df["age"] >= 50) & (df["age"] <= 59), 'age'] = 50
df.loc[df["age"] >= 60, 'age'] = 60

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

labelenc = LabelEncoder()
df[categorical_variables] = df[categorical_variables].apply(LabelEncoder().fit_transform)

X = df.drop(['Y','duration'],1)
y = df['Y']

X_train_im, X_test_im, y_train_im, y_test_im= train_test_split(X,y, test_size=0.3, random_state=0)

clf = LogisticRegression()
clf.fit(X_train_im, y_train_im)
y_pred_im = clf.predict_proba(X_test_im)

print('Accuracy on test set: {0:.4f}'.format(accuracy_score(y_test_im, clf.predict(X_test_im))))
print('Percision score on test set: {0:.4f}'.format(precision_score(y_test_im, clf.predict(X_test_im))))
print('Recall score on test set: {0:.4f}'.format(recall_score(y_test_im, clf.predict(X_test_im))))
print('F1 score on test set: {0:.4f}'.format(f1_score(y_test_im, clf.predict(X_test_im))))
print("AUC score: ", roc_auc_score(y_test_im, y_pred_im[:,1]))

X_train_b, X_test_b, y_train_b, y_test_b= train_test_split(X,y, test_size=0.3, random_state=0)

clf_b = LogisticRegression(class_weight='balanced')
clf_b.fit(X_train_b, y_train_b)
y_pred_b = clf_b.predict_proba(X_test_b)

print('Accuracy on test set: {0:.4f}'.format(accuracy_score(y_test_b, clf_b.predict(X_test_b))))
print('Percision score on test set: {0:.4f}'.format(precision_score(y_test_b, clf_b.predict(X_test_b))))
print('Recall score on test set: {0:.4f}'.format(recall_score(y_test_b, clf_b.predict(X_test_b))))
print('F1 score on test set: {0:.4f}'.format(f1_score(y_test_b, clf_b.predict(X_test_b))))
print("AUC score: ", roc_auc_score(y_test_b, y_pred_b[:,1]))

#Initialising Random Forest model
rf_clf=RandomForestClassifier(n_estimators=100,n_jobs=100,random_state=0, min_samples_leaf=100)

#Fitting on data
rf_clf.fit(X_train_im, y_train_im)

#Scoring the model on train data
score_rf=rf_clf.score(X_train_im, y_train_im)
print("Training score: %.2f " % score_rf)

#Scoring the model on test_data
score_rf=rf_clf.score(X_test_im, y_test_im)
print("Testing score: %.2f " % score_rf)

y_pred_rf = rf_clf.predict(X_test_im)
print('Accuracy on test set: {0:.4f}'.format(accuracy_score(y_test_im, rf_clf.predict(X_test_im))))
print('Percision score on test set: {0:.4f}'.format(precision_score(y_test_im, rf_clf.predict(X_test_im))))
print('Recall score on test set: {0:.4f}'.format(recall_score(y_test_im, rf_clf.predict(X_test_im))))
print('F1 score on test set: {0:.4f}'.format(f1_score(y_test_im, rf_clf.predict(X_test_im))))

rf_clf_b=RandomForestClassifier(n_estimators=100,n_jobs=100,random_state=0, min_samples_leaf=100, class_weight='balanced')

rf_clf_b.fit(X_train_b, y_train_b)

score_rf_b=rf_clf_b.score(X_train_b, y_train_b)
print("Training score: %.2f " % score_rf)

score_rf_b=rf_clf_b.score(X_test_b, y_test_b)
print("Testing score: %.2f " % score_rf)

y_pred_rf_b= rf_clf_b.predict(X_test_b)
print('Accuracy on test set: {0:.4f}'.format(accuracy_score(y_test_b, rf_clf_b.predict(X_test_b))))
print('Percision score on test set: {0:.4f}'.format(precision_score(y_test_b, rf_clf_b.predict(X_test_b))))
print('Recall score on test set: {0:.4f}'.format(recall_score(y_test_b, rf_clf_b.predict(X_test_b))))
print('F1 score on test set: {0:.4f}'.format(f1_score(y_test_b, rf_clf_b.predict(X_test_b))))
