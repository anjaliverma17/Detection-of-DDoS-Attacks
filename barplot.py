import pandas as pd
ufo = pd.read_csv('finaldataset.csv')

ufo.drop(['FLAGS','NODE_NAME_FROM','NODE_NAME_TO'], axis=1, inplace=True)

X = ufo.iloc[:, :-1].values 
y = ufo.iloc[:, -1].values

from sklearn.model_selection import train_test_split 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA 
pca = PCA(n_components = 16) 
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
  
explained_variance = pca.explained_variance_ratio_ 
print(explained_variance) 


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV   
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import precision_score, recall_score, auc,roc_curve, f1_score

names = ["Nearest Neighbors", "Logistic Regression",
         "Decision Tree", "Random Forest", "SGD",
         "Naive Bayes"]

classifiers=[KNeighborsClassifier(n_neighbors=2),LogisticRegression(random_state = 0),DecisionTreeClassifier(criterion='entropy',random_state=0),
             RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=0),SGDClassifier(loss="hinge"),GaussianNB()]

model_cols = []
df=pd.DataFrame(columns=model_cols)
index=0

for name, clf in zip(names, classifiers):
    clf.fit(X_train,y_train)
    df.loc[index,'Classifiers'] = name
    df.loc[index,'Train Accuracy'] = clf.score(X_train,y_train)
    index+=1
    df.loc[index,'Test Accuracy'] = clf.score(X_test,y_test)
    df.loc[index,'Precision'] = precision_score(y_test,clf.predict(X_test))
    df.loc[index,'Recall'] = recall_score(y_test,clf.predict(X_test))
    df.loc[index,'F1 Score'] = f1_score(y_test,clf.predict(X_test))
    index+=1

import seaborn as sns
sns.barplot(x='Classifiers',y='Train Accuracy', data=df, palette='hot',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Train Accuracy Comparision')
plt.show()

import seaborn as sns
sns.barplot(x='Classifiers',y='Test Accuracy', data=df, palette='hot',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Test Accuracy Comparision')
plt.show()


import seaborn as sns
sns.barplot(x='Classifiers',y='Precision', data=df, palette='hot',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Precision Comparision')
plt.show()


import seaborn as sns
sns.barplot(x='Classifiers',y='Recall', data=df, palette='hot',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model Recall Comparision')
plt.show()


import seaborn as sns
sns.barplot(x='Classifiers',y='F1 Score', data=df, palette='hot',
            edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Model F1 Score Comparision')
plt.show()
