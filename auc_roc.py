from sklearn.datasets import make_classification
import numpy as np
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
  


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV   
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#KNN
knn = KNeighborsClassifier(n_neighbors=2)
#Train the model using the training sets
knn.fit(X_train, y_train)

#Logistic Regression
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train)

#Naive Bayers
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)

#Stocastic Gradient Descent
clf = SGDClassifier(loss="hinge")
calibrated_clf=CalibratedClassifierCV(clf,cv=5,method='sigmoid')
calibrated_clf.fit(X_train, y_train)

#Decision Tree
classifiers = DecisionTreeClassifier(criterion='entropy',random_state=0) 
classifiers.fit(X_train,y_train)

#Random Forest
model = RandomForestClassifier(n_estimators=100, criterion='entropy',random_state=0)
model.fit(X_train,y_train)

r_probs = [0 for _ in range(len(y_test))]
knn_probs = knn.predict_proba(X_test)
classifier_probs = classifier.predict_proba(X_test)
classifiers_probs = classifiers.predict_proba(X_test)
calibrated_clf_probs = calibrated_clf.predict_proba(X_test)
gnb_probs = gnb.predict_proba(X_test)
model_probs = model.predict_proba(X_test)

knn_probs = knn_probs[:, 1]
classifier_probs = classifier_probs[:, 1]
calibrated_clf_probs = calibrated_clf_probs[:, 1]
gnb_probs = gnb_probs[:, 1]
classifiers_probs = classifiers_probs[:, 1]
model_probs = model_probs[:, 1]

from sklearn.metrics import roc_curve, roc_auc_score
r_auc = roc_auc_score(y_test, r_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
classifier_auc = roc_auc_score(y_test, classifier_probs)
classifiers_auc = roc_auc_score(y_test, classifiers_probs)
gnb_auc = roc_auc_score(y_test, gnb_probs)
calibrated_clf_auc = roc_auc_score(y_test, calibrated_clf_probs)
model_auc = roc_auc_score(y_test, model_probs)


print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('KNN: AUROC = %.3f' % (knn_auc))
print('Naive Bayes: AUROC = %.3f' % (gnb_auc))
print('DT: AUROC = %.3f' % (classifiers_auc))

print('LG: AUROC = %.3f' % (classifier_auc))

print('SGD: AUROC = %.3f' % (calibrated_clf_auc))
print('RF: AUROC = %.3f' % (model_auc))


r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
classifier_fpr, classifier_tpr, _ = roc_curve(y_test, classifier_probs)
classifiers_fpr, classifiers_tpr, _ = roc_curve(y_test, classifiers_probs)
gnb_fpr, gnb_tpr, _ = roc_curve(y_test, gnb_probs)
calibrated_clf_fpr, calibrated_clf_tpr, _ = roc_curve(y_test, calibrated_clf_probs)
model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)


import matplotlib.pyplot as plt

plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN (AUROC = %0.3f)' % knn_auc)
plt.plot(classifier_fpr, classifier_tpr, marker='.', label='Logistic Regression (AUROC = %0.3f)' % classifier_auc)
plt.plot(classifiers_fpr, classifiers_tpr, marker='.', label='Decision Tree (AUROC = %0.3f)' % classifiers_auc)
plt.plot(calibrated_clf_fpr, calibrated_clf_tpr, marker='.', label='Stochastic Gradient Descent (AUROC = %0.3f)' % calibrated_clf_auc)
plt.plot(gnb_fpr, gnb_tpr, marker='.', label='Naive Bayers (AUROC = %0.3f)' % gnb_auc)
plt.plot(model_fpr, model_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % model_auc)
x=[0.2,0.4,0.6,0.8,1.0,1.2]
y=[0.1,0.2,0.3,0.4,0.5,0.6]
plt.plot(x,y)
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()


