import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

x = df.drop('Outcome',axis = 1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=30)

rf_clf = RandomForestClassifier(random_state=10)
rf_clf.fit(x_train,y_train)