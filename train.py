import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, precision_score

ds = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\computer vision\hand-pose\hand-poses.csv')

X = ds.iloc[:, :-1]
y = ds.iloc[:,  -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print(f'The accuracy-score is: {accuracy_score(y_test, y_pred)}')
print(f'The precision-score is: {precision_score(y_test, y_pred, average="macro")}')

with open('model.pkl', 'wb') as f:
    pickle.dump(rfc, f)