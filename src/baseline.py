import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
def main():
    train = pd.read_csv("data/input/train.csv") 
    test = pd.read_csv("data/input/test.csv")

    mean_age = train['Age'].mean()
    train['Age'] = train['Age'].fillna(mean_age)

    y = train['Survived']
    X = train[['Age']]
    model = RandomForestClassifier()
    model.fit(X,y)

    test_mean_age = test['Age'].mean()
    test['Age'] = test['Age'].fillna(test_mean_age)
    X_test = test[['Age']]
    y_pred = model.predict(X_test)

    submission = pd.DataFrame({'PassengerID': X_test['Age'],
                           'Survived': y_pred})
    submission.to_csv("data/output/pair_prog_baseline.csv")
if __name__ == "__main__":
    main()