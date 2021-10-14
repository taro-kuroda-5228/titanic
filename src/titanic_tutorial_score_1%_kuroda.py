import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
def main():
    train= pd.read_csv("data/input/train.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
    test= pd.read_csv("data/input/test.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2) 
    train["Age"].fillna(train.Age.mean(), inplace=True) 
    train["Embarked"].fillna(train.Embarked.mean(), inplace=True) 
    combine1 = [train]

    for train in combine1: 
            train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 
    for train in combine1: 
            train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')
            train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')
            train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')
            del train['Name']
    Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 

    for train in combine1: 
            train['Salutation'] = train['Salutation'].map(Salutation_mapping) 
            train['Salutation'] = train['Salutation'].fillna(0) 
        
    for train in combine1: 
            train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
            train['Ticket_Lett'] = train['Ticket_Lett'].apply(lambda x: str(x)) 
            train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'], np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
            train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x)) 
            del train['Ticket'] 
    train['Ticket_Lett']=train['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 

        
    for train in combine1: 
        train['Cabin_Lett'] = train['Cabin'].apply(lambda x: str(x)[0]) 
        train['Cabin_Lett'] = train['Cabin_Lett'].apply(lambda x: str(x)) 
        train['Cabin_Lett'] = np.where((train['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),train['Cabin_Lett'], np.where((train['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))
    del train['Cabin'] 
    train['Cabin_Lett']=train['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1) 
    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
    for train in combine1:
        train['IsAlone'] = 0
        train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
    train_data = train.values
    xs = train_data[:, 2:] # Pclass以降の変数
    y  = train_data[:, 1]  # 正解データ
    test["Age"].fillna(train.Age.mean(), inplace=True)
    test["Fare"].fillna(train.Fare.mean(), inplace=True)

    combine = [test]
    for test in combine:
        test['Salutation'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for test in combine:
        test['Salutation'] = test['Salutation'].replace(['Lady', 'Countess','Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        test['Salutation'] = test['Salutation'].replace('Mlle', 'Miss')
        test['Salutation'] = test['Salutation'].replace('Ms', 'Miss')
        test['Salutation'] = test['Salutation'].replace('Mme', 'Mrs')
        del test['Name']
    Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for test in combine:
        test['Salutation'] = test['Salutation'].map(Salutation_mapping)
        test['Salutation'] = test['Salutation'].fillna(0)

    for test in combine:
            test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(x)[0])
            test['Ticket_Lett'] = test['Ticket_Lett'].apply(lambda x: str(x))
            test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],
                                    np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                                '0', '0'))
            test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
            del test['Ticket']
    test['Ticket_Lett']=test['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 

    for test in combine:
            test['Cabin_Lett'] = test['Cabin'].apply(lambda x: str(x)[0])
            test['Cabin_Lett'] = test['Cabin_Lett'].apply(lambda x: str(x))
            test['Cabin_Lett'] = np.where((test['Cabin_Lett']).isin(['T', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']),test['Cabin_Lett'],
                                    np.where((test['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                                '0','0'))        
            del test['Cabin']
    test['Cabin_Lett']=test['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1).replace("G",1) 
    test["FamilySize"] = train["SibSp"] + train["Parch"] + 1

    for test in combine:
        test['IsAlone'] = 0
        test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
        
    test_data = test.values
    xs_test = test_data[:, 1:]

    from sklearn.ensemble import RandomForestClassifier

    random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=25, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=15,
                min_weight_fraction_leaf=0.0, n_estimators=51, n_jobs=4,
                oob_score=False, random_state=0, verbose=0, warm_start=False)

    random_forest.fit(xs, y)
    Y_pred = random_forest.predict(xs_test)


    import csv
    with open("Predict_result_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(test_data[:,0].astype(int), Y_pred.astype(int)):
            writer.writerow([pid, survived])

    test["Survived"]=Y_pred.astype(int)
    ex= pd.read_csv("Predict_result_data.csv")
    test["Survived"]=ex["Survived"]
    train=train.append(test, ignore_index=True)

    train=train.iloc[:,[7,12,8,10,0,11,6,4,2,9,14,13,1,3,5]]
    train_data = train.values
    xs = train_data[:, 2:] # Pclass以降の変数
    y  = train_data[:, 1]  # 正解データ
    test_data = test.values
    xs_test = test_data[:, 1:]

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(xs,y,test_size=0.008,random_state=0)

    lgb_train = lgb.Dataset(np.array(X_train),np.array(y_train))
    lgb_eval = lgb.Dataset(np.array(X_test),np.array(y_test),reference=lgb_train)
    

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': 200,
        'learning_rate': 0.003,
        'num_iterations':100,
        'feature_fraction': 0.52,
        'bagging_fraction': 0.79,
        'bagging_freq': 7,
        'verbose': 0
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=1000)
    X_pred= gbm.predict(np.array(xs_test), num_iteration=gbm.best_iteration, predict_disable_shape_check=True)

    X_pred= gbm.predict(np.array(xs_test), num_iteration=gbm.best_iteration, predict_disable_shape_check=True)
    for i in range(418):
        if X_pred[i]>=0.51:
            X_pred[i]=1
        else:
            X_pred[i]=0

    import csv
    with open("Predict_result_data.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(test_data[:,0].astype(int), X_pred.astype(int)):
        #for pid, survived in zip(test_data[:,0].astype(int), X_pred):
            writer.writerow([pid, survived])
    
if __name__ == "__main__":
    main()