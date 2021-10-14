import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    train_data = pd.read_csv('data/input/train.csv')
    test_data = pd.read_csv('data/input/test.csv')

    # train_dataとtest_dataの連結
    test_data['Survived'] = np.nan
    df = pd.concat([train_data, test_data], ignore_index=True, sort=False)

    from sklearn.ensemble import RandomForestRegressor

    # 推定に使用する項目を指定
    age_df = df[['Age', 'Pclass','Sex','Parch','SibSp']]

    # ラベル特徴量をワンホットエンコーディング
    age_df=pd.get_dummies(age_df)

    # 学習データとテストデータに分離し、numpyに変換
    known_age = age_df[age_df.Age.notnull()].values  
    unknown_age = age_df[age_df.Age.isnull()].values

    # 学習データをX, yに分離
    X = known_age[:, 1:]  
    y = known_age[:, 0]
    #print(X)
    #print(y)

    # ランダムフォレストで推定モデルを構築
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(X, y)

    # 推定モデルを使って、テストデータのAgeを予測し、補完
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges 

    # 年齢別生存曲線と死亡曲線
    facet = sns.FacetGrid(df[0:890], hue="Survived",aspect=2)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, df.loc[0:890,'Age'].max()))
    facet.add_legend()

    df['Title'] = df['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
    df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
    df['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
    df['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
    df['Title'].replace(['Mlle'], 'Miss', inplace=True)
    df['Title'].replace(['Jonkheer'], 'Master', inplace=True)

    df['Surname'] = df['Name'].map(lambda name:name.split(',')[0].strip())

    # 同じSurname(苗字)の出現頻度をカウント(出現回数が2以上なら家族)
    df['FamilyGroup'] = df['Surname'].map(df['Surname'].value_counts()) 

    Female_Child_Group=df.loc[(df['FamilyGroup']>=2) & ((df['Age']<=16) | (df['Sex']=='female'))]
    Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()

    Male_Adult_Group=df.loc[(df['FamilyGroup']>=2) & (df['Age']>16) & (df['Sex']=='male')]
    Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()

    Dead_list=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
    Survived_list=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

    df.loc[(df['Survived'].isnull()) & (df['Surname'].apply(lambda x:x in Dead_list)),\
                ['Sex','Age','Title']] = ['male',28.0,'Mr']
    df.loc[(df['Survived'].isnull()) & (df['Surname'].apply(lambda x:x in Survived_list)),\
                ['Sex','Age','Title']] = ['female',5.0,'Mrs']

    fare=df.loc[(df['Embarked'] == 'S') & (df['Pclass'] == 3), 'Fare'].median()
    df['Fare']=df['Fare'].fillna(fare)

    df['Family']=df['SibSp']+df['Parch']+1
    df.loc[(df['Family']>=2) & (df['Family']<=4), 'Family_label'] = 2
    df.loc[(df['Family']>=5) & (df['Family']<=7) | (df['Family']==1), 'Family_label'] = 1  # == に注意
    df.loc[(df['Family']>=8), 'Family_label'] = 0

    Ticket_Count = dict(df['Ticket'].value_counts())
    df['TicketGroup'] = df['Ticket'].map(Ticket_Count)
    sns.barplot(x='TicketGroup', y='Survived', data=df, palette='Set3')

    df.loc[(df['TicketGroup']>=2) & (df['TicketGroup']<=4), 'Ticket_label'] = 2
    df.loc[(df['TicketGroup']>=5) & (df['TicketGroup']<=8) | (df['TicketGroup']==1), 'Ticket_label'] = 1  
    df.loc[(df['TicketGroup']>=11), 'Ticket_label'] = 0
    sns.barplot(x='Ticket_label', y='Survived', data=df, palette='Set3')

    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Cabin_label']=df['Cabin'].str.get(0)
    sns.barplot(x='Cabin_label', y='Survived', data=df, palette='Set3')

    df['Embarked'] = df['Embarked'].fillna('S') 
    df = df[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','Family_label','Cabin_label','Ticket_label']]

    # ラベル特徴量をワンホットエンコーディング
    df = pd.get_dummies(df)

    # データセットを trainとtestに分割
    train = df[df['Survived'].notnull()]
    test = df[df['Survived'].isnull()].drop('Survived',axis=1)

    # データフレームをnumpyに変換
    X = train.values[:,1:]  
    y = train.values[:,0] 
    test_x = test.values

    from sklearn.feature_selection import SelectKBest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate

    # 採用する特徴量を25個から20個に絞り込む
    select = SelectKBest(k = 20)

    clf = RandomForestClassifier(random_state = 10, 
                                warm_start = True,  # 既にフィットしたモデルに学習を追加 
                                n_estimators = 26,
                                max_depth = 6, 
                                max_features = 'sqrt')
    pipeline = make_pipeline(select, clf)
    pipeline.fit(X, y)

    # フィット結果の表示
    cv_result = cross_validate(pipeline, X, y, cv= 10)

    mask= select.get_support()

    # 項目のリスト
    list_col = list(df.columns[1:])

    # 項目別の採用可否の一覧表
    for i, j in enumerate(list_col):
        print('No'+str(i+1), j,'=',  mask[i])

    # シェイプの確認
    X_selected = select.transform(X)

    PassengerId=test_data['PassengerId']
    predictions = pipeline.predict(test_x)
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
    submission.to_csv("titanic_tutorial_score_2%_kuroda.csv", index=False)
if __name__ == "__main__":
    main()