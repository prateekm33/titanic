import pandas as pd
import numpy as np
import re as re

def process_data(train_data, test_data):
  train = pd.read_csv(train_data, header = 0)
  test = pd.read_csv(test_data, header=0)

  train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
  test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
  train['SoloTraveler'] = 0
  test['SoloTraveler'] = 0
  train.loc[train['FamilySize'] == 1, 'SoloTraveler'] = 1
  test.loc[test['FamilySize'] == 1, 'SoloTraveler'] = 1

  train['Embarked'] = train['Embarked'].fillna('S')
  test['Embarked'] = test['Embarked'].fillna('S')
  train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
  test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

  train['Fare'] = train['Fare'].fillna(train['Fare'].median())
  test['Fare'] = test['Fare'].fillna(train['Fare'].median())
  train['Fare'] = (train['Fare'] - train['Fare'].mean()) / train['Fare'].std()
  test['Fare'] = (test['Fare'] - test['Fare'].mean()) / test['Fare'].std()

  avg_age 	   = train['Age'].mean()
  std_age 	   = train['Age'].std()
  train_null_age_count = train['Age'].isnull().sum()
  test_null_age_count = test['Age'].isnull().sum()
  train_age_fill = np.random.randint(avg_age - std_age, avg_age + std_age, size=train_null_age_count)
  test_age_fill = np.random.randint(avg_age - std_age, avg_age + std_age, size=test_null_age_count)
  train['Age'][np.isnan(train['Age'])] = train_age_fill
  test['Age'][np.isnan(test['Age'])] = test_age_fill
  train['Age'] = (train['Age'] - train['Age'].mean()) / train['Age'].std()
  test['Age'] = (test['Age'] - test['Age'].mean()) / test['Age'].std()

  train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
  test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})

  train['HasCabin'] = 0
  train.loc[pd.notnull(train['Cabin']), 'HasCabin'] = 1
  test['HasCabin'] = 0
  test.loc[pd.notnull(test['Cabin']), 'HasCabin'] = 1

  train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
  test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

  return train, test