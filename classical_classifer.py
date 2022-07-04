import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head()); 
print(test.head()); 

train.info()
test.info() 

mean = [train['PassengerId'].mean(), train['Age'].mean(), train['Fare'].mean()]
std = [train['PassengerId'].std(), train['Age'].std(), train['Fare'].std()]

def transform(): 
    le = LabelEncoder()

    for col in ['Sex', 'Embarked']:
        le.fit(train[col])
        train[col] = le.transform(train[col])
    
    print(train.head())

    