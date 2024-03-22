# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Find the null and duplicate values.
3.Using logistic regression find the predicted values of accuracy , confusion matrices.
4.Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Alan Zion H
RegisterNumber:  212223240004
*/
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
## Placement Data

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/0a282804-72d5-4074-b371-edc76a869334)
## Salary Data

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/04ea4b3f-944d-4fdb-bcc5-35d9fc2cfeb8)
## Checking the null function()

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/034db86e-e4d8-4767-bfc7-428061698805)
## Data Duplicate

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/39f2d865-6400-4c99-a879-8274f9ee8205)
## Print Data

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/c5b1bb71-a2c5-42ab-8f8d-bb4afde519bd)

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/c652d1be-4331-4139-a1b0-7264905f6aac)

## Data Status

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/e43b23ab-f4d8-4987-9c48-83699117170b)

## y_prediction array

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/86e02d6b-fdce-47c9-a120-0ce4e64037de)

## Accuracy value

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/e1765eed-d7fc-4d0d-9b52-bb6ddb1928a1)

## Confusion matrix

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/9f1c00c1-5d12-410b-be13-d82669d4e22c)

## Classification Report

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/e8f52c79-0f1b-4a32-a450-bb2ca34a3713)

## Prediction of LR

![image](https://github.com/ALANZION/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145743064/af048973-de29-47f5-b328-b2569a4ae0c2)










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
