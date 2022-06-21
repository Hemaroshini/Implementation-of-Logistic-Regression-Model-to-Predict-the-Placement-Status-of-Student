# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values


## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HEMAROSHINI M
RegisterNumber:  212219220015

import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))*/
```

## Output:
Original data(first five columns):


![image](https://user-images.githubusercontent.com/107909531/174729508-5c86aa80-e5df-4b1b-b904-6189673090cd.png)


Data after dropping unwanted columns(first five):


![image](https://user-images.githubusercontent.com/107909531/174729564-c131cbd1-6bcd-4073-9b8d-1615866ab32a.png)


Checking the presence of null values:


![image](https://user-images.githubusercontent.com/107909531/174729608-bfa84001-06c4-42ae-b334-4250fa720998.png)


Checking the presence of duplicated values:


![image](https://user-images.githubusercontent.com/107909531/174729642-b43983d4-5a95-4779-96e7-2d6e10a39b37.png)


Data after Encoding:


![image](https://user-images.githubusercontent.com/107909531/174729694-bcb9ced2-48de-4449-9192-a73ac642e906.png)


X Data:


![image](https://user-images.githubusercontent.com/107909531/174729757-39211ba6-ba54-4580-8300-6f7dfb829c19.png)


Y Data:


![image](https://user-images.githubusercontent.com/107909531/174729780-487cfee4-db5a-4a2f-a1a7-de6e92cd8952.png)


Predicted Values:


![image](https://user-images.githubusercontent.com/107909531/174729830-172f8c19-c9ed-4a8b-94fa-a87c6bab4251.png)


Accuracy Score:


![image](https://user-images.githubusercontent.com/107909531/174729854-d2b86138-cc94-4996-9812-aef255caa005.png)


Confusion Matrix:


![image](https://user-images.githubusercontent.com/107909531/174729887-b150690a-a881-4040-ad88-45bcdeb9aff6.png)


Classification Report:


![image](https://user-images.githubusercontent.com/107909531/174729902-cbb4f3cb-a15d-4a69-9fe4-3874343df60e.png)


Predicting output from Regression Model:


![image](https://user-images.githubusercontent.com/107909531/174729946-bbb03c39-acf9-4a99-bf6f-72f109a1f68f.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
