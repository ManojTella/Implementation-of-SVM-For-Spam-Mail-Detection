# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Manoj Guna Sundar Tella.
RegisterNumber: 212221240026.
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![img1](https://user-images.githubusercontent.com/94883876/173190953-9ed3a662-1f8d-43a1-8fb4-e8852292f010.jpg)
![img2](https://user-images.githubusercontent.com/94883876/173190962-c7b5575a-9a40-48eb-bac3-d9bd87f0aac5.jpg)
![img3](https://user-images.githubusercontent.com/94883876/173190967-c1dacaa6-e3a2-4cd3-86bc-c179afba189f.jpg)
![img4](https://user-images.githubusercontent.com/94883876/173190971-5adc5127-2e13-4ae4-997f-bc0c33db92df.jpg)
![img5](https://user-images.githubusercontent.com/94883876/173190992-4b3016d5-2f6f-4385-b770-6e8d57152ac2.jpg)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
