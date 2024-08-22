<H3>ENTER YOUR NAME</H3>
YUVASAKTHI N.C
<H3>ENTER YOUR REGISTER NO.</H3>
212222240120
<H3>EX. NO.1</H3>
<H3>DATE</H3>
22-08-2024
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
Import Libraries
```

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

```
Read the dataset
```
df=pd.read_csv("Churn_Modelling.csv")
```
Checking Data
```
df.head()
df.tail()
df.columns
```
Check the missing data
```
df.isnull().sum()
df.duplicated()
y = df.iloc[:, -1].values
print(y)

```
Check for duplicates
```
df.duplicated()
```
Check for outliers
```
df.describe()
```
Dropping string values data from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

```
Normalize the dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

```
Split the dataset
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)

```
Training and testing model

```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/ec251a4d-618e-47c5-bf5e-6a7ae777c00c)

![image](https://github.com/user-attachments/assets/9065371d-0647-4f6d-9c9a-91323d0048b2)

![image](https://github.com/user-attachments/assets/68644f14-c445-424a-888e-adb498fe3529)

![image](https://github.com/user-attachments/assets/c0db3c51-d728-4ec7-a44c-ee1a7348a268)

![image](https://github.com/user-attachments/assets/13bb63b8-89b9-4078-b07b-564f25f099c8)

![image](https://github.com/user-attachments/assets/a71e9cf8-eb63-45f0-91bd-66fb7e783694)

![image](https://github.com/user-attachments/assets/1c734219-1e26-444e-9121-25202f6df22b)

![image](https://github.com/user-attachments/assets/f4c52c63-4a99-47a2-bd8e-bf546cd666f4)

![image](https://github.com/user-attachments/assets/2f349aad-d4cb-4faa-bf54-47b975d588c8)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


