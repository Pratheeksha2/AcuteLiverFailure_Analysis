# Acute Liver Failure Patients Analysis using Machine Learning

Acute liver failure is the appearance of severe complications rapidly after the first signs of liver disease.Since 1990, the JPAC Center for Health Diagnosis and Control, has conducted nationwide surveys of Indian adults. Using trained personnel, the center had collected a wide variety of demographic and health information using direct interviews, examinations, and blood samples. The data setconsists of selected information from 8,785 adults 20 years of age or older taken from the 2008–2009 and 2014–2015 surveys.

This dataset is downloaded from Kaggle and the link is mentioned below:
https://www.kaggle.com/rahul121/acute-liver-failure

The steps included in this analysis are:
1. Data Collection
2. Data Analysis
3. Data Visualization
4. Data Cleaning
5. Algorithm selection
6. Prediction
7. Saving the Model

### Importing Libraries
```
import pandas as pd
import seaborn as sb
from matplotlib.pyplot import scatter as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
```

## Step 1: Data Collection
Collecting the data into dataframe from the file path
```
df=pd.read_csv("G:\\COURSES\\python with mac learning Delithe\\DataSets\\Project\\ALF_Data.csv")
```

## Step 2: Data Analysis
In this dataset, the target attribute is 'ALF', which shows if patient has liver disease or not based on the observations. Hence, Its a classification problem as records need to be classified based on 0 and 1
1. Shows the number of rows and columns in the dataset (rows,columns).
```
df.shape

Output: 
(8785, 30)
```


2. Shows first 5 rows of dataset
```
df.head()
```
![](Images/1.Datahead(1).png)
![](Images/1.Datahead(2).png)


3. Shows last 5 rows of dataset
```
df.tail()
```
![](Images/2.Datatail(1).png)
![](Images/2.Datatail(2).png)

4. Counting the number of instances in each column
```
df.count()
```
![](Images/Screenshot (9).png)

2785 records donot have ALF,which is the target variable.

5. To show infromation about data
```
df.info()
```
![](Images/Screenshot (10).png)

Gives info about the null values in data,memory usage and datatypes.

6. Describes the data
```
df.describe()
```
![](Images/3. Datadescribe(1).png)
![](Images/4.datadescribe(2).png)
![](Images/5.datadescribe(3).png)

7. To replace M and F to integer values
```
classes={'M':0,'F':1}
df.replace(classes,inplace=True)
```

8. Checking unique instances in Region column
```
df['Region'].unique()

Output:
array(['east', 'south', 'north', 'west'], dtype=object)
```

9. Replacing regions bt integers
```
cla={'east':1,'west':2,'north':3,'south':4}
df.replace(cla,inplace=True)
```

10. Finding unique instances in 'Source of care' column
```
df['Source of Care'].unique()

Output:
array(['Governament Hospital', 'Never Counsulted', 'Private Hospital',
       'clinic', ' '], dtype=object)
```

11. Replacing Source of care instances by integers
```
ses={'Governament Hospital':1,'Never Counsulted':2,'Private Hospital':3,'clinic':4,' ':5}
df.replace(ses,inplace=True)
```

## Step 3: Data Visualization
1. To plot bar graph for 'Region'
```
sb.countplot(df['Region'])
```
image
It shows more records are of the people from east.

2. To plot bar graph for 'ALF'
```
sb.countplot(df['ALF'])
```
image
1 represents the patients with liver disease and 0 represents the patients having no liver disease.
This dataset has more records of patients with no liver disease.

3. To count ALF instances
```
df['ALF'].value_counts()

Output:
0.0    5536
1.0     464
Name: ALF, dtype: int64
```
image
1= 464 patients have acute liver failure
0= 5536 patients donot have liver acute liver failure

4. To plot bar grapgh for 'Gender'
```
sb.countplot(df['Gender'])
```
image
The number of male records is more than the female

5. To count the instances of 'Gender'
```
df['Gender'].value_counts()

Output:
0    4630
1    4155
Name: Gender, dtype: int64
```
image
Count of male and female records
0= Male, 1= Female

6. To check relation between SOC and ALF
```
sm(df['Source of Care'],df['ALF'],color='g')
```
image
This shows when source of care is present, patient may or might have disease.So, it isnt highly correlated in prdiction

7. Pairplot shows the relation between all attributes
```
sb.pairplot(df,hue='ALF',height=5,markers=['o','D'],diag_kind='kde',kind='reg')
```
'It shows how two attributes and their instances are correlated, if they are positively correlated or negatively by represting it by uphill and downhill.

8.To show the correlation between 2 attributes
```
df.corr()
```
image
Note: Its a part of the output as the dataset is larger. Heatmap gives complete information.

9. Heatmap
```
plt.subplots(figsize=(20,15))
sb.set(font_scale=0.8)
x=sb.heatmap(df.corr(),annot=True,cmap='coolwarm')  #now it shows correlation between the attributes
plt.show()
```
image
Heat map is used to find the correlation betweeen attributes and the attributes that are less 
correlated to eachother and highly correlated to the target variable have to be kept for analysis.
If the two attriubtes other than the target variable is kept for analysis, it decreases the accuracy of the model.

## Step 4: Data Cleaning
This is to remove or drop unwanted attributes to increase the accuracy of our model.

1. From the above heatmap, now the independent attributes that are highly correlated 
can be removed which increases the quality and accuracy of the modal.
In this the attributes:
1. gender and height are highly positive correlated
2.Weight and Body mass index , Obesity, Waist are highly positive correlated.
3. Body mass index and Obesity ,Waist are highly positive correlated.
4. Obesity and Waist are highly positive correlated.
5. Max Blood preassure and hypertension are highly correlated.
6. Age and max blood preassure are moderately correlated.
7.Age and hypertension are moderately correlated.

So,in these pairs, one attribute in each has to be removed which is less correlated
to the target attribue i.e., ALF(Acute liver failure).
1. Gender is less correlated   2. weight,obesity and BMI are less correlated than  waist. 
3. Max B.P is less correlated than Hypertension and Age 
```
df.drop('Gender',axis=1,inplace=True)
df.drop('Weight',axis=1,inplace=True)
df.drop('Maximum Blood Pressure',axis=1,inplace=True)
df.drop('Obesity',axis=1,inplace=True)
df.drop('Body Mass Index',axis=1,inplace=True)

df.drop('Region',axis=1,inplace=True)          #Region is not highly correlated to the target attribute i.e.,ALF
df.drop('Dyslipidemia',axis=1,inplace=True)    #Dyslipidemia is not highly correlated to the target attribute i.e.,ALF
df.drop('Source of Care',axis=1,inplace=True)  #source of care does not affect  target attribute i.e.,ALF
```
Drops all the above columns from dataset.

2. To drop null values if any
```
df.dropna(inplace=True)
```

3. To check shape of dataset after dropping some columns
```
df.shape

Output:
(4334, 22)
```

## Step 5: Algorithm Selection
This is a classification problem as the target attribute has 2 type of instance which are 0 and 1 to indicate if a patient is diseased or not.
So, the people have to be classified to these two groups.As this is a classification problem, I am using "Logistic Regression" algorithm.

1. Creating arrays for the model
x=df.iloc[:,:-1].values   #All rows and 0-3 columns in values and no column names its same as [:,0:4]
y=df.iloc[:,-1].values    #All rows and only target column

2. From this loop,can find which random state and test_size gives highest accuracy with printing the random_state
```
       for i in range(1,1000):
           print(i)
           x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=i)
           from sklearn.linear_model import LogisticRegression
           logreg=LogisticRegression()   #creating object
           logreg.fit(x_train,y_train)
           loregaccuracy=logreg.score(x_test,y_test)
           print(loregaccuracy*100)
```
image
From the above code, can be concluded that having test_size=0.25 and random_state=238 yields highest accuracy of 95.20%.Hence, I am using it.
This helps in choosing optimal value for random_state

3. Dividing values to training and testing set.
```
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=238)
```

4. Creating Object
```
logreg=LogisticRegression() 
logreg.fit(x_train,y_train)   #learning step (give data to model to learn)
loregaccuracy=logreg.score(x_test,y_test)
loregaccuracy*100

Output:
95.20295202952029
```

## Step 6: Prediction
This model has accuracy of 95.2% in predicting the disease.
1. To check predicted values
```
logregpred=logreg.predict(x_test)
logregpred 

Output:
array([0., 0., 0., ..., 0., 0., 0.])
```

2.To compare right vs wrong predictions
  Here its comparing to knw how many matches the actual prdiction and how many are wrongly predicted.
  Hence, for this confusion matrix is used
  ```
  conmat=confusion_matrix(y_test,logregpred)
  conmat
  
  Output:
  array([[1020,    9],
       [  43,   12]], dtype=int64)
  ```     
  
  ## Step 7: Saving the Model
  1. syntax: pickle.dump(model_name,file_name)
  ```
 import pickle
 file_name=open('ALF.pkl','wb') #write binary file
 model_name=logreg
 pickle.dump(model_name,file_name)
```

2. Loading model
This model can be used again in future by importing, for analysing similar dataset without having to do all the above steps.
```
loaded_model=pickle.load(open('ALF.pkl','rb'))
```

3. Using loaded model
```
loaded_model.score(x_test,y_test)
```
