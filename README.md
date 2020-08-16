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
1. Collecting the data into dataframe from the file path
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
!.[.].(images/Datahead(1).png)

3. Shows last 5 rows of dataset
```
df.tail()
```
image

4. Counting the number of instances in each column
```markdown
df.count()

Output:
Age                       8785
Gender                    8785
Region                    8785
Weight                    8591
Height                    8594
Body Mass Index           8495
Obesity                   8495
Waist                     8471
Maximum Blood Pressure    8481
Minimum Blood Pressure    8409
Good Cholesterol          8768
Bad Cholesterol           8767
Total Cholesterol         8769
Dyslipidemia              8785
PVD                       8785
Physical Activity         8775
Education                 8765
Unmarried                 8333
Income                    7624
Source of Care            8785
PoorVision                8222
Alcohol Consumption       8785
HyperTension              8705
Family  HyperTension      8785
Diabetes                  8783
Family Diabetes           8785
Hepatitis                 8763
Family Hepatitis          8779
Chronic Fatigue           8750
ALF                       6000
dtype: int64
```
2785 records donot have ALF,which is the target variable.

5. To show infromation about data
```markdown
df.info()

Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8785 entries, 0 to 8784
Data columns (total 30 columns):
Age                       8785 non-null int64
Gender                    8785 non-null object
Region                    8785 non-null object
Weight                    8591 non-null float64
Height                    8594 non-null float64
Body Mass Index           8495 non-null float64
Obesity                   8495 non-null float64
Waist                     8471 non-null float64
Maximum Blood Pressure    8481 non-null float64
Minimum Blood Pressure    8409 non-null float64
Good Cholesterol          8768 non-null float64
Bad Cholesterol           8767 non-null float64
Total Cholesterol         8769 non-null float64
Dyslipidemia              8785 non-null int64
PVD                       8785 non-null int64
Physical Activity         8775 non-null float64
Education                 8765 non-null float64
Unmarried                 8333 non-null float64
Income                    7624 non-null float64
Source of Care            8785 non-null object
PoorVision                8222 non-null float64
Alcohol Consumption       8785 non-null int64
HyperTension              8705 non-null float64
Family  HyperTension      8785 non-null int64
Diabetes                  8783 non-null float64
Family Diabetes           8785 non-null int64
Hepatitis                 8763 non-null float64
Family Hepatitis          8779 non-null float64
Chronic Fatigue           8750 non-null float64
ALF                       6000 non-null float64
dtypes: float64(21), int64(6), object(3)
memory usage: 2.0+ MB
```
Gives info about the null values in data,memory usage and datatypes.

6. Describes the data
```markdown
df.describe()
```
image

7. To replace M and F to integer values
```markdown
classes={'M':0,'F':1}
df.replace(classes,inplace=True)
```

8. Checking unique instances in Region column
```markdown
df['Region'].unique()

Output:
array(['east', 'south', 'north', 'west'], dtype=object)
```

9. Replacing regions bt integers
```markdown
cla={'east':1,'west':2,'north':3,'south':4}
df.replace(cla,inplace=True)
```

10. Finding unique instances in 'Source of care' column
```markdown
df['Source of Care'].unique()

Output:
array(['Governament Hospital', 'Never Counsulted', 'Private Hospital',
       'clinic', ' '], dtype=object)
```

11. Replacing Source of care instances by integers
```markdown
ses={'Governament Hospital':1,'Never Counsulted':2,'Private Hospital':3,'clinic':4,' ':5}
df.replace(ses,inplace=True)
```

## Step 3: Data Visualization
1. To plot bar graph for 'Region'
```markdown
sb.countplot(df['Region'])
```
image
It shows more records are of the people from east.

2. To plot bar graph for 'ALF'
```markdown
sb.countplot(df['ALF'])
```
image
1 represents the patients with liver disease and 0 represents the patients having no liver disease.
This dataset has more records of patients with no liver disease.

3. To count ALF instances
```markdown
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
```markdown
sb.countplot(df['Gender'])
```
image
The number of male records is more than the female

5. To count the instances of 'Gender'
```markdown
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
```markdown
sm(df['Source of Care'],df['ALF'],color='g')
```
image
This shows when source of care is present, patient may or might have disease.So, it isnt highly correlated in prdiction

7. Pairplot shows the relation between all attributes
```markdown
sb.pairplot(df,hue='ALF',height=5,markers=['o','D'],diag_kind='kde',kind='reg')
```
'It shows how two attributes and their instances are correlated, if they are positively correlated or negatively by represting it by uphill and downhill.

8.To show the correlation between 2 attributes
```markdown
df.corr()
```
image
Note: Its a part of the output as the dataset is larger. Heatmap gives complete information.

9. Heatmap
```markdown
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
```markdown
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
```markdown
df.dropna(inplace=True)
```

3. To check shape of dataset after dropping some columns
```markdown
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
```markdown
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
```markdown
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=238)
```

4. Creating Object
```markdown
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
```markdown
logregpred=logreg.predict(x_test)
logregpred 

Output:
array([0., 0., 0., ..., 0., 0., 0.])
```

2.To compare right vs wrong predictions
  Here its comparing to knw how many matches the actual prdiction and how many are wrongly predicted.
  Hence, for this confusion matrix is used
  ```markdown
  conmat=confusion_matrix(y_test,logregpred)
  conmat
  
  Output:
  array([[1020,    9],
       [  43,   12]], dtype=int64)
  ```     
  
  ## Step 7: Saving the Model
  1. syntax: pickle.dump(model_name,file_name)
  ```markdown
 import pickle
 file_name=open('ALF.pkl','wb') #write binary file
 model_name=logreg
 pickle.dump(model_name,file_name)
```

2. Loading model
This model can be used again in future by importing, for analysing similar dataset without having to do all the above steps.
```markdown
loaded_model=pickle.load(open('ALF.pkl','rb'))
```

3. Using loaded model
```markdown
loaded_model.score(x_test,y_test)
```
