# Acute Liver Failure Patients Analysis using Machine Learning

Acute liver failure is the appearance of severe complications rapidly after the first signs of liver disease.Since 1990, the JPAC Center for Health Diagnosis and Control, has conducted nationwide surveys of Indian adults. Using trained personnel, the center had collected a wide variety of demographic and health information using direct interviews, examinations, and blood samples. The data setconsists of selected information from 8,785 adults 20 years of age or older taken from the 2008–2009 and 2014–2015 surveys.

This dataset is downloaded from Kaggle and the link is mentioned below:
https://www.kaggle.com/rahul121/acute-liver-failure

the steps included in this analysis are:
1. Data Collection
2. Data Analysis
3. Data Visualization
4. Data Cleaning
5. Algorithm selection
6. Prediction
7. Saving the Model

### Importing Libraries
```markdown
import pandas as pd
import seaborn as sb
from matplotlib.pyplot import scatter as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
```

## Step 1: Data Collection
Collecting the data into dataframe from the file path
```markdown
df=pd.read_csv("G:\\COURSES\\python with mac learning Delithe\\DataSets\\Project\\ALF_Data.csv")
```
## Step 2: Data Analysis
Shows the number of rows and columns in the dataset (rows x columns)
```markdown
df.shape

Output: 
(8785, 30)
```

Shows first 5 rows of dataset
```markdown
df.head()
```
image

Shows last 5 rows of dataset
```markdown
df.tail()
```
image

2785 records donot have ALF,which is the target variable.
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

Gives info about the null values in data,memory usage and datatypes.
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

Describes the data
```markdown
df.describe()
```
image

Replacing M and F to integer values
```markdown
classes={'M':0,'F':1}
df.replace(classes,inplace=True)
```

Checking unique instances in Region column
```markdown
df['Region'].unique()

Output:
array(['east', 'south', 'north', 'west'], dtype=object)
```

Replacing regions bt integers
```markdown
cla={'east':1,'west':2,'north':3,'south':4}
df.replace(cla,inplace=True)
```

Finding unique instances in 'Source of care' column
```markdown
df['Source of Care'].unique()

Output:
array(['Governament Hospital', 'Never Counsulted', 'Private Hospital',
       'clinic', ' '], dtype=object)
```

Replacing Source of care instances by integers
```markdown
ses={'Governament Hospital':1,'Never Counsulted':2,'Private Hospital':3,'clinic':4,' ':5}
df.replace(ses,inplace=True)
```

## Step 3: Data Visualization
```markdown
sb.countplot(df['Region'])
```
It shows more records are of the people from east

