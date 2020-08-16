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
