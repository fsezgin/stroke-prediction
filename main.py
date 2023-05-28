import pandas as pd
import numpy as np
from stroke_data_visualizer import StrokeDataVisualizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Read the datasets
stroke_prediction_train_data = pd.read_csv(r'C:\Users\furka\Documents\GitHub\stroke-prediction\train.csv')
stroke_prediction_test_data = pd.read_csv(r'C:\Users\furka\Documents\GitHub\stroke-prediction\test.csv')
#print(stroke_prediction_train_data.head())

# check features
print(stroke_prediction_train_data.columns)
# check missing value
# print(stroke_prediction_train_data.isnull().sum())

# find useless values so as not to be mistaken
cat = stroke_prediction_train_data.select_dtypes(include=['object'])
for col in cat.columns:
    print('*' * 20 +  str(col) + '*' * 20)
    print(stroke_prediction_train_data[col].unique())
    print()
    print()

# check the "unknown" and "others" misleading variables in gender and smoking_status
# drop 'other' and 'unknown' values on columns to avoid wrong predict
stroke_prediction_train_data = stroke_prediction_train_data.drop(stroke_prediction_train_data[stroke_prediction_train_data['gender'] == 'Other'].index)
stroke_prediction_test_data = stroke_prediction_test_data.drop(stroke_prediction_test_data[stroke_prediction_test_data['gender'] == 'Other'].index)
stroke_prediction_train_data = stroke_prediction_train_data.drop(stroke_prediction_train_data[stroke_prediction_train_data['smoking_status'] == 'Unknown'].index)
stroke_prediction_test_data = stroke_prediction_test_data.drop(stroke_prediction_test_data[stroke_prediction_test_data['smoking_status'] == 'Unknown'].index)
#print(stroke_prediction_train_data.head())

# copy and add a new column to make the dataset more detailed in the data visualization part
graph_data = stroke_prediction_train_data.copy()
graph_data['age_group'] = pd.cut(graph_data['age'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89],
                           labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89'])
graph_data['bmi_group'] = pd.cut(graph_data['bmi'], bins=[9, 19, 29, 39, 49, 59, 69],
                           labels=['10-19', '20-29', '30-39', '40-49', '50-59', '60-69'])

cols =['gender','ever_married', 'work_type', 'Residence_type','smoking_status']
le = preprocessing.LabelEncoder() # one-hot encoder
for col in cols:
    stroke_prediction_train_data[col] = le.fit_transform(stroke_prediction_train_data[col])
    stroke_prediction_test_data[col] = le.transform(stroke_prediction_test_data[col])
#print(stroke_prediction_train_data.head())

x = stroke_prediction_train_data.drop(['stroke'], axis=1)
y = stroke_prediction_train_data['stroke']

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=1)
lr = LogisticRegression(max_iter=1000) # change iter count other push
lr.fit(X_train, y_train)
pred = lr.predict(X_val)

param_grid = {
    'penalty' : ['l1','l2','elasticnet','None'],
    'C': np.logspace(-4,4,20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100,150,200,250,300,350]
}

GS = GridSearchCV(
    estimator= lr,
    param_grid= param_grid,
    cv=3,
    n_jobs= -1,
)

GS.fit(X_train, y_train)

tunedpredictions = GS.predict(X_val)
accuracy = accuracy_score(y_val, tunedpredictions)
print(accuracy)

visualizer = StrokeDataVisualizer(graph_data)

# Get the list of functions in the StrokeDataVisualizer class
functions = [func for func in dir(visualizer) if callable(getattr(visualizer, func)) and not func.startswith("__")]

# Loop through the functions and call them one by one
for func_name in functions:
    # Call the function
    func = getattr(visualizer, func_name)
    func()

