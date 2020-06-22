# Import the necessary modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
# Create the pipeline: pipeline
pipeline = Pipeline(steps=steps)
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)
# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test))) 
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
'''
Accuracy with Scaling: 0.7700680272108843
Accuracy without Scaling: 0.6979591836734694
'''
# Scaling significantly influences the quality of prediction

# ============================================ PIPELINE FOR CLASSIFICATION ========================================
# Setup the pipeline
steps = [('scaler', StandardScaler()), ('SVM', SVC())]

pipeline = Pipeline(steps)
# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100], 'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=21)
# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)
# Fit to the training set
cv.fit(X_train, y_train)
# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)
# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

'''
Accuracy: 0.7795918367346939
                 precision    recall  f1-score   support
    
          False       0.83      0.85      0.84       662
           True       0.67      0.63      0.65       318
    
    avg / total       0.78      0.78      0.78       980
    Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}
'''

# ============================================ PIPELINE FOR REGRESSION ========================================
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps=steps)
# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, param_grid=parameters)
# Fit to the training set
gm_cv.fit(X_train, y_train)
# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

# ================================================ OBJECTS TO CATEGORIES ================================================================
# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')
# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
# Print the converted dtypes
print(df.dtypes)
# ======================================== UNIQUE CATEGORICAL LABELS =======================================
num_unique_labels = df[LABELS].apply(pd.Series.nunique)
num_unique_labels.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Number of unique values')
plt.show()
#  =============================================== MISSING DATA ==================================================
# Read the dataset 'college.csv'
college = pd.read_csv('college.csv')
print(college.head())
# Print the info of college
print(college.info())
# Store unique values of 'csat' column to 'csat_unique'
csat_unique = college.csat.unique()
# Print the sorted values of csat_unique
print(np.sorted(csat_unique))
# Read the dataset 'college.csv' with na_values set to '.'
college = pd.read_csv('college.csv', na_values='.')
print(college.head())
# Print the info of college
print(college.info())
# --------------------------------- MISSING Numerical Values --------------------------
# Print the description of the data
print(diabetes.describe())
# Store all rows of column 'BMI' which are equal to 0 
zero_bmi = diabetes.BMI[diabetes.BMI == 0]
print(zero_bmi)
# Set the 0 values of column 'BMI' to np.nan
# diabetes.BMI[diabetes.BMI == 0] = np.nan
diabetes.BMI[zero_bmi.index] = np.nan
# Print the 'NaN' values in the column BMI
print(diabetes.BMI[np.isnan(diabetes.BMI)])
# ------------------------------------ MISINGNESS ANALYSIS ---------------------------------------
# Load the airquality dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')
# Create a nullity DataFrame airquality_nullity
airquality_nullity = airquality.isnull()
print(airquality_nullity.head())
# Calculate total of missing values
missing_values_sum = airquality_nullity.sum()
print('Total Missing Values:\n', missing_values_sum)
# Calculate percentage of missing values
missing_values_percent = airquality_nullity.mean() * 100
print('Percentage of Missing Values:\n', missing_values_percent)

# Import missingno as msno
import missingno as msno
# Plot amount of missingness
msno.bar(airquality)
# Plot nullity matrix of airquality
msno.matrix(airquality)
# Plot nullity matrix of airquality with frequency 'M'
msno.matrix(airquality, freq='M')
# Plot the sliced nullity matrix of airquality with frequency 'M'
msno.matrix(airquality.loc['May-1976':'Jul-1976'], freq='M')
# Plot missingness heatmap of diabetes
msno.heatmap(diabetes)
# Plot missingness dendrogram of diabetes
msno.dendrogram(diabetes)

# ============================================== FILL DUMMIES ===================================================
# function that automates creating dummy values for missing data
def fill_dummy_values(df, scaling_factor=0.075):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    col_null = col.isnull()    
    # Calculate number of missing values in column 
    num_nulls = col_null.sum()
    # Calculate column range
    col_range = col.max() - col.min()
    # Scale the random values to scaling_factor times col_range
    dummy_values = (rand(num_nulls) - 2) * scaling_factor * col_range + col.min()
    col[col_null] = dummy_values
  return df_dummy




