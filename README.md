# End-to-end Machine Learning Pipeline for AI Apprenticeship Programme

## Details
Name: Tan Han Ming Colin. Email: Colinzthm@gmail.com

## Folder structure
The base folder contains the following folders and files:
- eda.ipynd: exploratory data analysis in jupyternotebook file format
- src folder: folder containing python scripts for end-to-end maachine learning pipeline. **data_ingestion.py** contains codes to load and process data. **models.py** contains the 3 machine learning models used (logistic regression, gradient boosted classification, random forrest). **main.py** is the main code file to execute the python scripts.
- run.sh: bash script to run python scripts
- requirements.txt: names of the libraries required to run the python scripts 
- README.md: explains the machine learning pipeline employed

## Instructions for executing machine learning pipeline
Make the neccessary amendments to **categorical_var** and **numeric_var** in **data_ingestion.py**. This part is crusical as it performs the data transformation so that it can fit the machine learning pipeline. Futher explnation about this step can be found below.

Once that is done, the rest of the intructions are carried out in **main.py**:
- Call the function *connect_db* to load data from a .db file. The parameters are the file path  and the name of the .db file.
- Call the functions **clean_data** and **transform_data** to process the data. 
- Split the data into training and test sets with a ratio of 8:2. The random state is set to 42 as a defauult so that the split is the same if the code is executed multiple times
- Call the 3 model functions from **models.py** which will give the trained models as well as the accurcy metrics

The processes above will automatically execute when you run **run.sh** as part of the main function in **main.py**. This is to satisfy the main deliverables for the technical assessment. 
There are some functions that is not used in main(). These were added both as part of develpments and to prepare for future pipeline uses. 

- **view_data** to check the state of the data after any cleaning or processing made.
- **prune_customer** filters the data set to only include the assumed 4 Customer types that the company uses 
- **predict_new** is a simple function that takes an already trained model and use it on a new dataset

## Pipeline logic flow

Naturally, this machine learning pipeline follows the standard flow of loading data, processing and transforming it to prepare it to be used for machine learning and then train the models on it. 

From the exploratory data analysis (EDA), it was observed that the Page Value and Product Page Time variables were heavily right-skewed, indicating a large number of zero values in the dataset. A log transformation was applied in an attempt to normalize these distributions. However, the transformation did not significantly improve the distribution of the Page Value variable. As a result, this feature was excluded from the machine learning analysis for this assessment.In contrast, the Product Page Time variable showed a more approximately normal distribution after the log transformation was applied. Therefore, it was retained as a feature for the machine learning models.

The Customer Type variable also contained several inconsistencies in its labeling. These inconsistencies were addressed through data cleaning based on the assumptions identified during the EDA process. As part of a future-proofing approach, and under the assumption that there are only four valid customer types, a function named prune_customer was developed to filter out records that do not contain valid customer type labels. However, this function was not used in the current assessment because applying it would significantly reduce the amount of available training data. In situations where a substantially larger dataset is available, this filtering approach may be preferable to manually correcting inconsistent customer type labels, as it provides a more systematic and scalable method for maintaining data quality.

The EDA also revealed a strong correlation between the Bounce Rate and Exit Rate variables. To reduce multicollinearity and better satisfy the assumption of feature independence required by many statistical models, the Bounce Rate variable was removed from the dataset, while Exit Rate was retained for the analysis.

Categorical variables cannot be directly used by most machine learning algorithms because models typically require numerical input features. To address this, one-hot encoding was applied to convert categorical variables into a numerical format by creating a new binary column for each category in a categorical feature. Each row will contain a value of 1 for the category it belongs to and 0 for all other categories. This allows machine learning models to interpret categorical information without incorrectly assuming any ordinal relationship between categories. One-hot encoding was used on all the categorical variables identified in the EDA. 

Three machine learning models were implemented in this assessment: Logistic Regression, Gradient Boosted Classification, and Random Forest. These models were selected because they are well suited for binary classification tasks and can effectively handle datasets containing a mix of numerical and categorical features. Hyperparameter tuning was performed to optimize each model by systematically adjusting parameters that control the learning process, allowing the models to better capture patterns in the data. As no benchmark model was provided, Random Forest was used as a baseline for performance comparison. Model performance was evaluated using the accuracy metric, which measures the proportion of correctly classified observations.

All three models performed well on the dataset: Logistic Regression achieved 84% accuracy, Random Forest achieved 85%, and Gradient Boosted Classification achieved 84%. However, the Gradient Boosted Classifier issued a warning that it did not predict one of the classes, likely due to the class imbalance, as most of the dataset corresponds to Purchase Completed = False. Based on accuracy and stability, the Random Forest model was selected as the best performing model. Given the small performance differences, further validation using larger and more balanced datasets would help confirm the robustness and generalizability of these results.

## Features processed for machine learning models

|Feature         | Prerporcessed    |
|----------------|------------------|
|ProductPageTime | normalized and scaled |
|BounceRate      | excluded from dataset |
|ExitRate        | left as is|
|PageValue       | Excluded from dataset|
|SpecialDayProximity| one-hot encoding |
|GeographicRegion | one-hot encoding |
|TrafficSource | one-hot encoding |
|CustomerType  | one-hot encoding  | 
| PurchaseCompleted | left as is |