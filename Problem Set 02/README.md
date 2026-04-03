# Problem Set [02] - [Bank Term Deposit Prediction]

## 1. Approach
The core objective is a Binary Classification task. Since the bank wants to understand the probability of a "Yes" or "No" outcome, Logistic Regression was chosen for its interpretability and efficiency in handling linear relationships between features.

The project follows a standard Data Science Lifecycle:

* Data Acquisition: Loading the dataset from a structured format (CSV).

* Exploratory Data Analysis (EDA): Identifying categorical vs. numerical features and checking for class imbalance.

* Pipeline Construction: Building a repeatable flow for data cleaning and model training.


## 2. Methodology
# Data Preprocessing
To ensure the Logistic Regression model performs optimally, the following steps were taken:

* Target Encoding: The target variable y was mapped from string labels (yes, no) to integers (1, 0).

* One-Hot Encoding: Categorical features (such as job, marital, and education) were converted into dummy variables using pd.get_dummies. This allows the mathematical model to process non-numeric data.

* Feature Scaling: We applied StandardScaler to the feature set. This shifts the data so it has a mean of 0 and a standard deviation of 1, which is critical for Logistic Regression to converge quickly.

* Handling Class Imbalance: The dataset is naturally skewed toward "No" responses. We utilized the class_weight='balanced' parameter to ensure the model does not ignore the minority class (actual subscribers).

# Model Selection
Algorithm: Logistic Regression.

* Optimization: We increased max_iter to 1000 to ensure the solver reaches the optimal weights for the 17+ attributes.

* Validation: The data was split into 80% Training and 20% Testing sets to evaluate the model on unseen data.


## 3. Findings
The model’s performance is evaluated using more than just accuracy, as accuracy can be misleading in imbalanced datasets.

# Metrics Explained

* Confusion Matrix: A table used to describe the performance of the classification model (True Positives, True Negatives, False Positives, False Negatives).

* Precision: The ability of the classifier not to label a negative sample as positive.

* Recall (Sensitivity): The ability of the classifier to find all the positive samples.

* F1-Score: The weighted average of Precision and Recall.
