This project aims to classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous) using data preprocessing, feature analysis, and machine learning models.
It is based on the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains various features computed from digitized images of breast masses.

->Technologies Used

-Python 3.12.7

-Pandas, NumPy — data handling & preprocessing

-Matplotlib, Seaborn — visualization

-Scikit-learn — model building and evaluation

-Jupyter Notebook — development environment

->Project Workflow
1. Data Preprocessing

Dropped irrelevant columns (Unnamed: 32)

Encoded the target column:

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


Checked for missing data using:

data.isnull().sum()

2. Splitting the Dataset

Data was split into:

X_train (80%) — training features

X_test (20%) — testing features

y_train, y_test — corresponding labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3. Model Building

A Pipeline was created for consistent preprocessing and training:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
pipeline.fit(X_train, y_train)

4. Feature Importance

After training, the most influential features were extracted:

model = pipeline.named_steps['clf']
coefficients = pd.Series(model.coef_[0], index=X.columns)
coefficients.abs().sort_values(ascending=False).head(10)

5. Model Evaluation

Evaluated using accuracy, confusion matrix, and classification report:

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

--Results Summary
Metric	   Score
Accuracy : ~0.98
Precision: (Malignant)	High
Recall : (Benign)	High
F1-Score : Excellent

--> The model achieved high accuracy and balanced performance across both classes, indicating strong generalization.
