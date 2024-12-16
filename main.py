from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN  # Optional alternative
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the dataset
file_path = r'ML_CW\bank-full.csv'
df = pd.read_csv(file_path, sep=";")
print(df.head())

# Splitting features and target
X = df.drop(columns=['y'])
y = df['y']
y = LabelEncoder().fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers in ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Preprocess the entire dataset
X_preprocessed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Combine SMOTE and Random Undersampling
smote = SMOTE(random_state=42)
undersampler = RandomUnderSampler(random_state=42)

# Balance the training data
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_res, y_train_res = undersampler.fit_resample(X_train_res, y_train_res)

# Random Forest Implementation
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {rf_accuracy}")
print("Random Forest Classification Report:")
print(rf_report)

# Neural Network Implementation
nn_model = MLPClassifier(random_state=42, max_iter=500)
nn_model.fit(X_train_res, y_train_res)

# Predict and evaluate Neural Network
y_pred_nn = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_report = classification_report(y_test, y_pred_nn)

print(f"Neural Network Accuracy: {nn_accuracy}")
print("Neural Network Classification Report:")
print(nn_report)
