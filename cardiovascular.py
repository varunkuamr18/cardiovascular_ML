import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer

# Load data
df = pd.read_csv('cardio_train_cleaned.csv')

# 1. Cleaning: Remove nulls and outliers
df.dropna(inplace=True)
df = df[(df['ap_hi'] < 250) & (df['ap_hi'] > 60)]
df = df[(df['ap_lo'] < 150) & (df['ap_lo'] > 40)]

# 2. Preprocessing
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

# 3. Scaling (Convert to 0 and 1 range/standardized)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. GridSearchCV Comparison
model_params = {
    'RF': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    },
    'LR': {
        'model': LogisticRegression(),
        'params': {'C': [0.1, 1, 10]}
    }
}

f2_scorer = make_scorer(fbeta_score, beta=2)
best_model = None
highest_acc = 0

for name, mp in model_params.items():
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    if grid.best_score_ > highest_acc:
        highest_acc = grid.best_score_
        best_model = grid.best_estimator_

# 5. Save the "Efficient" model and scaler
joblib.dump(best_model, 'cardio_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully!")
