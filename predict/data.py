import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Load dataset
data = pd.read_csv("students_dropout_and_success_transformation.csv", sep=',', decimal=".")
x = data.drop("target", axis=1)  # Fitur
y = data["target"]  # Label

# Encode target if it's categorical
y = y.map({'Dropout': 0, 'Graduate': 1})  # Adjust mapping if necessary

# Ensure all features are numeric using OneHotEncoder
encoder = OneHotEncoder()
x_encoded = encoder.fit_transform(x)
x = pd.DataFrame(x_encoded.toarray(), columns=encoder.get_feature_names_out())

# Simpan encoder
joblib.dump(encoder, "encoder.pkl")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Handle missing values
x_train.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)

# Balance the data using SMOTE
smote = SMOTE(random_state=42)
print("Shape before SMOTE:", x_train.shape)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
print("Shape after SMOTE:", x_resampled.shape)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_resampled, y_resampled)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy: {accuracy:.2f}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_resampled, y_resampled)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
best_y_pred = best_model.predict(x_test)
print("Best Model Accuracy:", accuracy_score(y_test, best_y_pred))

# Save the best model and encoder
joblib.dump(best_model, "dropout_model.pkl")
joblib.dump(encoder, "encoder.pkl")

# Flask API
app = Flask(__name__)
CORS(app)

# Load the model and encoder
model = joblib.load("dropout_model.pkl")
encoder = joblib.load("encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        required_features = [
            'marital_status', 'daytime_attendance', 'previous_qualification', 
            'previous_qualification_grade', 'mothers_qualification', 
            'fathers_qualification', 'mothers_occupation', 'fathers_occupation', 
            'admission_grade', 'displaced', 'educational_special_needs', 
            'debtor', 'tuition_fees_update', 'gender', 'scholarship_holder', 
            'age', 'international', 'sem_one_grade', 'sem_two_grade'
        ]
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

       # Konversi data input ke DataFrame
        input_features_df = pd.DataFrame([data], columns=required_features)
        print("Input Features (Prediction):", input_features_df.columns.tolist())
        print("Expected Features (Training):", encoder.get_feature_names_out())
        
        # Terapkan encoding yang sama
        input_features_encoded = encoder.transform(input_features_df)
        input_features_array = input_features_encoded.toarray()

        
        

        # Predict
        prediction = model.predict(input_features_array)
        result = "Graduate" if prediction[0] == 1 else "Dropout"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        


if __name__ == '__main__':
    app.run(debug=True)
