# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib  # To save models

# Load the training dataset
train_data = pd.read_excel("traffic_anomaly_data.xlsx")

# Encode the target variable (Anomaly Class)
label_encoder = LabelEncoder()
train_data['Anomaly Class'] = label_encoder.fit_transform(train_data['Anomaly Class'])

# Select features and target
features = ['Video Start', 'Video End', 'Anomaly Start', 'Anomaly End', 'Number of Frames']
target = 'Anomaly Class'

# Normalize features
scaler = StandardScaler()
train_data[features] = scaler.fit_transform(train_data[features])

X_train, y_train = train_data[features], train_data[target]

# Save label encoder and scaler
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Define and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")
    print(f"Trained and saved: {name}")

# Define and train the ANN model
ann_model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and train the ANN model
ann_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# Save the ANN model
ann_model.save("ANN_model.h5")
print("Trained and saved: ANN")
