import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load the test dataset
test_data = pd.read_excel("traffic_anomaly_data_val.xlsx")

# Load label encoder and scaler
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Encode and normalize the test data
test_data['Anomaly Class'] = label_encoder.transform(test_data['Anomaly Class'])
features = ['Video Start', 'Video End', 'Anomaly Start', 'Anomaly End', 'Number of Frames']
target = 'Anomaly Class'

test_data[features] = scaler.transform(test_data[features])
X_test, y_test = test_data[features], test_data[target]

# Function to evaluate a model and generate confusion matrix
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    performance = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Confusion Matrix': conf_matrix
    }
    return performance

# Load and evaluate traditional models
results = []
models = ["Logistic_Regression", "Decision_Tree", "Random_Forest", "Gaussian_Naive_Bayes", "SVM"]
for model_name in models:
    model = joblib.load(f"{model_name}.pkl")
    performance = evaluate_model(model_name, model, X_test, y_test)
    results.append(performance)

# Load and evaluate the ANN model
ann_model = load_model("ANN_model.h5")
y_pred_ann = ann_model.predict(X_test).argmax(axis=1)
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann)
ann_performance = {
    'Model': 'ANN',
    'Accuracy': accuracy_score(y_test, y_pred_ann),
    'Precision': precision_score(y_test, y_pred_ann, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, y_pred_ann, average='weighted', zero_division=0),
    'F1 Score': f1_score(y_test, y_pred_ann, average='weighted', zero_division=0),
    'Confusion Matrix': conf_matrix_ann
}
results.append(ann_performance)

# Convert results to DataFrame and save to Excel
results_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'Confusion Matrix'} for result in results])
results_df.to_excel("model_results.xlsx", index=False)
print("Results saved to 'model_results.xlsx'")

# Plot metrics for all models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=results_df, x='Model', y=metric, palette='viridis',width=0.5)
    plt.title(f'Model Comparison: {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png")
    plt.show()

# Plot confusion matrices
for result in results:
    conf_matrix = result['Confusion Matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix: {result['Model']}")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{result['Model']}.png")
    plt.show()
