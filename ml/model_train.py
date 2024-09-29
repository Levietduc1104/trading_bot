from sklearn.model_selection import train_test_split
from ml.ml_model import MLModel
from ml.data_preprocessing import preprocess_data, scale_data

def train_model(data, labels):
    """Train the ML model using the data and labels."""
    features = preprocess_data(data)
    scaled_features = scale_data(features)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

    # Initialize and train the model
    ml_model = MLModel()
    ml_model.train_model(X_train, y_train)

    # Evaluate the model
    accuracy = ml_model.model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return ml_model

