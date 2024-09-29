from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class MLModel:
    def __init__(self, model_path='ml/model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the ML model from a file."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def predict(self, features):
        """Make predictions using the loaded ML model."""
        if self.model is not None:
            prediction = self.model.predict([features])
            return prediction[0]
        else:
            raise ValueError("Model is not loaded.")

    def save_model(self):
        """Save the model to a file."""
        joblib.dump(self.model, self.model_path)

    def train_model(self, X_train, y_train):
        """Train the model with the given training data."""
        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)
        self.save_model()

