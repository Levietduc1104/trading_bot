import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """Preprocess the raw data and extract features."""
    features = data[['Close', 'High', 'Low', 'Volume']]
    return features

def scale_data(features):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

