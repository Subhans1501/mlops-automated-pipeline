
import os
import pytest
from train import load_data

def test_data_loading():
    """Test if the dataset loads correctly and is not empty."""
    # Point to the actual dataset path relative to this test file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'phishing_email.csv')
    
    # Run the function
    X, y = load_data(data_path)
    
    # Assertions
    assert X is not None, "Features should not be None"
    assert len(X) == len(y), "Features and labels must match"
    assert len(X) > 0, "Dataset should not be empty"