#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ClassCancellationNaiveBayes:
    """
    A Naive Bayes classifier for predicting whether university classes will be held.
    
    This implementation calculates probabilities directly rather than using sklearn's
    implementation to provide more transparency and educational value.
    """
    
    def __init__(self, laplace_smoothing=1.0):
        """
        Initialize the Naive Bayes classifier
        
        Parameters:
        -----------
        laplace_smoothing : float
            Smoothing parameter to handle zero probabilities (also called alpha)
        """
        self.laplace_smoothing = laplace_smoothing
        self.class_probabilities = {}  # P(y)
        self.feature_probabilities = {}  # P(x_i | y)
        self.classes = None
        self.encoders = {}
        self.categorical_features = ['Weather', 'Day', 'Time', 'Professor_Status', 'Is_Holiday']
        self.numerical_features = ['Temperature']
        
        # For numerical features, we'll store means and standard deviations
        self.numerical_params = {}
        
    def _preprocess_data(self, X):
        """
        Preprocess the data by encoding categorical variables
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature data to preprocess
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed data with encoded categorical variables
        """
        X_processed = X.copy()
        
        # Encode categorical features if not in training mode (where encoders already exist)
        for feature in self.categorical_features:
            if feature in X.columns:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    self.encoders[feature].fit(X[feature])
                
                X_processed[feature] = self.encoders[feature].transform(X[feature])
        
        return X_processed
    
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature data for training
        y : pandas.Series
            The target variable (class_held)
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Preprocess the data
        X_processed = self._preprocess_data(X)
        
        # Get unique classes
        self.classes = np.unique(y)
        
        # Calculate class probabilities P(y)
        for c in self.classes:
            self.class_probabilities[c] = (np.sum(y == c) + self.laplace_smoothing) / (len(y) + len(self.classes) * self.laplace_smoothing)
        
        # Calculate feature probabilities for each class P(x_i | y)
        for c in self.classes:
            self.feature_probabilities[c] = {}
            
            # Get indices for this class
            indices = (y == c)
            
            # Calculate probabilities for categorical features
            for feature in self.categorical_features:
                if feature in X.columns:
                    # Get unique values for this feature
                    unique_values = np.unique(X_processed[feature])
                    self.feature_probabilities[c][feature] = {}
                    
                    # Calculate P(feature=value | class=c) for each value
                    for value in unique_values:
                        count = np.sum((X_processed[feature] == value) & indices)
                        total = np.sum(indices)
                        
                        # Apply Laplace smoothing
                        prob = (count + self.laplace_smoothing) / (total + len(unique_values) * self.laplace_smoothing)
                        self.feature_probabilities[c][feature][value] = prob
            
            # Calculate parameters for numerical features (Gaussian assumption)
            self.numerical_params[c] = {}
            for feature in self.numerical_features:
                if feature in X.columns:
                    values = X_processed.loc[indices, feature].values
                    # Store mean and standard deviation
                    self.numerical_params[c][feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values) + 1e-9  # Add small constant to avoid division by zero
                    }
        
        return self
    
    def _calculate_feature_probability(self, feature, value, c):
        """
        Calculate the probability of a feature having a specific value given the class
        
        Parameters:
        -----------
        feature : str
            The feature name
        value : object
            The feature value
        c : object
            The class
            
        Returns:
        --------
        float
            The probability P(feature=value | class=c)
        """
        if feature in self.numerical_features:
            # For numerical features, use Gaussian probability density
            mean = self.numerical_params[c][feature]['mean']
            std = self.numerical_params[c][feature]['std']
            
            # Calculate Gaussian probability density
            return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((value - mean) / std) ** 2)
        else:
            # For categorical features, look up the probability
            if value in self.feature_probabilities[c][feature]:
                return self.feature_probabilities[c][feature][value]
            else:
                # If value not seen during training, return Laplace smoothing value
                unique_values = len(self.feature_probabilities[c][feature])
                return self.laplace_smoothing / (sum(self.feature_probabilities[c][feature].values()) + unique_values * self.laplace_smoothing)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature data to predict
            
        Returns:
        --------
        numpy.ndarray
            Predicted class probabilities for each sample
        """
        X_processed = self._preprocess_data(X)
        num_samples = len(X_processed)
        
        # Initialize probabilities array
        probas = np.zeros((num_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Start with the prior probability for this class
            log_probs = np.ones(num_samples) * np.log(self.class_probabilities[c])
            
            # Add log probabilities for each feature for each sample
            for feature in X_processed.columns:
                if feature in self.categorical_features or feature in self.numerical_features:
                    # Skip features not in training data
                    if feature in self.categorical_features and feature not in self.feature_probabilities[c]:
                        continue
                    
                    # Process each sample
                    for j in range(num_samples):
                        # Get the value for this feature
                        value = X_processed.iloc[j][feature]
                        
                        # Calculate probability and add log probability
                        prob = self._calculate_feature_probability(feature, value, c)
                        
                        # Avoid log(0) by adding a small constant
                        log_probs[j] += np.log(prob + 1e-10)
            
            # Save the log probabilities
            probas[:, i] = log_probs
        
        # Convert log probabilities back to probabilities
        probas = np.exp(probas)
        row_sums = probas.sum(axis=1)
        probas = probas / row_sums[:, np.newaxis]
        
        return probas
    
    def predict(self, X):
        """
        Predict classes for the input samples
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature data to predict
            
        Returns:
        --------
        numpy.ndarray
            Predicted class for each sample
        """
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]
    
    def score(self, X, y):
        """
        Calculate the accuracy of the model
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The feature data
        y : pandas.Series
            The true target values
            
        Returns:
        --------
        float
            The accuracy of the model
        """
        return np.mean(self.predict(X) == y)


def train_model(data_file='university_classes_data.csv', test_size=0.2, random_state=42):
    """
    Train the Naive Bayes model using the provided data file
    
    Parameters:
    -----------
    data_file : str
        The path to the CSV data file
    test_size : float
        The proportion of the data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    # Load the data
    data = pd.read_csv(data_file)
    
    # Split features and target
    X = data.drop(columns=['Class_Held'])
    y = data['Class_Held']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train the model
    model = ClassCancellationNaiveBayes()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Train the model
    model, X_train, X_test, y_train, y_test = train_model()
    
    # Make a sample prediction
    sample = X_test.iloc[0].to_frame().T
    
    print("\nSample prediction:")
    print(f"Features: {sample.to_dict('records')[0]}")
    
    probas = model.predict_proba(sample)
    prediction = model.predict(sample)
    
    print(f"Probability of class held: {probas[0][1]:.4f}")
    print(f"Probability of class cancelled: {probas[0][0]:.4f}")
    print(f"Prediction: {'Class will be held' if prediction[0] else 'Class will be cancelled'}") 