#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import generate_training_data, save_data
from naive_bayes import ClassCancellationNaiveBayes, train_model
import os

def generate_probability_visualization(probabilities, save_path=None):
    """
    Generate a visualization of the prediction probabilities
    
    Parameters:
    -----------
    probabilities : numpy.ndarray
        The probabilities for each class
    save_path : str or None
        If provided, save the visualization to this path
    """
    labels = ['Class Cancelled', 'Class Held']
    colors = ['#FF5555', '#55DD55']
    
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    
    # Create a bar chart
    bars = plt.bar(labels, probabilities[0], color=colors)
    
    # Add percentage labels on top of the bars
    for bar, prob in zip(bars, probabilities[0]):
        plt.text(bar.get_x() + bar.get_width()/2., 
                 bar.get_height() + 0.01, 
                 f'{prob:.1%}', 
                 ha='center', va='bottom', fontsize=12)
    
    # Add a horizontal line at 50%
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set the title and labels
    plt.title('Class Cancellation Prediction Probabilities', fontsize=16)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Highlight the predicted class
    predicted_idx = np.argmax(probabilities[0])
    plt.text(predicted_idx, 0.5, 'PREDICTED', 
             ha='center', va='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[predicted_idx], alpha=0.7),
             fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def print_prediction_table(probabilities, features):
    """
    Print a formatted table of the prediction results
    
    Parameters:
    -----------
    probabilities : numpy.ndarray
        The probabilities for each class
    features : dict
        The input features
    """
    # Print header
    print("\n" + "="*60)
    print(" "*20 + "PREDICTION RESULTS")
    print("="*60)
    
    # Print input features
    print("\nINPUT FEATURES:")
    print("-"*60)
    for key, value in features.items():
        print(f"{key:20}: {value}")
    
    # Print prediction
    print("\nPREDICTION:")
    print("-"*60)
    
    # Determine the prediction
    prediction = "Class will be HELD" if probabilities[0][1] >= 0.5 else "Class will be CANCELLED"
    confidence = max(probabilities[0]) * 100
    
    # Print with color coding
    if "HELD" in prediction:
        color_code = "\033[92m"  # Green
    else:
        color_code = "\033[91m"  # Red
    
    reset_code = "\033[0m"
    print(f"Result:             {color_code}{prediction}{reset_code}")
    print(f"Confidence:         {confidence:.1f}%")
    print(f"Probability held:   {probabilities[0][1]:.1%}")
    print(f"Probability cancel: {probabilities[0][0]:.1%}")
    
    print("="*60 + "\n")

def get_user_input():
    """
    Collect user input for making a prediction
    
    Returns:
    --------
    dict
        A dictionary of user input features
    """
    print("\n" + "="*60)
    print(" "*15 + "CLASS CANCELLATION PREDICTOR")
    print("="*60 + "\n")
    
    print("Please provide the following information:")
    
    # Weather input
    print("\nWeather Conditions:")
    weather_options = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
    for i, option in enumerate(weather_options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            weather_choice = int(input("\nEnter choice (1-5): "))
            if 1 <= weather_choice <= 5:
                weather = weather_options[weather_choice-1]
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Temperature input
    while True:
        try:
            temperature = float(input("\nEnter temperature (°C): "))
            if -30 <= temperature <= 50:
                break
            else:
                print("Please enter a reasonable temperature between -30°C and 50°C.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Day input
    print("\nDay of the week:")
    day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, option in enumerate(day_options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            day_choice = int(input("\nEnter choice (1-7): "))
            if 1 <= day_choice <= 7:
                day = day_options[day_choice-1]
                break
            else:
                print("Please enter a number between 1 and 7.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Time input
    while True:
        time_input = input("\nEnter class time (HH:00, 24-hour format): ")
        if time_input.endswith(":00") and time_input[:2].isdigit():
            hour = int(time_input[:2])
            if 0 <= hour <= 23:
                time = time_input
                break
        print("Please enter a valid time in HH:00 format (e.g., 09:00, 14:00).")
    
    # Professor status
    print("\nProfessor Status:")
    prof_options = ['Available', 'Sick', 'Conference', 'Personal_Leave']
    for i, option in enumerate(prof_options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            prof_choice = int(input("\nEnter choice (1-4): "))
            if 1 <= prof_choice <= 4:
                prof_status = prof_options[prof_choice-1]
                break
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Holiday input
    while True:
        holiday_input = input("\nIs today a holiday? (y/n): ").lower()
        if holiday_input in ['y', 'n']:
            is_holiday = holiday_input == 'y'
            break
        print("Please enter 'y' for yes or 'n' for no.")
    
    # Create a features dictionary
    features = {
        'Weather': weather,
        'Temperature': temperature,
        'Day': day,
        'Time': time,
        'Professor_Status': prof_status,
        'Is_Holiday': is_holiday,
        'Date': pd.Timestamp.now().strftime('%Y-%m-%d')  # Current date
    }
    
    return features

def main():
    """
    Main function to run the console application
    """
    # Check if data file exists, otherwise generate it
    data_file = 'university_classes_data.csv'
    if not os.path.exists(data_file):
        print("Generating training data...")
        data = generate_training_data(num_samples=1000)
        save_data(data, data_file)
    
    # Train the model
    print("\nTraining the Naive Bayes classifier...")
    model, _, _, _, _ = train_model(data_file)
    
    while True:
        # Get user input
        features = get_user_input()
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([features])
        
        # Make prediction
        probabilities = model.predict_proba(input_df)
        prediction = model.predict(input_df)
        
        # Display results
        print_prediction_table(probabilities, features)
        
        # Generate visualization
        generate_probability_visualization(probabilities)
        
        # Ask if the user wants to make another prediction
        while True:
            another = input("Would you like to make another prediction? (y/n): ").lower()
            if another in ['y', 'n']:
                break
            print("Please enter 'y' for yes or 'n' for no.")
        
        if another == 'n':
            print("\nThank you for using the Class Cancellation Predictor. Goodbye!")
            break

if __name__ == "__main__":
    main() 