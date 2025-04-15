#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_training_data(num_samples=1000, seed=42):
    """
    Generate synthetic training data for the university class cancellation prediction model.
    
    Parameters:
    -----------
    num_samples : int
        Number of data samples to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Generated dataset with features and target variable
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Define possible values for categorical features
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Stormy']
    weather_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # Probability weights
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    professor_availability = ['Available', 'Sick', 'Conference', 'Personal_Leave']
    
    # Generate a base date and then create dates spanning a term
    base_date = datetime(2023, 9, 1)  # Start of academic year
    
    # Create empty lists for each feature
    dates = []
    weathers = []
    temperatures = []
    days = []
    times = []
    professor_statuses = []
    is_holiday = []
    
    # Target variable - class held or not
    class_held = []
    
    for _ in range(num_samples):
        # Generate date (within a 120-day period)
        random_days = np.random.randint(0, 120)
        date = base_date + timedelta(days=random_days)
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Day of week
        day = days_of_week[date.weekday()]
        days.append(day)
        
        # Time (8:00 to 18:00, hourly intervals)
        hour = np.random.randint(8, 19)
        time_str = f"{hour:02d}:00"
        times.append(time_str)
        
        # Weather
        weather = random.choices(weather_conditions, weights=weather_weights)[0]
        weathers.append(weather)
        
        # Temperature (in Celsius)
        if weather == 'Sunny':
            temp = np.random.normal(25, 5)
        elif weather == 'Cloudy':
            temp = np.random.normal(20, 5)
        elif weather == 'Rainy':
            temp = np.random.normal(15, 5)
        elif weather == 'Snowy':
            temp = np.random.normal(-2, 3)
        else:  # Stormy
            temp = np.random.normal(18, 7)
        temperatures.append(round(temp, 1))
        
        # Professor availability
        prof_status = np.random.choice(professor_availability, p=[0.85, 0.05, 0.05, 0.05])
        professor_statuses.append(prof_status)
        
        # Is it a holiday?
        holiday = np.random.choice([True, False], p=[0.1, 0.9])
        is_holiday.append(holiday)
        
        # Determine if class is held based on rules
        held = True
        
        # Weekend rule - In Iranian academic schedule, Friday is the weekend
        if day == 'Friday':
            held = False
        
        # Thursday rule - Lower probability of classes on Thursday
        if day == 'Thursday' and np.random.random() < 0.3:  # 30% chance of no class on Thursday
            held = False
        
        # Weather rule
        if weather == 'Stormy' or (weather == 'Snowy' and temp < -5):
            held = False
        
        # Professor rule
        if prof_status != 'Available':
            held = False
            
        # Holiday rule
        if holiday:
            held = False
            
        # Time rule
        if hour < 8 or hour > 18:
            held = False
            
        # Add some randomness
        if np.random.random() < 0.05:  # 5% chance to flip the outcome
            held = not held
            
        class_held.append(held)
    
    # Create the dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Day': days,
        'Time': times,
        'Weather': weathers,
        'Temperature': temperatures,
        'Professor_Status': professor_statuses,
        'Is_Holiday': is_holiday,
        'Class_Held': class_held
    })
    
    return df

def save_data(df, filename='university_classes_data.csv'):
    """
    Save the generated data to a CSV file
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The data to save
    filename : str
        The name of the output file
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename} successfully!")
    
if __name__ == "__main__":
    # Generate the data
    data = generate_training_data(num_samples=1000)
    
    # Display sample and stats
    print("Generated dataset sample:")
    print(data.head())
    
    print("\nDataset statistics:")
    print(f"Shape: {data.shape}")
    print(f"Classes held: {data['Class_Held'].sum()} ({data['Class_Held'].mean():.1%})")
    print(f"Classes cancelled: {(~data['Class_Held']).sum()} ({(~data['Class_Held']).mean():.1%})")
    
    # Save to CSV
    save_data(data) 