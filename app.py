#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from data_generator import generate_training_data, save_data
from naive_bayes import ClassCancellationNaiveBayes, train_model

app = Flask(__name__)

# Ensure the templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load or train the model
data_file = 'university_classes_data.csv'
if not os.path.exists(data_file):
    print("Generating training data...")
    data = generate_training_data(num_samples=1000)
    save_data(data, data_file)

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        print("Training the Naive Bayes classifier...")
        model, _, _, _, _ = train_model(data_file)
    return model

def generate_probability_plot(probabilities):
    """
    Generate a base64-encoded image of the prediction probabilities
    
    Parameters:
    -----------
    probabilities : numpy.ndarray
        The probabilities for each class
        
    Returns:
    --------
    str
        Base64-encoded PNG image
    """
    labels = ['کلاس لغو می‌شود', 'کلاس برگزار می‌شود']
    colors = ['#FF5555', '#55DD55']
    
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'  # Use a font that supports Persian
    
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
    plt.title('احتمال برگزاری یا لغو کلاس', fontsize=16)
    plt.ylabel('احتمال', fontsize=14)
    plt.ylim(0, 1.1)
    
    # Highlight the predicted class
    predicted_idx = np.argmax(probabilities[0])
    plt.text(predicted_idx, 0.5, 'پیش‌بینی', 
             ha='center', va='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[predicted_idx], alpha=0.7),
             fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    # Get the trained model
    nb_model = load_model()
    
    # Get form data
    weather = request.form.get('weather')
    temperature = float(request.form.get('temperature'))
    day = request.form.get('day')
    time = request.form.get('time')
    professor_status = request.form.get('professor_status')
    is_holiday = request.form.get('is_holiday') == 'true'
    
    # Create a features dictionary
    features = {
        'Weather': weather,
        'Temperature': temperature,
        'Day': day,
        'Time': time,
        'Professor_Status': professor_status,
        'Is_Holiday': is_holiday,
        'Date': pd.Timestamp.now().strftime('%Y-%m-%d')  # Current date
    }
    
    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([features])
    
    # Make prediction
    probabilities = nb_model.predict_proba(input_df)
    prediction = nb_model.predict(input_df)
    
    # Generate the probability plot
    plot_image = generate_probability_plot(probabilities)
    
    # Prepare the response
    result = {
        "prediction": bool(prediction[0]),
        "prediction_text": "کلاس برگزار می‌شود" if prediction[0] else "کلاس لغو می‌شود",
        "probability_held": float(probabilities[0][1]),
        "probability_cancelled": float(probabilities[0][0]),
        "plot_image": plot_image,
        "features": features
    }
    
    return jsonify(result)

# Create HTML template
@app.route('/create_templates')
def create_templates():
    """Create the HTML templates for the application"""
    # Create base layout
    with open('templates/layout.html', 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>سیستم پیش‌بینی برگزاری کلاس دانشگاه</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css" rel="stylesheet">
    <style>
        body {
            font-family: "Tahoma", "Vazirmatn", sans-serif;
            background-color: #f7f7f7;
        }
        .container {
            max-width: 800px;
        }
        .form-card, .result-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            padding: 20px;
        }
        .card-header {
            border-bottom: 1px solid #eee;
            margin-bottom: 20px;
            padding-bottom: 10px;
        }
        .feature-label {
            font-weight: bold;
            color: #555;
        }
        .probability-bar {
            height: 30px;
            border-radius: 5px;
        }
        .prediction-box {
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
        }
        .prediction-box.cancelled {
            background-color: #ffebee;
            border: 1px solid #ffcdd2;
            color: #c62828;
        }
        .prediction-box.held {
            background-color: #e8f5e9;
            border: 1px solid #c8e6c9;
            color: #2e7d32;
        }
        .prediction-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        h1, h2, h3, h4 {
            color: #333;
        }
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <div class="container py-4">
        <h1 class="text-center mb-4">سیستم پیش‌بینی برگزاری کلاس دانشگاه</h1>
        <div class="row">
            <div class="col-12">
                {% block content %}{% endblock %}
            </div>
        </div>
        <footer class="text-center mt-4 text-muted">
            <p>پیاده‌سازی با استفاده از الگوریتم Naive Bayes | طراحی شده با Flask و Bootstrap</p>
        </footer>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>''')
    
    # Create index page
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''{% extends "layout.html" %}

{% block content %}
<div class="row">
    <div class="col-md-6">
        <div class="form-card">
            <div class="card-header">
                <h3><i class="bi bi-input-cursor"></i> ورود اطلاعات</h3>
            </div>
            <form id="prediction-form">
                <div class="mb-3">
                    <label for="weather" class="form-label feature-label">وضعیت آب و هوا:</label>
                    <select class="form-select" id="weather" name="weather" required>
                        <option value="" selected disabled>انتخاب کنید...</option>
                        <option value="Sunny">آفتابی</option>
                        <option value="Cloudy">ابری</option>
                        <option value="Rainy">بارانی</option>
                        <option value="Snowy">برفی</option>
                        <option value="Stormy">طوفانی</option>
                    </select>
                </div>
                
                <div class="mb-3">
                    <label for="temperature" class="form-label feature-label">دما (درجه سانتیگراد):</label>
                    <input type="number" class="form-control" id="temperature" name="temperature" min="-30" max="50" step="0.1" required>
                </div>
                
                <div class="mb-3">
                    <label for="day" class="form-label feature-label">روز هفته:</label>
                    <select class="form-select" id="day" name="day" required>
                        <option value="" selected disabled>انتخاب کنید...</option>
                        <option value="Monday">دوشنبه</option>
                        <option value="Tuesday">سه‌شنبه</option>
                        <option value="Wednesday">چهارشنبه</option>
                        <option value="Thursday">پنج‌شنبه</option>
                        <option value="Friday">جمعه</option>
                        <option value="Saturday">شنبه</option>
                        <option value="Sunday">یکشنبه</option>
                    </select>
                </div>
                
                <div class="mb-3">
                    <label for="time" class="form-label feature-label">ساعت کلاس:</label>
                    <select class="form-select" id="time" name="time" required>
                        <option value="" selected disabled>انتخاب کنید...</option>
                        {% for hour in range(8, 19) %}
                        <option value="{{ '%02d' % hour }}:00">{{ '%02d' % hour }}:00</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="mb-3">
                    <label for="professor_status" class="form-label feature-label">وضعیت استاد:</label>
                    <select class="form-select" id="professor_status" name="professor_status" required>
                        <option value="" selected disabled>انتخاب کنید...</option>
                        <option value="Available">در دسترس</option>
                        <option value="Sick">بیمار</option>
                        <option value="Conference">در کنفرانس</option>
                        <option value="Personal_Leave">مرخصی شخصی</option>
                    </select>
                </div>
                
                <div class="mb-3">
                    <label class="form-label feature-label">آیا امروز تعطیل رسمی است؟</label>
                    <div class="form-check">
                        <input class="form-check-input" type="radio" name="is_holiday" id="holiday-yes" value="true">
                        <label class="form-check-label" for="holiday-yes">بله</label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="radio" name="is_holiday" id="holiday-no" value="false" checked>
                        <label class="form-check-label" for="holiday-no">خیر</label>
                    </div>
                </div>
                
                <div class="d-grid gap-2">
                    <button type="submit" class="btn btn-primary">پیش‌بینی کن</button>
                </div>
            </form>
        </div>
    </div>
    
    <div class="col-md-6">
        <div id="results-container" style="display:none;">
            <div class="result-card">
                <div class="card-header">
                    <h3><i class="bi bi-graph-up"></i> نتیجه پیش‌بینی</h3>
                </div>
                
                <div id="prediction-result">
                    <!-- Prediction result will be displayed here -->
                </div>
                
                <div class="mt-4">
                    <h4>جزئیات احتمالات:</h4>
                    <div class="row mb-2">
                        <div class="col-6">
                            <span class="feature-label">احتمال برگزاری:</span>
                        </div>
                        <div class="col-6 text-end">
                            <span id="probability-held"></span>
                        </div>
                    </div>
                    <div class="row mb-3">
                        <div class="col-6">
                            <span class="feature-label">احتمال لغو:</span>
                        </div>
                        <div class="col-6 text-end">
                            <span id="probability-cancelled"></span>
                        </div>
                    </div>
                    
                    <div class="text-center mt-4 mb-2">
                        <img id="plot-image" class="img-fluid" src="" alt="نمودار احتمالات">
                    </div>
                </div>
            </div>
            
            <div class="result-card">
                <div class="card-header">
                    <h3><i class="bi bi-info-circle"></i> اطلاعات ورودی</h3>
                </div>
                
                <div id="features-info">
                    <!-- Features info will be displayed here -->
                </div>
            </div>
        </div>
        
        <div id="loading-container" style="display:none;">
            <div class="text-center my-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">در حال بارگذاری...</span>
                </div>
                <p class="mt-2">در حال پردازش اطلاعات...</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
$(document).ready(function() {
    // Dictionary for translating weather conditions
    const weatherTranslations = {
        'Sunny': 'آفتابی',
        'Cloudy': 'ابری',
        'Rainy': 'بارانی',
        'Snowy': 'برفی',
        'Stormy': 'طوفانی'
    };
    
    // Dictionary for translating days
    const dayTranslations = {
        'Monday': 'دوشنبه',
        'Tuesday': 'سه‌شنبه',
        'Wednesday': 'چهارشنبه',
        'Thursday': 'پنج‌شنبه',
        'Friday': 'جمعه',
        'Saturday': 'شنبه',
        'Sunday': 'یکشنبه'
    };
    
    // Dictionary for translating professor status
    const professorTranslations = {
        'Available': 'در دسترس',
        'Sick': 'بیمار',
        'Conference': 'در کنفرانس',
        'Personal_Leave': 'مرخصی شخصی'
    };
    
    // Handle form submission
    $('#prediction-form').on('submit', function(e) {
        e.preventDefault();
        
        // Show loading
        $('#results-container').hide();
        $('#loading-container').show();
        
        // Get form data
        const formData = $(this).serialize();
        
        // Send prediction request
        $.ajax({
            url: '/predict',
            method: 'POST',
            data: formData,
            dataType: 'json',
            success: function(response) {
                // Hide loading
                $('#loading-container').hide();
                
                // Update prediction result
                let resultHTML = '';
                if (response.prediction) {
                    resultHTML = `
                        <div class="prediction-box held">
                            <div class="prediction-icon"><i class="bi bi-check-circle-fill"></i></div>
                            <h2>کلاس برگزار می‌شود</h2>
                            <p>با احتمال ${(response.probability_held * 100).toFixed(1)}%</p>
                        </div>
                    `;
                } else {
                    resultHTML = `
                        <div class="prediction-box cancelled">
                            <div class="prediction-icon"><i class="bi bi-x-circle-fill"></i></div>
                            <h2>کلاس لغو می‌شود</h2>
                            <p>با احتمال ${(response.probability_cancelled * 100).toFixed(1)}%</p>
                        </div>
                    `;
                }
                $('#prediction-result').html(resultHTML);
                
                // Update probabilities
                $('#probability-held').text(`${(response.probability_held * 100).toFixed(1)}%`);
                $('#probability-cancelled').text(`${(response.probability_cancelled * 100).toFixed(1)}%`);
                
                // Update plot image
                $('#plot-image').attr('src', 'data:image/png;base64,' + response.plot_image);
                
                // Update features info
                const features = response.features;
                const featuresHTML = `
                    <div class="row">
                        <div class="col-6 mb-2">
                            <span class="feature-label">وضعیت آب و هوا:</span>
                        </div>
                        <div class="col-6 mb-2 text-end">
                            ${weatherTranslations[features.Weather]}
                        </div>
                        
                        <div class="col-6 mb-2">
                            <span class="feature-label">دما:</span>
                        </div>
                        <div class="col-6 mb-2 text-end">
                            ${features.Temperature} °C
                        </div>
                        
                        <div class="col-6 mb-2">
                            <span class="feature-label">روز هفته:</span>
                        </div>
                        <div class="col-6 mb-2 text-end">
                            ${dayTranslations[features.Day]}
                        </div>
                        
                        <div class="col-6 mb-2">
                            <span class="feature-label">ساعت کلاس:</span>
                        </div>
                        <div class="col-6 mb-2 text-end">
                            ${features.Time}
                        </div>
                        
                        <div class="col-6 mb-2">
                            <span class="feature-label">وضعیت استاد:</span>
                        </div>
                        <div class="col-6 mb-2 text-end">
                            ${professorTranslations[features.Professor_Status]}
                        </div>
                        
                        <div class="col-6 mb-2">
                            <span class="feature-label">تعطیل رسمی:</span>
                        </div>
                        <div class="col-6 mb-2 text-end">
                            ${features.Is_Holiday ? 'بله' : 'خیر'}
                        </div>
                    </div>
                `;
                $('#features-info').html(featuresHTML);
                
                // Show results
                $('#results-container').show();
            },
            error: function(error) {
                console.error('Error:', error);
                $('#loading-container').hide();
                alert('خطا در پردازش درخواست. لطفا دوباره تلاش کنید.');
            }
        });
    });
});
</script>
{% endblock %}''')
    
    return "Templates created successfully!"

if __name__ == '__main__':
    # Create templates on startup if they don't exist
    if not os.path.exists('templates/index.html'):
        create_templates()
    
    # Initialize model
    load_model()
    
    # Run the app
    app.run(debug=True, port=5000) 