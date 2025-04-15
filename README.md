# University Class Prediction System

This project implements a Naive Bayes classifier to predict whether university classes will be held based on various factors such as weather conditions, temperature, day and time information, and administrative factors.

## Features

- **Naive Bayes Classifier**: Custom implementation with detailed probability calculations
- **Data Generator**: Creates synthetic training data with realistic patterns
- **Console Application**: User-friendly interface with color-coded results and probability visualization
- **Web Interface**: Flask-based web application with bilingual support (English/Persian)
- **Visualization**: Graphical representation of prediction probabilities

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Project Structure

- `naive_bayes.py` - Naive Bayes classifier implementation
- `data_generator.py` - Training data generation
- `console_app.py` - Command-line interface
- `app.py` - Flask web application
- `requirements.txt` - Dependencies list

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Console Application

Run the command-line interface:

```
python console_app.py
```

Follow the prompts to input the required information:
- Weather conditions
- Temperature
- Day of the week
- Class time
- Professor status
- Holiday status

The application will display the prediction result with probability visualization.

### Web Application

Start the Flask web server:

```
python app.py
```

Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

The web interface provides:
- A user-friendly form to input prediction factors
- Real-time prediction results with visualizations
- Bilingual support (English/Persian)

## How It Works

The system uses a Naive Bayes classifier, which calculates the probability of classes being held based on the conditional probabilities of each feature given the class.

### Training Data

The training data includes:
- Weather conditions (Sunny, Cloudy, Rainy, Snowy, Stormy)
- Temperature values
- Day of the week
- Time of class
- Professor availability status
- Holiday information

### Model Details

The classifier:
1. Uses Laplace smoothing to handle zero probabilities
2. Handles both categorical and numerical features
3. Calculates probabilities using the Naive Bayes assumption
4. Provides prediction probabilities for both outcomes

## License

MIT

## Acknowledgments

- This project was created as a demonstration of Naive Bayes classification for educational purposes
- Uses NumPy, Pandas, Matplotlib, and Flask frameworks 