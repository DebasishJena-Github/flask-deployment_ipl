import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained model
import os

# Get the absolute path to the directory containing the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the absolute path to the pickle file
model_file_path = os.path.join(script_dir, 'first-innings-score-lr-model.pkl')

# Open the file using the absolute path
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract the form data from the JSON request
    batting_team = data.get('batting-team')
    bowling_team = data.get('bowling-team')
    overs = float(data.get('overs'))
    runs = int(data.get('runs'))
    wickets = int(data.get('wickets'))
    runs_in_prev_5 = int(data.get('runs_in_prev_5'))
    wickets_in_prev_5 = int(data.get('wickets_in_prev_5'))

    # Preprocess the input data
    # Assuming you have a function to preprocess the input
    input_data = preprocess_input(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5)
    
    # Predict using the model
    prediction = model.predict(input_data)
    lower_limit = int(prediction[0] - 10)
    upper_limit = int(prediction[0] + 10)

    return jsonify({'lower_limit': lower_limit, 'upper_limit': upper_limit})

def preprocess_input(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
    # Convert team names to one-hot encoding or label encoding based on how your model was trained
    teams = ['Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Kings XI Punjab', 
             'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']
    
    batting_team_encoded = [1 if team == batting_team else 0 for team in teams]
    bowling_team_encoded = [1 if team == bowling_team else 0 for team in teams]

    # Create the input array
    input_array = batting_team_encoded + bowling_team_encoded + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
    
    return np.array(input_array).reshape(1, -1)

# if __name__ == '__main__':
#     app.run(debug=True)
