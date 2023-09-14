import pandas as pd
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from feat import Detector
from feat.data import Fex
import pandas as pd
from feat.utils.io import get_test_data_path
from feat.plotting import imshow

app = Flask(__name__)

# Define the folder where uploaded photos will be stored
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to categorize the value
def categorize(value, categories):
    for (lower, upper), label in categories.items():
        if lower <= value <= upper:
            return label
    return 'Undefined'  # Handle values outside defined ranges

# Function to get photo dimensions
def get_photo_dimensions(photo_path):
    detector = Detector(emotion_model = 'resmasknet')
    #img = Image.open(photo_path)
    single_face_prediction = detector.detect_image(photo_path)
    scoress = (single_face_prediction[['anger','disgust','fear','happiness','sadness','surprise','neutral']])
    dff = pd.DataFrame(scoress)

    anger_value = dff.loc[0, 'anger']
    disgust_value = dff.loc[0, 'disgust']
    fear_value = dff.loc[0, 'fear']
    happiness_value = dff.loc[0, 'happiness']
    sadness_value = dff.loc[0, 'sadness']
    surprise_value = dff.loc[0, 'surprise'] 
    neutral_value = dff.loc[0, 'neutral']

    weights = {'anger': 0.4,   # Weighted higher
           'disgust': 0.4, # Weighted higher
           'fear': 0.3,    # Weighted higher
           'happiness': 0.01,
           'sadness': 0.1,
           'surprise': 0.15,
           'neutral': 0.05}
    
    overall_score = sum(dff[emotion] * weights[emotion] for emotion in dff.columns)

    categories = {
        (0.0, 0.1): 'You are a newbie',
        (0.1, 0.2): 'You are picking up some glam rock',
        (0.2, 0.3): 'You are becoming a hard-ass metal fan',
        (0.3, 0.4): 'You are an M*D*F Slayer',
        (0.4, 1.0): 'You are an all-time hard-core metal head'
    }

    overall_score2 = overall_score.iloc[0]
    metal_category = categorize(overall_score2, categories)

    #return anger_value, disgust_value, fear_value, happiness_value, sadness_value, surprise_value, neutral_value
    #return overall_score
    return metal_category

@app.route('/', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return "No selected file"

        # If the file is valid, save it to the UPLOAD_FOLDER
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Get photo score
            #anger_value, disgust_value, fear_value, happiness_value, sadness_value, surprise_value, neutral_value = get_photo_dimensions(filename)
            #overall_score = get_photo_dimensions(filename)
            metal_category = get_photo_dimensions(filename)


            # Create a Pandas DataFrame with photo information
            #photo_info = pd.DataFrame({'Attribute': ['Filename', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
            #                           'Value': [file.filename, anger_value, disgust_value, fear_value, happiness_value, sadness_value, surprise_value, neutral_value]})
            
            photo_info = pd.DataFrame({
                                       '': [metal_category]})

            return render_template('result.html', photo_info=photo_info.to_html(classes='table table-striped'), filename=file.filename)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
