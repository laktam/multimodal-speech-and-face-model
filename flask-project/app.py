from flask import Flask
from model import predict
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)


# Define upload folders
UPLOAD_FOLDER_IMAGES = 'uploads/images'
UPLOAD_FOLDER_VOICES = 'uploads/voices'

# Ensure upload folders exist
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_VOICES, exist_ok=True)

app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_VOICES'] = UPLOAD_FOLDER_VOICES

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Check if the POST request has the file part
    image_file = request.files.get('image')
    voice_file = request.files.get('voice')

    result = handle_file_uploads(image_file, voice_file, app.config['UPLOAD_FOLDER_IMAGES'], app.config['UPLOAD_FOLDER_VOICES'])

    if isinstance(result, tuple):  # File uploads successful
        image_path, voice_path = result
    
    prediction = predict(image_path, voice_path)
    return jsonify({'result': prediction}), 200


def handle_file_uploads(image_file, voice_file, upload_folder_images, upload_folder_voices):
    # Check if the POST request has the file part
    if image_file is None or voice_file is None:
        return jsonify({'error': 'No file part'}), 400

    # If user does not select file, browser also submit an empty part without filename
    if image_file.filename == '' or voice_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Secure file names
    image_filename = secure_filename(image_file.filename)
    voice_filename = secure_filename(voice_file.filename)

    # Save files to appropriate folders
    image_path = os.path.join(upload_folder_images, image_filename)
    voice_path = os.path.join(upload_folder_voices, voice_filename)
    image_file.save(image_path)
    voice_file.save(voice_path)

    return image_path, voice_path