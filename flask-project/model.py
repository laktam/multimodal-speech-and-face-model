import cv2
from keras_facenet import FaceNet
from keras.models import load_model
import joblib
from tensorflow.keras.models import load_model
from utils import get_face_from_image, extract_features
from keras.preprocessing.sequence import pad_sequences
import numpy as np


face_model = FaceNet()
max_length = 100
speech_model = load_model('to_load/speech_embedding_model.keras')
label_encoder = joblib.load('to_load/label_encoder.pkl')
model = load_model('to_load/final_model.keras')



def generate_concatenated_embedding(image_path, voice_path):
    features = extract_features(voice_path)
    #padding
    padded_features = pad_sequences([features], maxlen=max_length, padding='post', dtype='float32')
    speech_embedding = speech_model.predict(padded_features)

    face = get_face_from_image(image_path)
    face_embedding = face_model.embeddings(face)
    # concatenation
    concatenated_embedding = np.concatenate((face_embedding[0], speech_embedding[0]))
    # returing a numpy array with one element to pass it directly to model
    return np.array([concatenated_embedding])

def identify_person(predictions, threshold=0.9, unknown_label="unknown"):
    max_prob = max(predictions[0])
    print("max prob ",max_prob)
    # max_index = np.argmax(predictions)
    decoded_labels= label_encoder.inverse_transform(np.argmax(predictions, axis=1))
   
    if max_prob >= threshold:
        # Person identified with high confidence
        return decoded_labels[0]
    else:
        # Confidence below threshold, person is unknown
        return unknown_label
    
def predict(image_path, voice_path):
    embedding = generate_concatenated_embedding(image_path, voice_path)
    predictions = model.predict(embedding)
    return identify_person(predictions,  threshold=0.9)
