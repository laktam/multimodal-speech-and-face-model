import cv2
from keras_facenet import FaceNet
from PIL import Image
from numpy import asarray
from numpy import expand_dims
import librosa

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# cette method fait l'extraction de visage dans l'image (HaarCascade)
def get_face_from_image(path):
    gbr1 = cv2.imread(path)
    
    faces = HaarCascade.detectMultiScale(gbr1,1.1,4)
    
    if len(faces)>0:
        x1, y1, width, height = faces[0]         
    else:
        x1, y1, width, height = 1, 1, 10, 10
        
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Image.fromarray(gbr)                  # conversion from OpenCV to PIL
    gbr_array = asarray(gbr)
    
    face = gbr_array[y1:y2, x1:x2]                        
    
    face = Image.fromarray(face)                       
    face = face.resize((160,160))
    face = asarray(face)
    
    # face = face.astype('float32')
    # mean, std = face.mean(), face.std()
    # face = (face - mean) / std
    
    face = expand_dims(face, axis=0)
    return face

# extraction des features du speech (mfcc)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)# 13 = refers to the number of extracted features
    return mfcc.T  # Transpose to have time steps first










