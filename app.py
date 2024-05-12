from flask import Flask, render_template, redirect, url_for, Response
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Add an Input layer specifying the input shape
        tf.keras.Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Define input shape and number of classes
input_shape = (48, 48, 1)  # Assuming grayscale images of size 48x48
num_classes = 7  # Number of emotion classes (e.g., happy, sad, angry, etc.)

# Create CNN model
model = create_cnn_model(input_shape, num_classes)

# Print model summary
model.summary()

emotion_model = create_cnn_model(input_shape, num_classes)
try:
    emotion_model.load_weights('C:\\Users\\balse\\OneDrive\\Documents\\trait_python\\weight2.h5')
    print("Weights loaded successfully.")
except OSError as e:
    print("Unable to load weights:", e)

# Initialize MTCNN detector
detector = MTCNN()

# Define emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Function to perform emotion detection
def detect_emotion(face):
    # Preprocess the face image for emotion detection
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    # Perform emotion prediction
    preds = emotion_model.predict(face)[0]
    label = np.argmax(preds)
    
    # Print the predicted probabilities for each emotion
    print("Predicted probabilities:", preds)

    # Get the corresponding emotion label
    emotion_label = emotion_labels[label]
    return emotion_label




# Function to perform real-time face detection and emotion recognition
def detect_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        for result in faces:
            x, y, width, height = result['box']
            face = frame[y:y+height, x:x+width]

            # Perform emotion detection
            emotion_label = detect_emotion(face)
            emotion_text = "Emotion: " + str(emotion_label)

            # Draw rectangle around the face and display emotion
            cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to redirect to camera page
@app.route('/camera')
def camera():
    return render_template('camera.html')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on port 5001
    app.run(debug=True, port=5001)
