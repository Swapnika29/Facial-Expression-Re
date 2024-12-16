import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained MobileNet emotion recognition model
model = load_model("20_epochs_mobilenet_reloaded.keras")  # Update with your model file path
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y + h, x:x + w]
        
        # Preprocess the face for the MobileNet model
        resized_face = cv2.resize(face_roi, (48, 48))  # Resize to the model input size
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        processed_face = np.expand_dims(rgb_face, axis=0) / 255.0  # Normalize

        # Predict the emotion
        predictions = model.predict(processed_face, verbose=0)
        emotion_index = np.argmax(predictions)
        emotion = emotion_labels[emotion_index]

        # Draw a red rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Display the emotion text in red above the rectangle
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with emotion detection
    cv2.imshow('Real-time Emotion Detection with MobileNet', frame)

    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
