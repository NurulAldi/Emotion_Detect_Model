import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('emotion_detec_mode.h5') # path to the model
emotions = ['angry', 'fear', 'sad', 'happy', 'neutral', 'disgust', 'surprise']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        resized_face = cv2.resize(face_roi, (48, 48))
        
        normalized_face = resized_face / 255.0

        reshaped_face = np.expand_dims(normalized_face, axis=0)
        reshaped_face = np.expand_dims(reshaped_face, axis=-1)

        prediction = model.predict(reshaped_face)
        predicted_emotion = emotions[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, predicted_emotion, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit the window
        break

cap.release()
cv2.destroyAllWindows()