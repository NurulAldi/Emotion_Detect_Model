# Description
This Emotion Detection Model is used to detect the emotion of people in any given images/frames. There are 7 classes of emotions that this model can identified, they are ```[Angry, Fear, Sad, Happy, Neutral, Disgust, Surprise]```

# Model Architecture
The model was built using Tensorflow Keras API and its architecture is shown below :
| Layer (type)       | Output Shape        | Param #   |
|--------------------|--------------------|----------|
| conv2d (Conv2D)   | (None, 46, 46, 64)  | 640       |
| max_pooling2d (MaxPooling2D) | (None, 23, 23, 64) | 0 |
| conv2d_1 (Conv2D) | (None, 21, 21, 128) | 73,856    |
| max_pooling2d_1 (MaxPooling2D) | (None, 10, 10, 128) | 0 |
| conv2d_2 (Conv2D) | (None, 8, 8, 256)   | 295,168   |
| max_pooling2d_2 (MaxPooling2D) | (None, 4, 4, 256) | 0 |
| flatten (Flatten) | (None, 4096)        | 0         |
| dense (Dense)     | (None, 128)         | 524,416   |
| dropout (Dropout) | (None, 128)         | 0         |
| dense_1 (Dense)   | (None, 7)           | 903       |

# Test Model Using real-time videos
Haar Cascade model is used here in order to detect faces in a given frame, this is because the Emotion Detect model was built to detect emotion, not to find faces in a given frame.

Run this code on your own machine and make sure it has camera, these are needed for the code to worked properly
```
import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('emotion_detec_model.h5') # path to the model
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
