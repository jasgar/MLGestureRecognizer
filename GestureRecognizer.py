import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

show_hands_wireframe = True


def convert_landmark(landmark):
  new_landmark = []
  for i in range(0,21):
    new_landmark.append(landmark[i].x)
    new_landmark.append(landmark[i].y)
    new_landmark.append(landmark[i].z)

  return new_landmark

def run_gesture_recognition():

  GESTURE_CLASSES = ['none', 'thumbs up', 'thumbs_down', 'rock on', 'hang loose', 'pinch']

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  model = tf.keras.models.load_model('gestures.keras')

  video_capture = cv2.VideoCapture(0)
  with mp_hands.Hands() as hands:
    current_state = 'none'
    while video_capture.isOpened():
      success, image = video_capture.read()

      if not success:
        print("Ignoring empty camera frame.")
        continue

      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

          prediction = model.predict(np.reshape(convert_landmark(hand_landmarks.landmark), (1, 63)), verbose=0)
          if (current_state != GESTURE_CLASSES[np.argmax(prediction)]):
            current_state = GESTURE_CLASSES[np.argmax(prediction)]
            print("Current state: " + current_state)

          if (show_hands_wireframe == True):
            mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())


      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break

  video_capture.release()

if __name__ == '__main__':
  run_gesture_recognition()
