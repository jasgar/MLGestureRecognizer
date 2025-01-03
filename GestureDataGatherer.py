import cv2
import mediapipe as mp

import keyboard


NONE_KEY = 'x'

THUMBS_UP_KEY = 'a'
THUMBS_DOWN_KEY = 'z'
ROCK_ON_KEY = 'r'
HANG_LOOSE_KEY = 'h'
PINCH_KEY = 'p'

GESTURE_CLASSES = ['x', 'a', 'z', 'r', 'h', 'p']

show_hands_wireframe = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

video_capture = cv2.VideoCapture(0)

stream = None


landmarks = []

def save_landmark(landmark, gesture_class_index):
  new_landmark = []
  for i in range(0,21):
    new_landmark.append(landmark[i].x)
    new_landmark.append(landmark[i].y)
    new_landmark.append(landmark[i].z)
  
  for i in range(0, len(GESTURE_CLASSES)):
    if (i == gesture_class_index):
      new_landmark.append(1)
    else:
      new_landmark.append(0)

  # print(new_landmark)
  landmarks.append(new_landmark)

def write_landmarks_to_file(filename):
  if (len(landmarks) > 0):
    with open(filename, 'a') as output_file:
      for i in range(0, len(landmarks)):
        landmark = landmarks[i]
        the_str = ""
        for j in range(0, len(landmark)):
          the_str += str(landmark[j]) + ", "
        the_str = the_str[:-2]
        the_str += '\n'
        output_file.write(the_str)



def handHasHorns(landmarks):
  #Y 0 is top of the image
  if ((landmarks[8].y < landmarks[6].y) and
    (landmarks[20].y < landmarks[18].y) and
    (landmarks[12].y > landmarks[10].y) and
    (landmarks[16].y > landmarks[14].y)):
    return True
  return False

def print_landmark(landmark):
  print("---")
  for i in range(0, 21):
    print(f"[{i}]: {landmark[i].x}, {landmark[i].y}, {landmark[i].z}")


with mp_hands.Hands() as hands:
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

    if (keyboard.is_pressed('q')):
      break

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        if (keyboard.is_pressed(NONE_KEY)):
          print("NONE pressed")
          save_landmark(hand_landmarks.landmark, 0)
        if (keyboard.is_pressed(THUMBS_UP_KEY)):
          print("thumbs up pressed")
          save_landmark(hand_landmarks.landmark, 1)
        if (keyboard.is_pressed(THUMBS_DOWN_KEY)):
          print("thumbs down pressed")
          save_landmark(hand_landmarks.landmark, 2)
        if (keyboard.is_pressed(ROCK_ON_KEY)):
          print("rock on pressed")
          save_landmark(hand_landmarks.landmark, 3)
        if (keyboard.is_pressed(HANG_LOOSE_KEY)):
          print("hang loose pressed")
          save_landmark(hand_landmarks.landmark, 4)
        if (keyboard.is_pressed(PINCH_KEY)):
          print("pinch pressed")
          save_landmark(hand_landmarks.landmark, 5)

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

write_landmarks_to_file('landmarks_data.csv')