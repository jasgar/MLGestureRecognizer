import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_column_names():
  names = []
  for i in range(0,21):
    names.append(str(i) + "x")
    names.append(str(i) + "y")
    names.append(str(i) + "z")

  names.append("is_none")
  names.append("is_thumbs_up")
  names.append("is_thumbs_down")
  names.append("is_rock_on")
  names.append("is_hang_loose")
  names.append("is_pinch")
  return names

def prepare_data():
  data = pd.read_csv('landmarks_data.csv', names=get_column_names())


  features = data.copy()
  labels = features[['is_none', 'is_thumbs_up', 'is_thumbs_down', 'is_rock_on', 'is_hang_loose', 'is_pinch']].copy()
  features = features.drop(['is_none', 'is_thumbs_up', 'is_thumbs_down', 'is_rock_on', 'is_hang_loose', 'is_pinch'], axis=1)

  features = np.array(features)
  labels = np.array(labels)

  x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
  return x_train, x_test, y_train, y_test

def train_model(x_train, x_test, y_train, y_test, desired_accuracy):

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(6, activation='tanh'),
    tf.keras.layers.Softmax()
  ])

  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4),
    metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=120)

  loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

  if (accuracy > desired_accuracy):
    # print('Saving weights to gestures.weights.h5')
    # model.save_weights('gestures.weights.h5')
    print('Saving model to gestures.keras')
    model.save('gestures.keras')

  return model


def create_categories():
  categories = []
  categories.append("is_none")
  categories.append("is_thumbs_up")
  categories.append("is_thumbs_down")
  categories.append("is_rock_on")
  categories.append("is_hang_loose")
  categories.append("is_pinch")
  return categories

def evaluate_errors(x_test, y_test, model)
  categories = create_categories()
  errors = [0, 0, 0, 0, 0, 0]
  correct = [0, 0, 0, 0, 0, 0]
  bad_guesses = [[], [], [], [], [], []]
  for i in range(0, x_test.shape[0]):
    prediction = model.predict(np.reshape(x_test[i], (1, 63)))
    real_answer = np.argmax(y_test[i])
    if (np.argmax(prediction) != real_answer):
      errors[real_answer] += 1
      bad_guesses[real_answer].append(np.argmax(prediction))
    else:
      correct[real_answer] += 1

  for i in range(0, 6):
    print(categories[i] + " ---")
    print("  error %: " + str(errors[i] / (errors[i] + correct[i])))
    print("  bad guesses: " + str(bad_guesses[i]))
    print("")


if __name__ == "__main__":
  x_train, x_test, y_train, y_test = prepare_data()
  model = train_model(x_train, x_test, y_train, y_test, 0.92)

  #evaluate_errors(x_test, y_test, model)

# Accuracy > 90%
# 150 epochs, learning_rate 5e-4
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(64, activation='tanh'), #128 seems to sometimes get higher
#   tf.keras.layers.Dense(6, activation='tanh'),
#   tf.keras.layers.Softmax()
# ])

#Same things, but 128 nodes and 120 epochs does better
