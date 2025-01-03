# MLGestureRecognizer

## Project Objectives
This project is specifically about setting up an end-to-end machine learning project, to learn what goes into everything *around* training the model.  Specifically:

* Gathering data for a machine learning application
* Formatting data to be able to train a model
* Evaluating the trained model
* Saving a trained model
* Using a saved model in an application
* Evaluating model performance and how to improve it

It is *not* about:
* Getting the highest accuracy (though I did play around with some different model setups)
* Trying out a specific model type
* Solving a novel problem
* Creating production code (error handling, works on all camera devices, venv setup, etc.)
* Using production methodologies to get an accurate confidence level - I used just a train/test set, not a train/dev/test.

## Process
I first wrote and used GestureDataGatherer.py to gather data from my mac laptop camera.  Note that this script needs to be run with sudo so that we can use the keyboard library to detect key inputs without blocking.  I used mediapipe to convert images to hand landmarks.  I initially gathered over 1000 snapshots of hands in different poses, by making a hand pose and hitting a key to tell the program to save it as a certain pose.  These landmarks from each pose were then saved into a csv.

I then wrote GestureModelTrainer.py to consume the csv and train a simple neural network to recognize the gestures using the landmarks.  I evaluated performance on the test set.  I then also looked at the examples in the test set where the model was getting things wrong to see if there was a way to improve performance (see the evaluate_errors function in GestureModelTrainer.py).  I noticed that the model was failing primarily on the thumbs_down and none categories.  I added some additional data points for those categories to help train better, and this showed improvement.  The current model that is checked in has about 92% accuracy.

Finally, I wrote GestureRecognizer.py to load the saved model and use it to recognize gestures in the camera.  

To run just the end product, you can run GestureRecognizer.py.

## Learnings

* Hyperparameters can make a large difference (learning rate, number of training epochs, etc.)
* Model topology can also have a large impact on accuracy (I increased the accuracy by changing from a sigmoid activation function on hidden layers to a tanh activation function.  I believe that this is because the landmarks are coordinates, and so preserving the ability for those numbers to be positive and negative was helpful, while sigmoid converts it to a 0 to 1 scale)
* Evaluating the test cases where the model predicted incorrectly was very helpful, and gave me a direction for how to improve the model
* It was also helpful to compare the accuracy of the train and test sets.  Since they were both consistently close, I knew that it wasn't overfitting to the training set.
* It's important to have lots of none data - data that wasn't any of the poses

## Possible Improvements

* Using the model to allow controlling something via hand gestures
* Trying out different model configurations - more layers, etc.
* Productizing the code

