# This example is from: https://www.tensorflow.org/tutorials/keras/classification?hl=en-us

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

print("\nLoading the dataset...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\nExploring the dataset...")
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print("\nPreprocessing the dataset...")
# Scaling images pixels values to a range of 0 to 1:
train_images = train_images / 255.0
test_images = test_images / 255.0

# Checking if the data is in the correct format:
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

print("\nBuilding the model...")
# Setting up the neural network layers:
model = tf.keras.Sequential([
    # Flatten: this is the input layer. It transforms the format of the images from a two-dimensional
    # array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
    # This layer has no parameters to learn; it only reformats the data.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # This is an internal layer that has 128 neurons.
    tf.keras.layers.Dense(128, activation='relu'),
    # This is the output layer and has 10 neurons. Each neuron activation corresponds to the
    # chance that the current image belongs to the class represented by the neuron.
    tf.keras.layers.Dense(10)
])

# Compiling the model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\nTraining the model...")
# 1 - Feed the training data to the model. In this example,
# the training data is in the train_images and train_labels arrays.
# 2 - The model learns to associate images and labels.
# 3 - You ask the model to make predictions about a test set—in this example, the test_images array.
# 4 - Verify that the predictions match the labels from the test_labels array.

# Feed the model: loss and accuracy on the training data are displayed.
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy: compare how the model performs on the test dataset.
# This one performs worse on the test dataset, so it is overfitted.
# It's memorized specific details from training data and hasn't generalized the patterns enough.
print('\nTesting dataset accuracy:')
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# Make predictions: with the model trained, you can use it to make predictions about some images.

# Attaching a softmax layer after the logits to convert them to probabilities...
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# The model has predicted the label for each image in the testing set.
# Each prediction is an array of 10 numbers. Each number conveys the probability
# that the image is belongs to the class it represents.
predictions = probability_model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                 100*np.max(predictions_array),
                                 class_names[true_label]),
                                 color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')

# Plotting the prediction for the first image in the testing dataset's class:
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

print("\nUsing the trained model to make a prediction about a single image...")

# grabbing an image from the test dataset...
i = 5
img = test_images[i]
print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.
# Accordingly, even though you're using a single image, you need to add it to a list:
img = np.expand_dims(img, 0)
print(img.shape)

# tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data.
# Grab the predictions for our (only) image in the batch: (i.e, prediction[0])
prediction = probability_model.predict(img)
plot_value_array(i, prediction[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
