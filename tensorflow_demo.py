#!/usr/bin/env python3
"""! TensorFlow demonstration with figures
Most of this is from the following links :
https://www.tensorflow.org/tutorials/quickstart/beginner
https://www.tensorflow.org/tutorials

@author Seth McNeill
@date 2020 October 26
@copyright MIT
"""


import datetime  # used for start/end times
import argparse  # This gives better commandline argument functionality
import doctest   # used for testing the code from docstring examples
import tensorflow as tf     # for machine learning
import pdb          # For debugging
import matplotlib.pyplot as plt     # for plotting
import numpy as np      # for numerical functions

def imshow_random_subset(img_array, label_array, description='', n=5, m=6):
    """! Plots an n x m array of images from img_array
    @param img_array a numpy array of images
    @param description A one or two work description of the dataset
    @param n number of rows of images to show
    @param m number of columns of images to show
    """
    # choose the images to show
    img_idx = np.random.randint(0, len(img_array), n*m)
    plt.subplots(n, m, figsize=(10,6))
    for p in range(n*m):
        plt.subplot(n,m,p+1)  # subplot numbering starts at 1 not 0
        plt.imshow(img_array[img_idx[p]])
        plt.xlabel(f"{label_array[img_idx[p]]}, idx={img_idx[p]}")
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
        #plt.tick_params(axis='both', left='off', top='off', right='off',
        #        bottom='off', labelleft='off', labeltop='off', 
        #        labelright='off', labelbottom='on')
    plt.suptitle(f"Random subset of {description} images")  # title over all the subplots
    plt.tight_layout()

def show_mnist_img(img_data, rec_label):
    """! This nicely displays an image from the mnist data set

    @param img A 2D array for an image
    @param rec_label The correct category/class for this image
    """
    plt.subplots(figsize=(10,6))
    img = img_data
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"This image is labeled as a {rec_label}", fontsize=28)
    plt.tight_layout()

def plot_prediction_prob(predictions, probs, label, title=None):
    """! Plots predictions and probabilities

    @param predictions The prediction values
    @param probs The probability values
    @param labal the catagory/class the sample belongs to
    """
    if title is None:
        title = f'Predictions and Probability of Being a {label}'
    plt.subplots(figsize=(10,6))
    plt.subplot(211)
    plt.bar(np.arange(0,len(predictions)), predictions)
    plt.title('Prediction values', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplot(212)
    plt.bar(np.arange(0,len(probs)), probs)
    plt.title('Probability values', fontsize=20)
    plt.xlabel('Number', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.suptitle(title, fontsize=28)
    plt.tight_layout()


def plot_prob(probs, label, title=None):
    """! Plots probabilities

    @param probs The probability values
    @param labal the catagory/class the sample belongs to
    """
    plt.subplots(figsize=(10,6))
    if title is None:
        title = f'Probability of Being a {label}'
    plt.bar(np.arange(0,len(probs)), probs)
    plt.xlabel('Number', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=28)

def tensor_demo():
    """! This demonstration is copied from 
    https://www.tensorflow.org/tutorials/quickstart/beginner
    with some modifications to make it print better from a normal
    command line.

    There is also more explanation about the dataset and some plotting
    """
    testing_idx = 7144   # index for testing image to evaluate
    training_idx = 7970  # index for training image to evaluate
    # Loads the mnist data set (https://en.wikipedia.org/wiki/MNIST_database)
    # of handwritten digits. Note that it has both training and 
    # testing data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert the dataset from integers into floating point numbers (0-1?)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Let's have a look at the data we are working with:
    print(f"x_test a numpy array of dimensions: {x_test.shape}")
    print(f"x_train a numpy array of dimensions: {x_train.shape}")
    print(f"y_test a numpy array of dimensions: {y_test.shape}")
    print(f"y_train a numpy array of dimensions: {y_train.shape}")
    print(f"x_test max: {np.max(x_test)}")
    print(f"x_train max: {np.max(x_train)}")
    imshow_random_subset(x_test, y_test, 'mnist testing')
    imshow_random_subset(x_train, y_train, 'mnist training')

    # Setup (build) a model of type Sequential
    # https://keras.io/api/models/sequential/
    # Also choose optimizer and loss function
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    # Test the untrained model on the training data
    # this is testing the model on the first image
    predictions = model(x_train[training_idx:(training_idx+1)]).numpy()
    print(f"\nAn untrained model {predictions}\n")
    show_mnist_img(x_train[training_idx], y_train[training_idx])

    # softmax turns the predictions into "probabilities", these
    # probabilities are close to random (1/10 since 10 digits 0-9).
    probs = tf.nn.softmax(predictions).numpy()[0]
    print(f"\n{probs}\n")

    plot_prediction_prob(predictions[0], probs, y_train[training_idx],
            title=f'Untrained Probability of Being a {y_train[training_idx]}')

    # Build a loss function to negative log probability of the true class
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Therefore the untrained model should give a result near -ln(1/10) ~= 2.3
    print(f"\nUntrained loss: {loss_fn(y_train[:1], predictions).numpy()}\n")

    # compile the model
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    # Train the model by adjusting its parameters to minimize the loss function
    # The number of epochs is the number of times to reoptimize the paramters
    model.fit(x_train, y_train, epochs=5)

    # evaluate checks the model's performance on a validation/test set
    print(f"\nTrained model evaluation on a test set: " +
          f"{model.evaluate(x_test,  y_test, verbose=2)}\n")

    # setup the model to return a probability
    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

    probability_training_img = probability_model(x_train[training_idx:(training_idx+1)])
    plot_prob(probability_training_img[0], y_train[training_idx],
        title=f"Probability of training image {training_idx} which is a {y_train[training_idx]}")

    show_mnist_img(x_test[testing_idx], y_test[testing_idx])
    probability_testing_img = probability_model(x_test[testing_idx:(testing_idx+1)])
    plot_prob(probability_testing_img[0], y_test[testing_idx],
        title=f"Probability of testing image {testing_idx} which is a {y_test[testing_idx]}")

    print(f"\nProbability results on testing data {testing_idx}: \n" +
          f"{probability_model(x_test[testing_idx:(testing_idx+1)])}\n")

    plt.show()

    pdb.set_trace()


def main():
    """! Main function that runs TensorFlow example
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doctest', action='store_true',
                        help='Pass this flag to run doctest on the script')
    start_time = datetime.datetime.now()  # save the script start time
    args = parser.parse_args()  # parse the arguments from the commandline

    if(args.doctest):
        doctest.testmod(verbose=True)  # run the tests in verbose mode

    print("-------------------")
    tensor_demo()

    end_time = datetime.datetime.now()    # save the script end time
    print(f'{__file__} took {end_time - start_time} s to complete')



# This runs if the file is run as a script vs included as a module
if __name__ == '__main__':
    main()
