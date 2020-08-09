import datetime
import pickle

import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix

from constants import EMOTIONS, WEEK


def train(model: Model, x: [], y: [], EPOCHS: int, batch_size=4, early_stopping=True,
          save_history=True):
    print("Start Training")

    log_dir = "logs/week_{}/fit_{}_class/{}_{}_{}".format(WEEK, len(EMOTIONS), model.name,
                                                          type(model.optimizer).__name__,
                                                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callback_list = [
        ModelCheckpoint(
            filepath=model.name + '.h5',
            monitor='val_acc',
            save_best_only='True',
            verbose=1,
            mode='max'
        ), tensorboard_callback]

    if early_stopping:
        callback_list.append(
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                mode='min'
            )
        )

    if save_history:
        history_file = "history/week_{}/{}_{}_{}.csv".format(WEEK, model.name, type(model.optimizer).__name__,
                                                             datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callback_list.append(
            CSVLogger(filename=history_file)
        )

    tic = datetime.datetime.now()
    history = model.fit(x, y,
                        batch_size=batch_size, epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=True,
                        callbacks=callback_list)
    toc = datetime.datetime.now()

    diff = toc - tic
    print("Finished Training: Took : {} Seconds".format(diff.total_seconds()))

    return history, model


def test(model: Model, x: [], y: []):
    """

    :param model:
    :param x:
    :param y:
    :return:
    """
    matrices = model.evaluate(x, y)

    for i in range(len(model.metrics_names)):
        print("{} : \t {}".format(model.metrics_names[i], matrices[i]))


def randomize_split(data, split_ratio=0.8):
    # shuffle the dataset
    np.random.shuffle(data)

    # divide training and testing dataset
    training_count = int(len(data) * split_ratio)

    training_data = data[:training_count]
    testing_data = data[training_count:]
    return training_data, testing_data


def plot_model(model, filename):
    filename = 'model_plots/{}.png'.format(filename)
    print("Plotting the model. Saving at: {}".format(filename))
    tf.keras.utils.plot_model(
        model,
        to_file=filename,
        show_shapes=True,
        show_layer_names=True
    )


def plot_confusion_matrix(cm,
                          target_names,
                          filename,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, pad=6)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(filename)
    print("Saved Confusion Matrix to: {}".format(filename))


def get_confusion_matrix(model: Model, x_test, y_test, prediction_index=None):
    predictions = model.predict(x_test)
    if prediction_index is None:
        prediction_classes = np.argmax(predictions[0], axis=1)
    else:
        prediction_classes = np.argmax(predictions[prediction_index], axis=1)

    return confusion_matrix(np.argmax(y_test, axis=1), prediction_classes)


def get_dataset(filename='signal-dataset.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data
