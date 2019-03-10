# https://www.tensorflow.org/tutorials/keras/basic_classification
# https://www.tensorflow.org/guide/keras

import sys
if len(sys.argv) != 2:
    print('Not enough arguments! Required path to .csv!')
    exit()
else:
    csv_link = sys.argv[1]
    # imports
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from keras.utils.vis_utils import plot_model
    import time

data_type_dic = {'codeword': str,
                 'codeword_dec': 'int',
                 'corr_result': 'int'}

# Import data
input_data = pd.read_csv(csv_link, dtype = data_type_dic)
input_data = input_data.values # np.array
codewords_amount = input_data.shape[0]

codewords = input_data[:, 1]
correct_results = input_data[:, 2]

print("codewords: ", codewords)
print("correct_results: ", correct_results)

# CONFIGURATION PART
SAVE_MODEL = 0
PLOTS_ENABLED = 1
DETAILED_REPORT_ENABLED = 1

NEURONS_PER_LAYER = 16
AMOUNT_OF_TRAINING_EPOCHS = 200
# CONFIGURATION PART

model = keras.Sequential()
model.add(keras.layers.Embedding(codewords_amount, NEURONS_PER_LAYER))  # FIXIT
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(NEURONS_PER_LAYER, activation=tf.nn.relu))
model.add(keras.layers.Dense(codewords_amount, activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_codewords = codewords
train_corr_res = correct_results

validation_codewords = correct_results
validation_corr_res = codewords

start = time.time()
history = model.fit(train_codewords, train_corr_res, epochs=AMOUNT_OF_TRAINING_EPOCHS,
                    validation_data=(validation_codewords, validation_corr_res), verbose=DETAILED_REPORT_ENABLED)
end = time.time()
print("Training time = ", end - start, "seconds")

# It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset.
# This gap between training accuracy and test accuracy is an example of overfitting.
# Overfitting is when a machine learning model performs worse on new data than on their training data.
test_loss, test_acc = model.evaluate(codewords, correct_results)
print("Evaluation results : \n Test loss = ", test_loss, "\n Test accuracy = ", test_acc)


# TEST_WORD = np.array([11])
# predicts = model.predict(codewords)
# print(np.around(predicts, decimals=1))

# save/recreate model
if SAVE_MODEL:
    # save entire model to a HDF5 file
    model.save('my_model.h5')
    # recreate the exact same model, including weights and optimizer
    model = keras.models.load_model('my_model.h5')

# PLOTS AND ETC
if PLOTS_ENABLED:
    plot_model(model, to_file='model.png', show_shapes='True', show_layer_names='True')
    history_dict = history.history
    history_dict.keys()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()   # clear figure

    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt	.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
