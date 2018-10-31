# https://www.tensorflow.org/tutorials/keras/basic_text_classification

# Import
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_type_dic = {'codeword'  	: str,
				 'codeword_dec'	: 'int',
				 'corr_result' 	: 'int'}

# Import data
input_data = pd.read_csv('data/repetition_code_l2.csv', dtype = data_type_dic)
input_data = input_data.values # np.array
codewords_amount = input_data.shape[0]

codewords = input_data[:,1]
print("codewords: ")
print(codewords)

correct_results = input_data[:,2]
print("correct_results: ")
print(correct_results)

model = keras.Sequential()
model.add(keras.layers.Embedding(codewords_amount, 16)) # 16 neurons
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])



x_val = codewords
partial_x_train = codewords

y_val = correct_results
partial_y_train = correct_results


amout_of_training_epochs = 1500

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=amout_of_training_epochs,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(codewords, correct_results)
print(results)


result = model.predict(codewords)
print(result)
'''
# PLOTS AND ETC
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
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''


