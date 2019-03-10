# https://www.tensorflow.org/tutorials/keras/basic_text_classification
# https://www.tensorflow.org/guide/keras

import sys
if len(sys.argv) != 2:
	print('Not enough arguments! Required path to .csv!')
	exit()
else:
	csv_link = sys.argv[1]
	# imports
	from tensorflow import keras
	from keras import optimizers
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from keras.utils.vis_utils import plot_model
	import time
	import math

data_type_dic = {
	'received_seq': 'str',
	'is_cw': 'int'
}

# Import data
input_data = (pd.read_csv(csv_link, dtype=data_type_dic)).values

amount_of_received_seq = input_data.shape[0]
received_seq_len = len(input_data[0][0])
received_seq_len_without_spaces = int((received_seq_len + 1) / 2)

received_seq = np.ndarray(shape=(amount_of_received_seq, received_seq_len_without_spaces), dtype=int)
correct_results = np.ndarray(shape=(amount_of_received_seq, 1), dtype=int)

# Prepare data
for i in range(amount_of_received_seq):
	received_seq[i] = np.fromstring(input_data[:, 0][i], dtype=int, sep=' ')
	correct_results[i] = int(input_data[:, 1][i])

# CONFIGURATION PART
SAVE_MODEL = 0
PLOTS_ENABLED = 1
DETAILED_REPORT_ENABLED = 1
PREDICTION_ENABLED = 1

AMOUNT_OF_TRAINING_EPOCHS = 7700

NEURONS_PER_LAYER = 64
AMOUNT_OF_DENSE_LAYERS = 2

PART_OF_FULL_DATA_USED_TO_VAL = 0.2  # <- will be used as validation. (1-x) used for training
PART_TO_DROPOUT = 0.5
LEARNING_RATE = 0.0001
# CONFIGURATION PART

if 0:
	print("List of existing words: ", received_seq)
	print("Correct answers: ", correct_results)

model = keras.Sequential()
model.add(keras.layers.Dense(NEURONS_PER_LAYER, input_dim=received_seq_len_without_spaces, activation='relu'))

for i in range(AMOUNT_OF_DENSE_LAYERS):
	model.add(keras.layers.Dropout(PART_TO_DROPOUT))
	model.add(keras.layers.Dense(NEURONS_PER_LAYER, activation='relu'))

model.add(keras.layers.Dropout(PART_TO_DROPOUT))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(
	loss='binary_crossentropy',					   		   # binary_crossentropy is the best for binary classification
	optimizer=keras.optimizers.RMSprop(lr=LEARNING_RATE),  # RMSprop is the best for binary classification
	metrics=['accuracy']
)

print("Total amount of words: ", amount_of_received_seq)
training_size = math.floor(amount_of_received_seq * (1-PART_OF_FULL_DATA_USED_TO_VAL))
validation_size = amount_of_received_seq - training_size

x_train = received_seq[:training_size]
y_train = correct_results[:training_size]

x_validation = received_seq[training_size:]
y_validation = correct_results[training_size:]

start = time.time()
history = model.fit(
	x_train, y_train,
	epochs=AMOUNT_OF_TRAINING_EPOCHS,
	validation_data=(x_validation, y_validation),
	verbose=DETAILED_REPORT_ENABLED
)
end = time.time()
print("\nTRAINING ENDS.\nTraining time = ", end - start, "seconds")

test_loss, test_acc = model.evaluate(
	x_validation,
	y_validation,
	verbose=DETAILED_REPORT_ENABLED
)
print("\nEVALUATION ENDS.\nEvaluation results: \nTest loss = ", test_loss, "\nTest accuracy = ", test_acc, "\n")

train_stat_correct = 0
val_stat_correct = 0
if PREDICTION_ENABLED:
	predicts = model.predict(x_train)
	for i in range(training_size):
		if np.round(predicts[i]) == y_train[i]:
			train_stat_correct = train_stat_correct + 1
			print('For training example ', i, ' we have ', predicts[i], '. Correct is ', y_train[i], '. |CORRECT')
		else:
			print('For training example ', i, ' we have ', predicts[i], '. Correct is ', y_train[i], '. |BULLSHIT')
	print('[TRAINING] Correct ', train_stat_correct, 'from', training_size, '(', np.round(train_stat_correct/training_size*100), ') percents')
	print('\n=====\n')
	predicts = model.predict(x_validation)
	for i in range(validation_size):
		if np.round(predicts[i]) == y_validation[i]:
			val_stat_correct = val_stat_correct + 1
			print('For validation example ', i, ' we have ', predicts[i], '. Correct is ', y_validation[i], '. |CORRECT')
		else:
			print('For validation example ', i, ' we have ', predicts[i], '. Correct is ', y_validation[i], '. |BULLSHIT')
	print('[VALIDATION] Correct ', val_stat_correct, 'from', validation_size, '(', np.round(val_stat_correct/validation_size*100), ') percents')
# save/recreate model
if SAVE_MODEL:
	# save entire model to a HDF5 file
	model.save('my_model.h5')
	# recreate the exact same model, including weights and optimizer
	model = keras.models.load_model('my_model.h5')

# PLOTS
if PLOTS_ENABLED:
	plot_model(model, to_file='model.png', show_shapes='True', show_layer_names='True')

	history_dict = history.history
	# prepare graph parameters
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	# lets plot it
	plt.figure(figsize=(16, 9))

	plt.subplot(211)
	plt.title('Точность модели')
	plt.xlabel('Эпохи')
	plt.ylabel('Точность')
	plt.plot(epochs, acc, 'b', label='Обучение')
	plt.plot(epochs, val_acc, 'g', label='Тестирование')
	plt.legend()

	plt.subplot(212)
	plt.title('Потери модели')
	plt.xlabel('Эпохи')
	plt.ylabel('Потери')
	plt.plot(epochs, loss, 'b', label='Обучение')
	plt.plot(epochs, val_loss, 'g', label='Тестирование')
	plt.legend()
	plt.show()