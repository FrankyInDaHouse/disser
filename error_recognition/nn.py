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
SAVE_MODEL = 1
LOAD_MODEL = 0
PER_LAYER_WEIGHTS = 0
PLOTS_ENABLED = 0
DETAILED_REPORT_ENABLED = 1
PREDICTION_ENABLED = 0
################################
NEURONS_PER_LAYER = 14
AMOUNT_OF_TRAINING_EPOCHS = 5000
PART_OF_FULL_DATA_USED_TO_VAL = 0.2  # <- will be used as validation. (1-x) used for training
RETRIES_MAX_VALUE = 10
# CONFIGURATION PART

print("Total amount of words: ", amount_of_received_seq)
training_size = math.floor(amount_of_received_seq * (1-PART_OF_FULL_DATA_USED_TO_VAL))
validation_size = amount_of_received_seq - training_size

x_train = received_seq[:training_size]
y_train = correct_results[:training_size]

x_validation = received_seq[training_size:]
y_validation = correct_results[training_size:]

iteration = 0
resulting_table = np.zeros((RETRIES_MAX_VALUE + 3, 5))
for i in range(RETRIES_MAX_VALUE):
	iteration = iteration + 1
	resulting_table[i, 0] = iteration
	################################################################
	model = keras.Sequential()
	# 1 hidden layer
	model.add(keras.layers.Dense(NEURONS_PER_LAYER, input_dim=received_seq_len_without_spaces, kernel_initializer='random_normal', activation='relu'))

	# 2 hidden layer
	model.add(keras.layers.Dense(NEURONS_PER_LAYER, kernel_initializer='random_normal', activation='relu'))

	# Output layer
	model.add(keras.layers.Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
	################################################################
	model.summary()
	plot_model(model, to_file='models/model.jpg', show_shapes='True', show_layer_names='True')
	model.compile(
		loss='binary_crossentropy',	 # binary_crossentropy is the best for binary classification
		optimizer='rmsprop',  		 # rmsprop is the best for binary classification
		metrics=['accuracy']
	)
	start = time.time()
	history = model.fit(
		x_train, y_train,
		epochs=AMOUNT_OF_TRAINING_EPOCHS,
		validation_data=(x_validation, y_validation),
		verbose=DETAILED_REPORT_ENABLED
	)
	profiling = time.time() - start
	resulting_table[i, 4] = profiling
	train_loss, train_acc = model.evaluate(
		x_train,
		y_train,
		verbose=DETAILED_REPORT_ENABLED
	)
	resulting_table[i, 1] = train_acc * 100
	print('\nTRAINING ENDS (', profiling, ')seconds\nTraining results: \n   loss = ', train_loss, '\n   accuracy = ', train_acc)
	val_loss, val_acc = model.evaluate(
		x_validation,
		y_validation,
		verbose=DETAILED_REPORT_ENABLED
	)
	resulting_table[i, 2] = val_acc * 100
	print('Validation results: \n   loss = ', val_loss, '\n   accuracy = ', val_acc)
	total_loss, total_acc = model.evaluate(
		received_seq,
		correct_results,
		verbose=DETAILED_REPORT_ENABLED
	)
	resulting_table[i, 3] = total_acc * 100
	print('Total results: \n   loss = ', total_loss, '\n   accuracy = ', total_acc, '\niteration = ', iteration, '\n')

	if val_acc > 0.9:
		break
#
#
#
#

# table finishing
print('Per iteration table\n № iter  |   Test_acc   | Valid_acc | Total_acc | Profiling')
min_row = iteration
max_row = iteration + 1
avg_row = iteration + 2
resulting_table[min_row, 0] = -1
resulting_table[max_row, 0] = -2
resulting_table[avg_row, 0] = -3
for i in range(1, 5):
	resulting_table[min_row, i] = np.min(resulting_table[:iteration, i])  # min
	resulting_table[max_row, i] = np.max(resulting_table[:iteration, i])  # max
	resulting_table[avg_row, i] = np.average(resulting_table[:iteration, i])  # avg
print(resulting_table)

if SAVE_MODEL:
	# save entire model to a HDF5 file
	model.save('models/my_model.h5')

if LOAD_MODEL:
	# recreate the exact same model, including weights and optimizer
	model = keras.models.load_model('models/my_model.h5')

if PER_LAYER_WEIGHTS:
	print('Per layer weights:')
	for lay in model.layers:
		print(lay.name)
		print(lay.get_weights())

train_stat_correct = 0
val_stat_correct = 0
if PREDICTION_ENABLED:
	predicts = model.predict(x_train)
	print('\n=====\n')
	for i in range(training_size):
		if np.round(predicts[i]) == y_train[i]:
			train_stat_correct = train_stat_correct + 1
			print('For training example ', i, ' we have ', predicts[i], '. Correct is ', y_train[i], '. |CORRECT')
		else:
			print('For training example ', i, ' we have ', predicts[i], '. Correct is ', y_train[i], '. |BULLSHIT')
	print('[TRAINING] Correct ', train_stat_correct, 'from', training_size, '(', train_stat_correct/training_size*100, ') percents')
	print('\n=====\n')
	predicts = model.predict(x_validation)
	for i in range(validation_size):
		if np.round(predicts[i]) == y_validation[i]:
			val_stat_correct = val_stat_correct + 1
			print('For validation example ', i, ' we have ', predicts[i], '. Correct is ', y_validation[i], '. |CORRECT')
		else:
			print('For validation example ', i, ' we have ', predicts[i], '. Correct is ', y_validation[i], '. |BULLSHIT')
	print('[VALIDATION] Correct ', val_stat_correct, 'from', validation_size, '(', val_stat_correct/validation_size*100, ') percents')
	print('[TOTAL] Correct ', train_stat_correct + val_stat_correct, 'from', amount_of_received_seq, '(', (train_stat_correct + val_stat_correct) / amount_of_received_seq * 100, ') percents')

# PLOTS
if PLOTS_ENABLED:
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