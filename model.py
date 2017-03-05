import csv
import cv2
import numpy as np
import keras
import gc

lines = []
with open('./training_data/driving_log.csv') as csvfile:
	reader =csv.reader(csvfile)
	#skip header
	next(reader, None)
	for line in reader:
		lines.append(line)

images = []
measurements = []

#due to mem constraints only use random side cameras and flipped images to augment
line_count = 0
for line in lines:
	line_count +=1
	if line_count%3 == 0: 
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			local_path = "./training_data/IMG/" + filename
			image = cv2.imread(local_path)
			images.append(image)
		measurement = line[3]
		measurements.append(measurement)
		#need to correct for left and right cameras
		measurements.append(float(measurement)+ 0.15)
		measurements.append(float(measurement) - 0.15)
	else:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		local_path = "./training_data/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
		measurements.append(line[3])


#tried using the flipped images but it made training a lot slower and had already recorder the clockwise laps
augmented_images = []
augmented_measurements = []
augmented_count = 0
for image, measurement in zip(images, measurements):
	augmented_count +=1
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	if augmented_count%3 == 0:
		flipped_image = cv2.flip(image, 1)
		flipped_measurement = float(measurement) * -1.0
		augmented_images.append(flipped_image)
		augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(len(X_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout

model = Sequential()

#crop image to get rid of sky and font of car
model.add(Cropping2D(cropping=((65,23),(0,0)), input_shape=(160,320,3)))
print(model.output_shape)
#normalize features -- color image normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5))
print(model.output_shape)
#try nvidias autonomous vehicle netowrk architecture
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
print(model.output_shape)
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
print(model.output_shape)
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
print(model.output_shape)
model.add(Convolution2D(64,3,3, activation="relu"))
print(model.output_shape)
model.add(Convolution2D(64,3,3, activation="relu"))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(100))
print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(50))
print(model.output_shape)
model.add(Dense(10))
print(model.output_shape)
model.add(Dense(1))
print(model.output_shape)
model.compile(optimizer='adam', loss='mse')

#by default already shuffles
model.fit(X_train, y_train, validation_split=0.2, nb_epoch=3)

model.save('model.h5')


