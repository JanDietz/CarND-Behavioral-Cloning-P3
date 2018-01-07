import csv
import cv2
import numpy as np

lines = []
with open('C:/Users/Jan/Desktop/sim_data_one_round/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:       
        lines.append(line)

images = []
measurements = []
correction = 0.2
image_sizeX = 320
image_sizeY = 160

def read_image(path,sizeX,sizeY):
	bgr_image = cv2.imread(path)
	bgr_image = cv2.resize(bgr_image,(sizeX, sizeY))
	b,g,r = cv2.split(bgr_image)       # get b,g,r
	rgb_image = cv2.merge([r,g,b])       # switch it to rgb
	return rgb_image
	

for line in lines:
	img_center = read_image(line[0],image_sizeX,image_sizeY)
	img_left = read_image(line[1],image_sizeX,image_sizeY)
	img_right = read_image(line[2],image_sizeX,image_sizeY)
	images.append(img_center)
	images.append(img_left)
	images.append(img_right)
	steer_angle_center = float(line[3])
	measurements.append(steer_angle_center)
	measurements.append(steer_angle_center + correction)
	measurements.append(steer_angle_center - correction)
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(image_sizeY,image_sizeX,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose = 1)
model.save('model.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
