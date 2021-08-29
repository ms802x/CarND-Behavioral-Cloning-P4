
import csv 
import numpy as np
import cv2
from scipy import ndimage

#this fumction reads the CSV file and return left,center and right images. Also the steering measurments.
def read_csv():
	lines = list()
	with open("./data/driving_log.csv") as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
        
	return  lines


# Read images from left,right,center and its measurement with correction factor to keep car in the center
def read_imgs(lines):
	images = list()
	measurements = list()
	for line in lines[1:]:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			#print(source_path)
			
			current_path = "./data/IMG/" + filename
			#print(current_path)
			#image =	cv2.imread(current_path)
			#print(image)
			#exit()
			#RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = ndimage.imread(current_path)
			images.append(image)
		
		correction_factor = 0.2
		measurement = float(line[3])
		measurements.append(measurement)
		measurements.append(measurement+correction_factor)
		measurements.append(measurement-correction_factor)
	return images,measurements


# Augment the images. #read the images and measurements. Then flip the images and reverse the measurments.
def augment_images(images,measurements):
	aug_imgs= list()
	aug_meas = list()
	for img,meas in zip(images,measurements):
		aug_imgs.append(img)
		aug_meas.append(meas)
		rev_img = cv2.flip(img,1)
		rev_meas= -1.0*float(meas)
		aug_imgs.append(rev_img)
		aug_meas.append(rev_meas)
	return aug_imgs, aug_meas

def Lenet(aug_imgs, aug_meas):
	X_train = np.array(aug_imgs)
	y_train = np.array(aug_meas)
	#print(X_train.shape)
	#exit()


	import keras
	from keras.models import Sequential 
	from keras.layers import Flatten, Dense, Lambda
	from keras.layers.convolutional import Convolution2D,Cropping2D
	from keras.layers.pooling import MaxPooling2D


	model = Sequential()
	#normlize the images
	model.add( Lambda (lambda x: x/255 -0.5, input_shape= (160,320,3) ))
	#crop top and bottom
	model.add(Cropping2D(cropping= ( (70,25),(0,0))))

	model.add(Convolution2D(6,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(16,5,5,activation="relu"))
	model.add(MaxPooling2D())

	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))

	model.compile(loss='mse',optimizer='adam')
	model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=3)
	model.save("model.h5")


def main():

	lines = read_csv()
	images , measurements = read_imgs(lines)
	aug_imgs, aug_meas = augment_images(images,measurements)
	Lenet(aug_imgs, aug_meas)

if __name__ == "__main__":
    main()