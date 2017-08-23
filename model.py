import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
samples=[]
'''
first_line_flag is a flag to avoid reading the first row which contains the labels of the columns
'''
first_line_flag=True

with open('./data/driving_log.csv')as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
       if first_line_flag:
          first_line_flag=False
          continue
       samples.append(line)
'''
split the data 80%-20%
'''
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
def generate(data,batch_size=32,train=True):
    num_samples=len(data)
    while(1):
        shuffle(data)
        images=[]
        measurements=[]
        aug_images=[]
        aug_measurements=[]
        for offset in range(0,num_samples,batch_size):
            batch_samples=data[offset:offset+batch_size]
            for i in range(3):
                images.clear()
                measurements.clear()
                for line in batch_samples:
                    source_path=line[i]
                    tokens=source_path.split('/')
                    filename=tokens[-1]
                    local_path="./data/IMG/"+filename
                    image=cv2.imread(local_path)
                    images.append(image)
                    measurement=float (line[3])
                    correction=0.1
                    if i==0:
                        measurements.append(measurement)
                    elif i==1:
                        measurements.append(measurement+correction)
                    elif i==2:
                        measurements.append(measurement-correction)
                    if train:
                        aug_images.clear()
                        aug_measurements.clear()
                        for image,measurement in zip(images,measurements):
                            aug_images.append(image)
                            aug_measurements.append(measurement)
                            flipped_img=cv2.flip(image,1)
                            flipped_measurement=measurement*(-1.0)
                            aug_images.append(flipped_img)
                            aug_measurements.append(flipped_measurement)
                if train:
                    X=np.array(aug_images)
                    y=np.array(aug_measurements)
                else:
                    X=np.array(images)
                    y=np.array(measurements)
                yield shuffle(X,y)


train_generator=generate(train_samples)
validation_generator=generate( validation_samples,train=False)

import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.core import SpatialDropout2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(SpatialDropout2D(0.1))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(SpatialDropout2D(0.1))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(SpatialDropout2D(0.1))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(1))
print('training')

'''
callbacks are to make useful check points and to make the training stop as optimum as possible 
'''
callbacks=[ModelCheckpoint('./model_checkPoint.h5', monitor='val_loss', mode='auto', period=1),
        EarlyStopping(monitor='val_loss', min_delta=0,patience=0, mode='auto')]

model.compile(optimizer='adam',loss='mse')
'''
len(train_samples= number of the rows in the csv file so by including 
the left and right images x3 then flipping x2 and go over the data twice x2
so len(train_samples)*3*2*2

in the validation data we dont need the flipping becase these data should represent the real enviroment so it is
len(validation_samples)*3*2 without the x2 of the flipping
'''
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*12, validation_data=validation_generator,nb_val_samples=len(validation_samples)*6, nb_epoch=10,callbacks=callbacks)
model.save('model.h5')
print('finished')
