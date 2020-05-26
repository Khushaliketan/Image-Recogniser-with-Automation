from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Activation
from keras.models import Sequential
import random
from keras.optimizers import RMSprop, Adam, Adamax, SGD, Nadam

#input shape
img_rows=250
img_cols=250

#Importing our images for recognition
from keras.preprocessing.image import ImageDataGenerator

train_data="/root/model/Dataset/train/"
test_data="/root/model/Dataset/val/"

#Resizing each image to 250x250
from PIL import Image
import os, sys

train_dirs = os.listdir(train_data)
test_dirs = os.listdir(test_data)

'''def resize(path, dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((250,250), Image.ANTIALIAS)
            imResize.save("D://Users//OWNER//MLOps-ws//DL Task - CNN//GOT_Cast//train//"+ f + ' resized.jpeg', 'JPEG', quality=90)

resize(train_data, train_dirs)
resize(test_data, test_dirs)'''


#Data image augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(img_rows, img_cols),
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_data,
        target_size=(img_rows, img_cols),
        class_mode='categorical',
        shuffle=False)

model = Sequential()

model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                        input_shape=(img_rows, img_cols, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu'
                       ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

def addCRP(model, no_CRP):
    for i in range(no_CRP):
        model.add(Convolution2D(filters=random.randint(30,257), 
                        kernel_size=random.choice(((3,3),(5,5),(7,7))), 
                        activation='relu'
                ))
        model.add(MaxPooling2D(pool_size=random.choice((2,2),(4,4))))
        model.add(BatchNormalization())
        model.add(Flatten())

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))

def addDense(model, no_Dense):
    for i in range(no_Dense):
        model.add(Dense(units=random.choice((32,64,128,256,512)), activation='relu'))

#Model summary
model.summary()

#Output layer - manually added - we have 5 classes
model.add(Dense(units=5, activation='softmax'))
    
    #Compiling the model
model.compile(optimizer=random.choice((RMSprop(lr=0.001), Adam(lr=0.001), Adamax(lr=0.001), SGD(lr=0.001), Nadam(lr=0.001))),
        loss='categorical_crossentropy',
        metrics=['accuracy']
              )
    

i=0
while(i!=10):
    #Model summary
    model.summary()

    #Output layer - manually added - we have 5 classes
    model.add(Dense(units=5, activation='softmax'))
    
    #Compiling the model
    model.compile(optimizer=random.choice((RMSprop(lr=0.001), Adam(lr=0.001), Adamax(lr=0.001), SGD(lr=0.001), Nadam(lr=0.001))),
        loss='categorical_crossentropy',
        metrics=['accuracy']
              )
    out = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=random.randint(5,10),
        validation_data=test_generator)
    
    print(out.history, end='\n\n\n')

    print(out.history['accuracy'][0])

    model.save('5CelebClassifier.h5')

    mod =str(model.summary())
    accuracy = str(out.history['accuracy'][0])

    if out.history['accuracy'][0] >= .80:
        import smtplib
        # creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security 
        s.starttls()

        # Authentication 
        s.login("thakkarkhushali368@gmail.com", "Spamyouspamme@28")


        # message to be sent 
        message1 = accuracy
        message2 = mod


        # sending the mail 
        s.sendmail("thakkarkhushali368@gmail.com", "khushali.thakkar9@gmail.com", message1)
        s.sendmail("thakkarkhushali368@gmail.com", "khushali.thakkar9@gmail.com", message2)

        # terminating the session 
        s.quit()
        break
    else:
        loaded_model=model.load_weights('5CelebClassifier.h5')
        loaded_model.layers.pop()
        #addCRP(loaded_model, random.randint(1,3))
        addDense(loaded_model, random.randint(1,5))
        #loaded_model.add(Dense(units=5, activation='softmax'))
        i=i+1

