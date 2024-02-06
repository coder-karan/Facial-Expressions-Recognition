from keras import Sequential
from keras import callbacks
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data=ImageDataGenerator(rescale=1./255)
validation_data=ImageDataGenerator(rescale=1./255)

train_gen=train_data.flow_from_directory(
    '/dataset/train',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

valid_gen=validation_data.flow_from_directory(
    '/dataset/test',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)


model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

earlystopping=callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=3,
    restore_best_weights=True
)

info=model.fit(
    train_gen,
    epochs=30,
    validation_data=valid_gen,
    validation_steps=7178//64,
    callbacks=[earlystopping]
)

model.save('model.h5')