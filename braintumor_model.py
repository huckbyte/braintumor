import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dropout, MaxPooling2D, Dense, Flatten
# %matplotlib inline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint



print("start model building")
model = Sequential()

model.add(Convolution2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2,2))

model.add(Convolution2D(100, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='softmax'))


# Model Summary
print("model summary")
print(model.summary())

# Compiling the model
print("start model compilation")
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

training_data_path = "Training"
testing_data_path = "Testing"

# Loading data and Image Augmentation

print("loading datasets")

training = ImageDataGenerator(rescale=1.0/255)
train_generator = training.flow_from_directory(training_data_path, 
                                                    batch_size=10, 
                                                    target_size=(150, 150))

testing = ImageDataGenerator(rescale=1.0/255)
validation_generator = testing.flow_from_directory(testing_data_path, 
                                                         batch_size=10, 
                                                         target_size=(150, 150))


# Model validatoring
print("validating the model")

validator = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


# Training Model
print("training the model building")
history = model.fit(train_generator,
                                   epochs=10,
                                   validation_data=validation_generator,
                                   callbacks=[validator],)


# Loss Curves
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],ls='--')
plt.legend(['Training Loss','Testing Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Curve")

# Accuracy Curves
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'],ls='--')
plt.legend(['Training Accuracy','Testing Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy Curve")


# Model Evaluation
print("model evalution")
model.evaluate(validation_generator)