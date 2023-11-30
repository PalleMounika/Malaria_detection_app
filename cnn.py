# Image Classification
import tensorflow as tf

# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense

# Initalize CNN
classifier = Sequential()

# Add 2 convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add global average pooling layer
classifier.add(GlobalAveragePooling2D())

# Add full connection
classifier.add(Dense(units=2, activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        "C:/Users/MPALLE/Downloads/Malaria/cell_images/training_data",
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')





########train_set= train_set.repeat()

test_set = test_datagen.flow_from_directory(
        "C:/Users/MPALLE/Downloads/Malaria/cell_images/validating_data",
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        train_set,
        steps_per_epoch=250,
        epochs=32,
        validation_data=test_set,
        validation_steps=2000)
##classifier.save('model_categorical_complex.h5')
classifier.save('C:/Users/MPALLE/Downloads/Malaria/cell_images/malaria-cnn-v1.keras')
# Test accuracy of classifier
def test_accuracy(classifier, test_set, steps):
    num_correct = 0
    num_guesses = 0
    for i in range(steps):
        a = test_set.next()
        guesses = classifier.predict(a[0])
        correct = a[1]
        for index in range(len(guesses)):
            num_guesses += 1
            if round(guesses[index][0]) == correct[index]:
                num_correct += 1
    return num_correct, num_guesses
 