import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

TRAINING_DIR = "data/training"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
training_set = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=40, class_mode='binary', target_size=(278, 278))

TESTING_DIR = "data/testing"
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)
testing_set = test_datagen.flow_from_directory(TESTING_DIR, batch_size=40, class_mode='binary', target_size=(278, 278))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(278,278, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(33, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(training_set, epochs=2, verbose=1, validation_data=testing_set) #fit_generator
score = model.evaluate(testing_set) #evaluate_generator

model.save('keras_model.h5')
