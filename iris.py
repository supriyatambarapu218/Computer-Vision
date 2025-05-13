import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define the path to the dataset
train_dir = r'D:\5f1\iris'  # Path to the parent directory containing Setosa, Versicolor, Virginica
test_dir = r'D:\5f1\iris'   # Assuming test data is in the same directory (or you can use a separate test set)

# Initialize ImageDataGenerators for training and testing datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize the images
    rotation_range=40,     # Randomly rotate images
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2,       # Shear angle
    zoom_range=0.2,        # Zoom images
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest',
    validation_split=0.2   # Fill any newly created pixels
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for the test data

# Set up the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the parent directory
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='categorical'  # Multi-class classification
)

# Set up the test data generator
test_generator = test_datagen.flow_from_directory(
    test_dir,  # Path to the parent directory
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='categorical'  # Multi-class classification
)

# Define the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the data for the fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer with 3 classes (Setosa, Versicolor, Virginica)
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Save the model after training
model.save('iris_flower_classifier.h5')

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()




from tensorflow.keras.models import load_model  # Import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('iris_flower_classifier.h5')

# Load a new image for prediction (adjust path as necessary)
img_path = 'virginic.jpg'  # Provide path to the image you want to classify

# Load the image and preprocess it
img = image.load_img(img_path, target_size=(128, 128))  # Resize image to 128x128
img_array = image.img_to_array(img)  # Convert image to numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 128, 128, 3)

# Normalize the image (same as done during training)
img_array = img_array / 255.0

# Use the model to predict the class
predictions = model.predict(img_array)

# The model will output a probability distribution (since we used softmax)
# Find the index of the class with the highest probability
class_index = np.argmax(predictions, axis=1)

# Class labels (ensure these match the order you used when training)
class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Print the predicted class label
predicted_class = class_labels[class_index[0]]
print(f'The predicted class is: {predicted_class}')


plt.show()
