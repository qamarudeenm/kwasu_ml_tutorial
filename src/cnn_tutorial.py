import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. DOWNLOAD THE DATA
# The computer downloads 60,000 images of planes, cars, birds, etc.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# 2. WASH THE INGREDIENTS (Normalization)
# Pixel values are 0 to 255. That's too big for the math to handle easily.
# We divide by 255.0 to squash all numbers between 0 and 1.
train_images, test_images = train_images / 255.0, test_images / 255.0

print("Groceries loaded and washed!")


# Let's look at the first 5 images
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()



model = models.Sequential()

# LAYER 1: The First Scan
# Filters=32 (32 different flashlights looking for lines/edges)
# Kernel=(3,3) (The size of the flashlight beam)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# LAYER 2: The Thumbnail Maker (Pooling)
# Shrink the image to make processing faster
model.add(layers.MaxPooling2D((2, 2)))

# LAYER 3: The Second Scan
# Filters=64 (Now we look for complex shapes like circles or corners)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# LAYER 4: Shrink it again
model.add(layers.MaxPooling2D((2, 2)))

# LAYER 5: The Final Scan (Looking for whole objects, like heads or wheels)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

print("Brain structure built!")




# Flatten: Take the 2D feature map and unroll it into a list (now it's okay to do this!)
model.add(layers.Flatten())

# Dense: The thinking layer. 64 Neurons connecting to each other.
model.add(layers.Dense(64, activation='relu'))

# Output: 10 Neurons (Because we have 10 categories: Dog, Cat, Ship, etc.)
model.add(layers.Dense(10))

model.summary()





# Setup the rules for the exam
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# START TRAINING
# Epochs = 10 (It will go through the flashcards 10 times)
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))




# Pick an image from the test set (e.g., image number 100)
import numpy as np
img_index = 100 

# Grab the image
img = test_images[img_index]
# Show us the image
plt.imshow(img)
plt.show()

# Ask the model: "What is this?"
# We have to add a fake dimension because the model expects a batch of images
prediction = model.predict(np.expand_dims(img, 0))
predicted_class = class_names[np.argmax(prediction)]

print(f"The model thinks this is a: {predicted_class}")




