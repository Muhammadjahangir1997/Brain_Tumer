import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Correct dataset path
dataset_path = r"C:\Users\A.M\Desktop\brain tomer\brain_tumor_dataset"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# Simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("brain_tumor_model.h5")

print("Training complete. Model saved as brain_tumor_model.h5")
