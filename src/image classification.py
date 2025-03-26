# Import necessary libraries
!pip install opencv-python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  # Import OpenCV
from google.colab import drive, files
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Mount Google Drive
drive.mount('/content/Mydrive')

# Function to check if an image is valid
def is_valid_image(filepath):
    try:
        Image.open(filepath).verify()
        return True
    except (IOError, SyntaxError) as e:
        return False

# Clean the dataset by removing invalid images
def clean_dataset(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if not is_valid_image(filepath):
                print(f'Removing invalid image: {filepath}')
                os.remove(filepath)

# Set paths
train_dir = '/content/Mydrive/MyDrive/Classification/train'
test_dir = '/content/Mydrive/MyDrive/Classification/test'

# Clean the datasets
clean_dataset(train_dir)
clean_dataset(test_dir)

# Image Data Generators with more augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Use a pre-trained model (MobileNetV2) for better performance
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Use num_classes from the generator
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('my_model.keras')
print("Model saved as 'my_model.keras'")

# Load the model
from tensorflow.keras.models import load_model
model = load_model('my_model.keras')

# Predict the labels for the test data
test_generator.reset()
y_pred = model.predict(
    test_generator,
    steps=int(np.ceil(test_generator.samples / test_generator.batch_size)),  # Convert to integer
    verbose=1
)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true labels
y_true = test_generator.classes[:len(y_pred_classes)]  # Slice y_true to match the length of y_pred_classes

# Classification report and confusion matrix
# Get the class labels from the train generator
class_labels = list(train_generator.class_indices.keys())

# Filter out '__MACOSX' from class_labels if it exists
class_labels = [label for label in class_labels if label != '__MACOSX']

# Ensure y_true and y_pred_classes have the same number of classes
unique_classes = np.unique(np.concatenate([y_true, y_pred_classes]))
if len(unique_classes) != len(class_labels):
    print(f"Warning: Number of unique classes in predictions ({len(unique_classes)}) does not match the number of classes in the dataset ({len(class_labels)}).")
    print("This may happen if some classes are missing in the test set.")

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Function to visualize predictions
def predict_and_visualize(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    img_resized = cv2.resize(img, (150, 150)) / 255.0
    prediction = model.predict(np.expand_dims(img_resized, axis=0))
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels = list(train_generator.class_indices.keys())
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted: {class_labels[predicted_class]}')
    plt.axis('off')
    plt.show()

# Test the function with sample images
predict_and_visualize('/content/Mydrive/MyDrive/Classification/dogs/dog_100.jpg')
predict_and_visualize('/content/Mydrive/MyDrive/Classification/cats/cat_103.jpg')

# Loop for user input or file upload
choice = input("Enter 'path' to input image path, 'upload' to upload an image: ").lower()

if choice == 'path':
    image_path = input("Please enter the path to the image: ")
    predict_and_visualize(image_path)
elif choice == 'upload':
    uploaded = files.upload()
    for filename in uploaded.keys():
        predict_and_visualize(filename)
else:
    print("Invalid choice. Please enter 'path' or 'upload'.")


# Function to analyze the training history
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Call the function to plot the training history
plot_training_history(history)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(cm, classes=class_labels, title='Confusion Matrix')

# Additional analysis: Plotting classification report metrics
def plot_classification_report(y_true, y_pred, class_labels):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    x = np.arange(len(class_labels))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1 Score')

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Report Metrics')
    plt.xticks(x, class_labels, rotation=45)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Call the function to plot classification report metrics
plot_classification_report(y_true, y_pred_classes, class_labels)

# Analyze class distribution in training and test datasets
def plot_class_distribution(generator, title):
    class_counts = np.bincount(generator.classes)
    class_labels = list(generator.class_indices.keys())
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_labels, y=class_counts)
    plt.title(f'{title} Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Plot class distribution for training and test datasets
plot_class_distribution(train_generator, 'Training')
plot_class_distribution(test_generator, 'Test')

