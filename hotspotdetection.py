import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
haarcascade = cv2.CascadeClassifier('C:/Users/MPALLE/Downloads/Malaria/cell_images/cascade.xml')



# Load the Haar Cascade classifier
haarcascade_path = 'C:/Users/MPALLE/Downloads/Malaria/cell_images/cascade.xml'
haarcascade = cv2.CascadeClassifier(haarcascade_path)

if haarcascade.empty():
    st.error("Error: Unable to load the Haar Cascade Classifier.")
else:
    st.success("Haar Cascade Classifier loaded successfully.")

# ... (rest of your code)

# Load your trained model
model = load_model('C:/Users/MPALLE/Downloads/Malaria/cell_images/malaria-cnn-v1.keras')

# Dictionary to map class indices to class names
class_names = {
    0: 'Class Parasitized',
    1: 'Class Uninfected',
    # Add more class names as needed
}

# Function to detect hotspots in an image

def detect_hotspots(image_array):
    # Ensure image is of the correct depth (uint8)
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image_array[0], cv2.COLOR_RGB2HSV)

    # Define the range of magenta color in HSV
    lower_magenta = np.array([140, 50, 50])
    upper_magenta = np.array([170, 255, 255])
    magenta_mask = cv2.inRange(hsv_image, lower_magenta, upper_magenta)

    # Perform morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_OPEN, kernel)
    magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the magenta mask
    contours, _ = cv2.findContours(magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the contours
    image_with_rectangles = image_array[0].copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_with_rectangles








# Streamlit UI
st.title("Image Classification and Hotspot Detection App")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for the model
    image = Image.open(uploaded_image)
    image = image.resize((128, 128))

    # Convert the image to uint8
    image_array = np.array(image, dtype=np.uint8)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    # Perform classification using the model
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)

    # Get the class name from the dictionary
    class_name = class_names.get(predicted_class, 'Unknown Class')

    # Display the result
    st.success(f"Predicted Class: {class_name}")

    # Detect hotspots
    hotspots_image = detect_hotspots(image_array)

    # Display the image with hotspots
    st.image(hotspots_image, caption="Image with Hotspots", use_column_width=True)
