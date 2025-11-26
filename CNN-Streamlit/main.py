from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model("../Models/fruit_classifier_modelCNN.h5")
    return model

modelCNN = load_model()

classes = ['apple', 'banana', 'orange']

st.title("Fruit Classifier using CNN")
st.write("Upload an image of a fruit **Banana**, **Apple**, or **Orange** to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    # Preprocess the image so it can be fed to the model
    image = image.resize((32, 32))
    image_array = np.array(image) # / 255.0
    image_array = np.expand_dims(image, axis=0) # the reason for this is that the model expects a batch of images, even if it's just one image
                                                    # a batch of images in definition is a 4D array: (batch_size, height, width, channels), where the batch_size is the number of images
                                                    # height and width are the dimensions of each image, and channels is the number of color channels (3 for RGB images)
    
    predictions = modelCNN.predict(image_array) # this returns a 2D array: [[prob_class1, prob_class2, prob_class3]]
    print(predictions)
    predicted_class = classes[np.argmax(predictions)] # np.argmax(predictions) gives the index of the highest probability
    
    st.subheader("**Prediction Results:**")
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {np.max(predictions) * 100:.2f}%") # np.max(predictions) gives the highest probability   
    
    st.write("## Prediction Probabilities:")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")