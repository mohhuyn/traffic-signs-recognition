# traffic-signs-recognition

Overview

This project leverages TensorFlow and Streamlit to create a web application that classifies traffic sign images. The app uses a pre-trained deep learning model to classify uploaded images into various traffic sign categories.

The model is based on a Convolutional Neural Network (CNN) architecture that has been trained to recognize a set of traffic signs.
Requirements

To run this project locally, ensure that you have the following dependencies installed:

    Python 3.x
    Streamlit: pip install streamlit
    TensorFlow: pip install tensorflow
    OpenCV: pip install opencv-python
    Pillow: pip install Pillow
    Matplotlib: pip install matplotlib

Files
Main Python Script (app.py)

This is the main script that runs the Streamlit web application.

    Imports: The script imports the necessary libraries, including streamlit, tensorflow, cv2, and PIL for image processing.

    Model Loading: The function load_model() loads the pre-trained model architecture (model.json) and weights (model.h5). The model is loaded using tensorflow.keras.models.model_from_json and model.load_weights.

    File Upload: The app allows users to upload an image (JPEG or PNG) through the st.file_uploader widget. Once an image is uploaded, the app processes and classifies it using the pre-trained model.

    Image Classification: The uploaded image is resized and normalized before being passed to the model for prediction. The model outputs the predicted class and the corresponding label for the traffic sign.

    Displaying Results: The classified label and similarity score (confidence level) are displayed on the Streamlit interface. The app also shows the uploaded image to give users visual feedback.

    Caching: The model loading is cached to avoid reloading it every time the user interacts with the app, improving performance.

Key Functions

    load_model(): Loads the pre-trained model from the JSON architecture file and the weights from the .h5 file.

    upload_predict(upload_image, model): This function processes the uploaded image, resizes it to the required dimensions (224x224), normalizes it, and feeds it to the model for classification. It returns the predicted class, score, and label.

Usage

    Clone or download the repository containing the script and required files.
    Place the following files in the same directory as the script:
        model.json (model architecture)
        model.h5 (model weights)
        universite-toulouse-iii-paul-sabatier-logo-vector.png (optional logo)
    Run the Streamlit app:

    streamlit run app.py

    Visit the app in your web browser. You will be prompted to upload an image for classification.

Traffic Sign Labels

The app recognizes 58 different traffic sign classes, including speed limits, road directions, warnings, and more. The classification results include the label and a similarity score that indicates the confidence of the model in its prediction.

Example of labels:

    Speed limit (5 km/h)
    No U-turn
    Zebra crossing
    Dangerous curve to the left
