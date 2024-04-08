import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import os

def predict_disease(ensemble_model, features):
    # Apply rules to create new features for diabetes
    if features['disease_type'] == 'diabetes':
        if 70 < features['Glucose'] <= 99:
            features['NewGlucose_Normal'] = 1
        else:
            features['NewGlucose_Normal'] = 0

        if features['Glucose'] > 126:
            features['NewGlucose_Secret'] = 1
        else:
            features['NewGlucose_Secret'] = 0

        if 16 <= features['Insulin'] <= 166:
            features['NewInsulinScore_Normal'] = 1
        else:
            features['NewInsulinScore_Normal'] = 0

    # Remove disease_type from features as it's not needed for prediction
    features.pop('disease_type', None)

    # Convert features to a numpy array for prediction
    features_array = np.array(list(features.values())).reshape(1, -1)

    prediction = ensemble_model.predict(features_array)
    return prediction[0]

def main():
    st.title('Disease Prediction App')

    with open('Model/ensemble_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)

    disease_types = ('heart', 'diabetes', 'liver', 'kidney', 'breastcancer', 'malaria', 'pneumonia')
    disease_type = st.selectbox('Select disease type', disease_types)

    if disease_type:
        if disease_type in ['breastcancer', 'malaria', 'pneumonia']:
            st.write("Upload an image for prediction.")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                #st.write(type(image))
                #st.write("Classifying...")

                if disease_type == 'breastcancer':
                    prediction = predict_breastcancer(image)
                elif disease_type == 'malaria':
                    prediction = predict_malaria(image)
                elif disease_type == 'pneumonia':
                    prediction = predict_pneumonia(image)

                st.success(f"Predicted disease status: {prediction}")
        else:
            features = get_features(disease_type)

            if st.button('Predict') and features is not None:
                prediction = predict_disease(ensemble_model, features)
                disease_status = "Have disease" if prediction == 1 else "Do not have disease"
                st.success(f"Predicted disease status: {disease_status}")

def get_features(disease_type):
    features = {}
    features['disease_type'] = disease_type

    # Display input fields based on disease type
    if disease_type == 'diabetes':
        feature_names = ['Glucose', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    elif disease_type == 'heart':
        feature_names = ['sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        st.write("Enter sex: 0 for male, 1 for female")

    elif disease_type == 'liver':
        feature_names = ['Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
                         'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
                         'Aspartate_Aminotransferase', 'Albumin', 'Albumin_and_Globulin_Ratio']
        st.write("Enter Gender: 0 for male, 1 for female")

    elif disease_type == 'kidney':
        feature_names = ['specific_gravity', 'albumin', 'pus_cell', 'haemoglobin',
                         'packed_cell_volume', 'red_blood_cell_count', 'hypertension',
                         'diabetes_mellitus']

    for name in feature_names:
        value = st.text_input(f"Enter value for {name}")
        features[name] = value

    return features

def predict_breastcancer(image, threshold=0.5):
    # Convert image to RGB if it's not already in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    model = tf.keras.models.load_model("Model/breastcancer.keras")
    pred = model.predict(img_array).tolist()
    pred_probability = pred[0][1]  # Probability of positive class (1)
    # Check if predicted probability is above the custom threshold
    if pred_probability >= threshold:
        return "Positive"
    else:
        return "Negative"

def predict_malaria(image):
    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    model = tf.keras.models.load_model("Model/malaria.keras")
    pred = model.predict(img_array).tolist()
    pred_label = np.argmax(pred)
    return "Positive" if pred_label == 1 else "Negative"


def predict_pneumonia(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    model = tf.keras.models.load_model("Model/pneumonia.keras")
    pred = model.predict(img_array).tolist()
    pred_label = np.argmax(pred)
    return "Positive" if pred_label == 1 else "Negative"

if __name__ == "__main__":
    main()
