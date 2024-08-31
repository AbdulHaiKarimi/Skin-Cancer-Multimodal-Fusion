import string

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from num2words import num2words
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Function to convert numerical age to text
def age_to_text(age):
    return num2words(age)


# defining the function to remove punctua|tion
def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# defining function for tokenization
def tokenization(text):
    tokens = text.split()
    return tokens


# defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words("english")

    output = [i for i in text if i not in stopwords]
    return output


def data_preprocssing(path="./HAM10000_metadata.csv"):
    meta_data = pd.read_csv(path)

    # drop the useless columns from the meta data
    del meta_data["lesion_id"]

    # fill the missing values of the age column
    meta_data["age"] = meta_data["age"].fillna(method="ffill")

    # Apply the conversion function to the age column
    meta_data["age_text"] = meta_data["age"].apply(age_to_text)

    # drop the age numerical column
    del meta_data["age"]

    # Convert the abravation of classes to full form
    dx_full_form = {
        "akiec": "Actinic Keratoses and Intraepidermal Carcinoma",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis-Like Lesions",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi",
        "vasc": "Vascular Lesions",
    }

    # Convert the dx_type to full form
    dx_type_full_from = {
        "confocal": "confocal Microscopy",
        "consensus": "consensus",
        "follow_up": "follow-up",
        "histo": "histopathology",
    }

    # applying full form to the columns
    meta_data["dx_type"] = meta_data["dx_type"].map(dx_type_full_from)
    meta_data["dx_Full"] = meta_data["dx"].map(dx_full_form)

    # replacing the unknown gender to other
    meta_data["sex"] = meta_data["sex"].replace("unknown", "other")

    sorted_data = meta_data.sort_values(by="dx")

    # Converting the feature columns into descriptive form
    sorted_data["description"] = sorted_data.apply(
        lambda row: f"""a {row["sex"]} patient, 
                                                   aged {row["age_text"]}, has been diagnosed with {row["dx_Full"]} the diagnosis was made through {row["dx_type"]} 
                                                   and it is located on the {row["localization"]}""",
        axis=1,
    )

    # storing the puntuation free text
    sorted_data["CL_DS"] = sorted_data["description"].apply(
        lambda x: remove_punctuation(x)
    )

    # lower casing: as our data is already in lower case but still we will do it
    sorted_data["CL_DS"] = sorted_data["CL_DS"].apply(lambda x: x.lower())

    # applying function to the column
    sorted_data["msg_tokenied"] = sorted_data["CL_DS"].apply(lambda x: tokenization(x))

    # Removing Stop words
    sorted_data["no_stopwords"] = sorted_data["msg_tokenied"].apply(
        lambda x: remove_stopwords(x)
    )

    return sorted_data


def combine_text_features(age, dx, dx_type, sex, localization):
    """
    Combine the selected text features into a single string.
    """

    sorted_data = data_preprocssing()

    labels = sorted_data["dx"].values
    text = [" ".join(val) for val in sorted_data["no_stopwords"].values]

    labels_index = {
        val: i
        for i, val in enumerate(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    }
    labels = [labels_index[val] for val in labels]
    len(set(" ".join(text).split()))

    train_set, test_set, train_labels, test_labels = train_test_split(
        text, labels, test_size=0.05, random_state=42
    )
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(train_set)

    # converting age numbered into text
    age = num2words(age)

    combined_text = f"a {sex} patient, aged {age}, has been diagnosed with {dx} the diagnosis was made through {dx_type} and it is located on the {localization}"

    # lower casing
    combined_text = combined_text.lower()
    # remove punctuation
    combined_text = remove_punctuation(combined_text)
    # tokenized
    combined_text = tokenization(combined_text)
    # remove stop words
    combined_text = remove_stopwords(combined_text)
    # join the text
    combined_text = " ".join(combined_text)
    # tokenized the preprocessed input text data
    combined_text = tokenizer.texts_to_sequences([combined_text])
    # # apply pad sequences to the preprocessed text data
    combined_text = pad_sequences(combined_text, maxlen=100)
    # return the preprocessed text
    return combined_text


def main():
    # Loading the saved model
    model = tf.keras.models.load_model("./FinalFusedModel.h5")
    # streamlit app
    st.title("Skin Cancer Multimodel Classification APP")

    st.write("""
        This app uses a pre-trained image model and a pre-trained text model to classify inputs.
    """)

    # Image input
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        # st.write("")

    age = st.selectbox(
        "Select Age",
        options=[str(i) for i in range(10, 101)],  # List of age options as strings
    )

    dx = st.selectbox(
        "Select Diagnosis (dx)",
        options=[
            "Nv - Melanocytic Nevi",
            "Mel - Melanoma",
            "Bkl - Benign Keratosis-Like Lesions",
            "Bcc - Basal Cell Carcinoma",
            "Akiec - Actinic Keratoses and Intraepidermal Carcinoma",
            "Vasc - Vascular Lesions",
            "Df - Dermatofibroma",
        ],
    )
        
    dx_type = st.selectbox(
        "Select Diagnosis Type (dx_type)",
        options=["Histo - Histopathology", "Follow-up - Follow-up", "Consensus - Consensus", "Confocal - Confocal Microscopy"],
    )

    sex = st.selectbox("Select Sex", options=["male", "female"])

    localization = st.selectbox(
        "Select Localization",
        options=[
            "Scalp",
            "Ear",
            "Face",
            "Back",
            "Trunk",
            "Chest",
            "Upper extremity",
            "Abdomen",
            "Unknown",
            "Lower extremity",
            "Genital",
            "Neck",
            "Hand",
            "foot",
            "Acral",
        ],
    )

    # Combine the selected text features
    combined_text = combine_text_features(age, dx.split(" - ")[1], dx_type.split(" - ")[1], sex, localization)
    
    print(dx.split("-"))
    # Predict button
    if st.button("Predict"):
        class_names = [
            "Actinic Keratoses and Intraepidermal Carcinoma",
            "Basal Cell Carcinoma",
            "Benign Keratosis-Like Lesions",
            "Dermatofibroma",
            "Melanoma",
            "Melanocytic Nevi",
            "Vascular Lesions",
        ]
        if uploaded_image:
            # Preprocess the image
            image = image.resize((224, 224))  # Resize the image
            image_array = np.array(image)  # Normalize the image
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            pred = model((image_array, combined_text))

            st.write("Predicted Class:", class_names[np.argmax(pred)])
        else:
            st.write("Please upload an image and enter some text to predict.")


if __name__ == "__main__":
    main()
