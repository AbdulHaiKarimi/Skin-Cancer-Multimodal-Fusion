import string

import nltk
import numpy as np
import pandas as pd
from num2words import num2words


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


def data_preprocssing(path="./Dataset/HAM10000_metadata.csv"):
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
    np.unique(meta_data["dx"]), np.unique(meta_data["dx_type"])
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

    # # saveing the file in json and csv format
    # sorted_data.to_json(
    #     "./data/processed/new_meta_data.json", orient="records", lines=True, indent=4
    # )
    # sorted_data.to_csv("./data/processed/new_meta_data.csv", index=False)

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
