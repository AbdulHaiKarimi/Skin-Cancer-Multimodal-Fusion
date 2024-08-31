import glob

import cv2
import keras
import mlflow
import numpy as np
import pandas as pd
from ImageModelTraining import pretrained_models, train_ds
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from src.models.TextModelTraining import test_x, text_models, train_x

from src.data.make_text_data import data_preprocssing


def custom_dataGen(
    input, output, labels_index, tokenizer, batch_size=32, image_size=(224, 224)
):
    Labels = output.values
    Labels = [labels_index[val] for val in Labels]
    Labels = to_categorical(np.asarray(Labels))

    # merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    num_samples = len(input["image_id"])
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_filenames = input["filename"][offset : offset + batch_size].values

            batch_labels = Labels[offset : offset + batch_size]
            # batch_labels = [labels_index[val] for val in batch_labels]
            # batch_labels = to_categorical(np.asarray(batch_labels))

            meta_data = input["no_stopwords"][offset : offset + batch_size].values
            meta_data = [" ".join(val) for val in meta_data]

            BatchMetaData = tokenizer.texts_to_sequences(meta_data)
            BatchMetaData = pad_sequences(BatchMetaData, maxlen=100)

            BatchImages = []
            for filename in batch_filenames:
                image = cv2.imread(filename)
                image = cv2.resize(image, image_size)

                BatchImages.append(image)

            BatchImages = np.array(BatchImages)
            BatchMetaData = np.array(BatchMetaData)
            yield (BatchImages, BatchMetaData), batch_labels


def LoadingData():
    # loaing the preprocessed text data
    sorted_data = data_preprocssing()

    # Defind lable indexing for each class
    labels_index = {val: i for i, val in enumerate(train_ds.class_names)}

    # join the text for tokenizer
    text = [" ".join(val) for val in sorted_data["no_stopwords"].values]
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(text)

    # sorted_data
    extracted_ids = [
        filename.split("\\")[-1].split(".")[0]
        for filename in glob.glob("./HAM_10000_Dataset/*/*.jpg")
    ]

    image_df = pd.DataFrame(
        {
            "image_id": extracted_ids,
            "filename": glob.glob("./HAM_10000_Dataset/*/*.jpg"),
        }
    )

    merged_df = pd.merge(
        image_df,
        sorted_data[["image_id", "dx", "no_stopwords"]],
        on="image_id",
        how="left",
    )

    train_x, test_x, train_y, test_y = train_test_split(
        merged_df[["image_id", "filename", "no_stopwords"]],
        merged_df["dx"],
        test_size=0.05,
        random_state=123,
    )

    train_gen = custom_dataGen(
        train_x, train_y, labels_index, tokenizer, batch_size=batch_size
    )
    test_gen = custom_dataGen(
        test_x, test_y, labels_index, tokenizer, batch_size=batch_size
    )

    return train_gen, test_gen


batch_size = 32
train_gen, test_gen = LoadingData()

def StartFusionTraining():
        
    # LSTM and Image Model fusion
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LSTM Late Fusion Experiments")

    for key in pretrained_models.keys():
        try:
            print("=" * 40, f"{key} Model Training Started", "=" * 40)
            # image_input = Input(shape=(224, 224, 3), name="image_input")

            PreImgModel = pretrained_models[key](
                include_top=False, input_shape=(224, 224, 3)
            )
            for layer in PreImgModel.layers:
                layer.trainable = False
            flat1 = keras.layers.Flatten()(PreImgModel.layers[-1].output)
            ImgFeatures = keras.layers.Dense(512, activation="relu")(flat1)
            image_output = keras.layers.Dense(7, activation="softmax")(flat1)

            ImgModel = keras.Model(inputs=PreImgModel.inputs, outputs=image_output)

            text_input = Input(
                shape=(100,), name="text_input"
            )  # Example input shape for text
            TextEmabedding = text_models["LSTM"].layers[0](text_input)
            TextFeatures = text_models["LSTM"].layers[1](TextEmabedding)
            text_output = text_models["LSTM"].layers[-1](TextFeatures)

            TextModel = keras.Model(inputs=text_input, outputs=text_output)

            combined_features = Concatenate()([TextFeatures, ImgFeatures])

            # Add some fully connected layers
            x = Dense(64, activation="relu")(combined_features)
            x = Dense(32, activation="relu")(x)
            final_output = Dense(7, activation="softmax")(
                x
            )  # Assuming 7 classes for classification

            # Define the combined model
            combined_model = Model(
                inputs=[PreImgModel.inputs, TextModel.inputs], outputs=final_output
            )

            # model compilation
            combined_model.compile(
                optimizer=keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            with mlflow.start_run(run_name=f"{key}-LSTM", nested=True) as run:
                # log model parametes
                mlflow.log_param("Optimizer", "Adam")
                mlflow.log_param("Loss", "Sparse_Categorical_Crossentropy")
                mlflow.log_param("Batch_size", 32)
                mlflow.log_param("Epochs", 20)

                # Define the number of steps per epoch and validation steps
                steps_per_epoch = len(train_x) // batch_size
                validation_steps = len(test_x) // batch_size

                # Fit the combined model using the generators
                history = combined_model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_gen,
                    validation_steps=validation_steps,
                    epochs=20,
                    callbacks=[early_stopping],
                )

                # Log metrics
                for epoch in range(20):
                    mlflow.log_metric(
                        "train_accuracy", history.history["accuracy"][epoch], step=epoch
                    )
                    mlflow.log_metric(
                        "val_accuracy", history.history["val_accuracy"][epoch], step=epoch
                    )
                    mlflow.log_metric(
                        "train_loss", history.history["loss"][epoch], step=epoch
                    )
                    mlflow.log_metric(
                        "val_loss", history.history["val_loss"][epoch], step=epoch
                    )

                    # log the model
                    # mlflow.keras.log_model(model,f"Keras-model-{key}")
        except Exception as e:
            print(e)


    # CNN1D and Image Model fusion
    mlflow.set_experiment("CNN1D Late Fusion Experiments")

    for key in pretrained_models.keys():
        try:
            print("=" * 40, f"{key} Model Training Started", "=" * 40)
            # image_input = Input(shape=(224, 224, 3), name="image_input")

            PreImgModel = pretrained_models[key](
                include_top=False, input_shape=(224, 224, 3)
            )
            for layer in PreImgModel.layers:
                layer.trainable = False
            flat1 = keras.layers.Flatten()(PreImgModel.layers[-1].output)
            ImgFeatures = keras.layers.Dense(512, activation="relu")(flat1)
            image_output = keras.layers.Dense(7, activation="softmax")(flat1)

            ImgModel = keras.Model(inputs=PreImgModel.inputs, outputs=image_output)

            text_input = Input(
                shape=(100,), name="text_input"
            )  # Example input shape for text
            text_output = text_models["CNN1D"].layers[0](text_input)
            text_output = text_models["CNN1D"].layers[1](text_output)
            text_output = text_models["CNN1D"].layers[2](text_output)
            text_output = text_models["CNN1D"].layers[3](text_output)
            text_output = text_models["CNN1D"].layers[4](text_output)
            text_output = text_models["CNN1D"].layers[5](text_output)
            TextFeatures = text_models["CNN1D"].layers[6](text_output)
            text_output = text_models["CNN1D"].layers[7](TextFeatures)

            TextModel = keras.Model(inputs=text_input, outputs=text_output)

            combined_features = Concatenate()([TextFeatures, ImgFeatures])

            # Add some fully connected layers
            x = Dense(64, activation="relu")(combined_features)
            x = Dense(32, activation="relu")(x)
            final_output = Dense(7, activation="softmax")(
                x
            )  # Assuming 7 classes for classification

            # Define the combined model
            combined_model = Model(
                inputs=[PreImgModel.inputs, TextModel.inputs], outputs=final_output
            )

            # model compilation
            combined_model.compile(
                optimizer=keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            with mlflow.start_run(run_name=f"{key}-CNN1D", nested=True) as run:
                # log model parametes
                mlflow.log_param("Optimizer", "Adam")
                mlflow.log_param("Loss", "Sparse_Categorical_Crossentropy")
                mlflow.log_param("Batch_size", 32)
                mlflow.log_param("Epochs", 20)

                # Define the number of steps per epoch and validation steps
                steps_per_epoch = len(train_x["image_id"]) // batch_size
                validation_steps = len(test_x["image_id"]) // batch_size

                # Fit the combined model using the generators
                history = combined_model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_gen,
                    validation_steps=validation_steps,
                    epochs=20,
                    callbacks=[early_stopping],
                )

                # Log metrics
                for epoch in range(20):
                    mlflow.log_metric(
                        "train_accuracy", history.history["accuracy"][epoch], step=epoch
                    )
                    mlflow.log_metric(
                        "val_accuracy", history.history["val_accuracy"][epoch], step=epoch
                    )
                    mlflow.log_metric(
                        "train_loss", history.history["loss"][epoch], step=epoch
                    )
                    mlflow.log_metric(
                        "val_loss", history.history["val_loss"][epoch], step=epoch
                    )

                # log the model
                # mlflow.keras.log_model(model,f"Keras-model-{key}")
        except Exception as e:
            print(e)
