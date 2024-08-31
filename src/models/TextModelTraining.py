import mlflow
import numpy as np
from src.models.ImageModelTraining import train_ds
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Embedding,
    GlobalMaxPooling1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


from src.data.make_text_data import data_preprocssing

sorted_data = data_preprocssing()


labels = sorted_data["dx"].values
text = [" ".join(val) for val in sorted_data["no_stopwords"].values]

labels_index = {val: i for i, val in enumerate(train_ds.class_names)}
labels = [labels_index[val] for val in labels]
len(set(" ".join(text).split()))


train_set, test_set, train_labels, test_labels = train_test_split(
    text, labels, test_size=0.05, random_state=42
)
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(train_set)

train_sequences = tokenizer.texts_to_sequences(train_set)
test_sequences = tokenizer.texts_to_sequences(test_set)

word_index = tokenizer.word_index

print("Found %s unique tokens." % len(word_index))
# converting this to sqequences to be fed into neural netwok
train_x = pad_sequences(train_sequences, maxlen=100)
test_x = pad_sequences(test_sequences, maxlen=100)

train_y = to_categorical(np.asarray(train_labels))
test_y = to_categorical(np.asarray(test_labels))


text_models = {}

text_model = Sequential()
text_model.add(Embedding(100, 64))
text_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
text_model.add(Dense(7, activation="softmax"))

text_models["LSTM"] = text_model


text_model = Sequential()
text_model.add(Embedding(100, 64))
text_model.add(Conv1D(64, 5, activation="relu"))
text_model.add(MaxPooling1D(5))

text_model.add(Conv1D(64, 5, activation="relu"))
text_model.add(MaxPooling1D(5))

text_model.add(GlobalMaxPooling1D())

text_model.add(Dense(64, activation="relu"))
text_model.add(Dense(7, activation="softmax"))

text_models["CNN1D"] = text_model


def StartTextTraining():
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Text Data Experiments")

    for key in text_models.keys():
        text_models[key].compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        try:
            with mlflow.start_run(run_name=key, nested=True) as run:
                # log model parametes
                mlflow.log_param("Optimizer", "Adam")
                mlflow.log_param("Loss", "Categorical_Crossentropy")
                mlflow.log_param("Batch_size", 32)
                mlflow.log_param("Epochs", 50)

                # fit the model
                history = text_models[key].fit(
                    train_x,
                    train_y,
                    batch_size=32,
                    epochs=50,
                    validation_data=(test_x, test_y),
                )

                # Log metrics
                for epoch in range(50):
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
