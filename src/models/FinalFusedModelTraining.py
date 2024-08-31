
import keras
import mlflow
from FeatureFusion import batch_size, test_gen, train_gen
from ImageModelTraining import pretrained_models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from TextModelTraining import test_x, text_models, train_x

PreImgModel = keras.applications.MobileNetV2(
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

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Final Fused Model Experiment")

with mlflow.start_run(run_name="Final Fused Model", nested=True) as run:
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

# save the model
combined_model.save("./FinalFusedModel.h5")