import keras
import mlflow
import tensorflow as tf

# keras datalaoder
train_ds = tf.keras.utils.image_dataset_from_directory(
    "./HAM_10000_Dataset",
    validation_split=0.05,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "./HAM_10000_Dataset",
    validation_split=0.05,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
)

pretrained_models = {
    "VGG16": keras.applications.VGG16,
    "ConvNeXtBase": keras.applications.ConvNeXtBase,
    "ConvNeXtLarge": keras.applications.ConvNeXtLarge,
    "ConvNeXtSmall": keras.applications.ConvNeXtSmall,
    "ConvNeXtTiny": keras.applications.ConvNeXtTiny,
    "DenseNet121": keras.applications.DenseNet121,
    "DenseNet169": keras.applications.DenseNet169,
    "DenseNet201": keras.applications.DenseNet201,
    "EfficientNetB0": keras.applications.EfficientNetB0,
    "EfficientNetB1": keras.applications.EfficientNetB1,
    "EfficientNetB2": keras.applications.EfficientNetB2,
    "EfficientNetB3": keras.applications.EfficientNetB3,
    "EfficientNetB4": keras.applications.EfficientNetB4,
    "EfficientNetB5": keras.applications.EfficientNetB5,
    "EfficientNetB6": keras.applications.EfficientNetB6,
    "EfficientNetB7": keras.applications.EfficientNetB7,
    "EfficientNetV2B0": keras.applications.EfficientNetV2B0,
    "EfficientNetV2B1": keras.applications.EfficientNetV2B1,
    "EfficientNetV2B2": keras.applications.EfficientNetV2B2,
    "EfficientNetV2B3": keras.applications.EfficientNetV2B3,
    "EfficientNetV2L": keras.applications.EfficientNetV2L,
    "EfficientNetV2M": keras.applications.EfficientNetV2M,
    "EfficientNetV2S": keras.applications.EfficientNetV2S,
    "InceptionResNetV2": keras.applications.InceptionResNetV2,
    "InceptionV3": keras.applications.InceptionV3,
    "MobileNet": keras.applications.MobileNet,
    "MobileNetV2": keras.applications.MobileNetV2,
    "MobileNetV3Large": keras.applications.MobileNetV3Large,
    "MobileNetV3Small": keras.applications.MobileNetV3Small,
    "NASNetLarge": keras.applications.NASNetLarge,
    "NASNetMobile": keras.applications.NASNetMobile,
    "ResNet101": keras.applications.ResNet101,
    "ResNet101V2": keras.applications.ResNet101V2,
    "ResNet152": keras.applications.ResNet152,
    "ResNet152V2": keras.applications.ResNet152V2,
    "ResNet50": keras.applications.ResNet50,
    "ResNet50V2": keras.applications.ResNet50V2,
    "VGG19": keras.applications.VGG19,
    "Xception": keras.applications.Xception,
}


print(f"Total Number of Pretrained Models: {len(pretrained_models.keys())}")

def StartImageTraining():
        
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Image Data Experiments")

    for key in pretrained_models.keys():
        try:
            print("=" * 40, f"{key} Model Training Started", "=" * 40)
            model = pretrained_models[key](include_top=False, input_shape=(224, 224, 3))
            for layer in model.layers:
                layer.trainable = False
            flat1 = keras.layers.Flatten()(model.layers[-1].output)
            class1 = keras.layers.Dense(512, activation="relu")(flat1)

            output = keras.layers.Dense(7, activation="softmax")(class1)
            # # define new model
            model = keras.Model(inputs=model.inputs, outputs=output)
            # model compilation
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            with mlflow.start_run(run_name=key, nested=True) as run:
                # log model parametes
                mlflow.log_param("Optimizer", "Adam")
                mlflow.log_param("Loss", "Sparse_Categorical_Crossentropy")
                mlflow.log_param("Batch_size", 32)
                mlflow.log_param("Epochs", 50)

                # fit the model
                history = model.fit(train_ds, epochs=50, validation_data=val_ds)

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
