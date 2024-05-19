# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt

from keras.models import load_model
import matplotlib.pyplot
import tensorflow as tf
from sklearn.metrics import *
from keras.callbacks import *
from keras.losses import *
from keras import optimizers
import seaborn as sns
from keras.layers import Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from app.constants import app_constants
import asyncio
import cv2
import imutils
from tensorflow.keras.preprocessing import image
from app.models import Reports, PredictionLogs

matplotlib.use("agg")


class GradCAM:

    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output.shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output],
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


class CNN:

    def __init__(self):
        pass

    async def plot_async_matrix(self, cm, class_names):

        instance_matplot = matplotlib.pyplot
        instance_matplot.clf()
        instance_matplot.figure(figsize=(len(class_names), len(class_names)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 8},
        )
        instance_matplot.xticks(fontsize=6)
        instance_matplot.yticks(fontsize=6)
        instance_matplot.title("Confusion Matrix")
        instance_matplot.xlabel("Predicted Labels")
        instance_matplot.ylabel("True Labels")
        instance_matplot.savefig("media/confusion_matrix.png")

        print("Done plotting and saving confusion matrix.")
        instance_matplot.clf()

        return True

    async def plot_async_history(self, history):

        instance_matplot = matplotlib.pyplot
        instance_matplot.clf()
        instance_matplot.plot(history.history["accuracy"])
        instance_matplot.plot(history.history["val_accuracy"])
        instance_matplot.title("Model Accuracy")
        instance_matplot.ylabel("Accuracy")
        instance_matplot.xlabel("Epoch")
        instance_matplot.legend(["Train", "Validation"], loc="upper left")
        instance_matplot.savefig("media/accuracy_plot.png")

        instance_matplot = matplotlib.pyplot
        instance_matplot.clf()
        instance_matplot.plot(history.history["loss"])
        instance_matplot.plot(history.history["val_loss"])
        instance_matplot.title("Model Loss")
        instance_matplot.ylabel("Loss")
        instance_matplot.xlabel("Epoch")
        instance_matplot.legend(["Train", "Validation"], loc="upper left")
        instance_matplot.savefig("media/loss_plot.png")
        print("Done plotting and saving history.")
        instance_matplot.clf()

        return True

    async def execute_asyncs(self, history, cm, class_names):
        """
        A single method to execute asynchronous functions in one.
        to be called in a synchronous function.
        """
        val = asyncio.create_task(self.plot_async_history(history))
        conf_matrix = asyncio.create_task(self.plot_async_matrix(cm, class_names))

        await val
        await conf_matrix

    def train(self):

        resnet_architecture = ResNet152V2(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

        for layer in resnet_architecture.layers[:-3]:
            layer.trainable = False
        x = Flatten()(resnet_architecture.output)
        x = Dense(256, activation="relu")(x)
        x = predictions = Dense(3, activation="softmax")(x)
        model = Model(inputs=resnet_architecture.input, outputs=predictions)
        model.summary()

        opi = optimizers.Adam(learning_rate=0.002)
        model.compile(
            optimizer=opi, loss=categorical_crossentropy, metrics=["accuracy"]
        )

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255, preprocessing_function=preprocess_input
        )

        valid_datagen = ImageDataGenerator(
            rescale=1.0 / 255, preprocessing_function=preprocess_input
        )

        batch_size = 16

        # Train generator with 80% of the data

        train_generator = train_datagen.flow_from_directory(
            app_constants.TRAIN_LOCATION,  # Point to the single folder containing all data
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        # Validation generator with 20% of the data
        valid_generator = valid_datagen.flow_from_directory(
            app_constants.VALIDATION_LOCATION,  # Point to the single folder containing all data
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )
        class_indices = train_generator.class_indices
        class_names = list(class_indices.keys())

        if True:

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,  # Factor by which the learning rate will be reduced. New_lr = lr * factor
                patience=3,  # Number of epochs with no improvement after which learning rate will be reduced.
                verbose=1,  # 0: quiet, 1: update messages
                mode="min",  # In mode 'min', lr will be reduced when the quantity monitored has stopped decreasing
                min_delta=0.001,  # Minimum change in the monitored quantity to qualify as an improvement
                cooldown=0,  # Number of epochs to wait before resuming normal operation after lr has been reduced.
                min_lr=0,  # Lower bound on the learning rate.
            )

            # checkpoint = ModelCheckpoint(
            #     app_constants.MODEL_NAME,
            #     monitor="val_acc",
            #     verbose=1,
            #     save_best_only=True,
            #     #save_weights_only=False,
            #     mode="auto",
            #     period=1,
            # )

            checkpoint = ModelCheckpoint(
                filepath=app_constants.MODEL_NAME,  # Path where the model will be saved
                monitor="val_accuracy",  # Metric to monitor
                save_best_only=True,  # Save only the best model
                mode="max",  # Mode: 'min' for minimizing the monitored metric
                verbose=1,  # Verbosity mode
            )

            early = EarlyStopping(
                monitor="val_accuracy", min_delta=0, patience=20, verbose=1, mode="auto"
            )

            history = model.fit_generator(
                steps_per_epoch=len(train_generator),
                generator=train_generator,
                validation_data=valid_generator,
                validation_steps=len(valid_generator),
                epochs=app_constants.EPOCHS,
                callbacks=[checkpoint, early, reduce_lr],
            )

            model = load_model(app_constants.MODEL_NAME)

            # model.save(app_constants.MODEL_NAME)
            evaluation = model.evaluate(valid_generator)
            print("Test Accuracy:", evaluation[1], evaluation)

            preds = model.predict(valid_generator)
            predicted_classes = np.argmax(preds, axis=1)

            # Get the actual labels
            true_classes = valid_generator.classes

            # Calculate the confusion matrix
            conf_matrix = confusion_matrix(true_classes, predicted_classes)

            print(conf_matrix)

            self.view_metrics(true_classes, preds, predicted_classes, class_names)

            accuracy = accuracy_score(true_classes, predicted_classes)
            precision = precision_score(
                true_classes, predicted_classes, average="weighted"
            )
            recall = recall_score(true_classes, predicted_classes, average="weighted")
            f1 = f1_score(true_classes, predicted_classes, average="weighted")
            print(accuracy, precision, recall, f1)
            self.update_metrics_database(accuracy, precision, recall, f1)
            asyncio.run(self.execute_asyncs(history, conf_matrix, class_names))

    def update_metrics_database(self, acc, prec, recall, f1):

        acc = round((float(acc) * 100), 2)
        prec = round((float(prec) * 100), 2)
        recall = round((float(recall) * 100), 2)
        f1 = round((float(f1) * 100), 2)

        try:
            report_length = Reports.objects.all()
            if len(report_length) < 1:
                Reports.objects.create(
                    accuracy=acc, precision=prec, recall=recall, f1_score=f1
                )
            else:
                Reports.objects.all().update(
                    accuracy=acc, precision=prec, recall=recall, f1_score=f1
                )
        except Exception as e:
            print(e)

    def view_metrics(self, y_true, y_pred, y_pred_classes, class_names):

        indexes = 0
        PredictionLogs.objects.all().delete()
        for correct_answers in y_true:
            PredictionLogs.objects.create(
                correct_answer= str(class_names[int(str(correct_answers))]).upper(),
                forecasted= str(class_names[int(str(y_pred_classes[indexes]))]).upper(),
                remarks=(
                    "CORRECT"
                    if class_names[int(str(correct_answers))] == class_names[int(str(y_pred_classes[indexes]))]
                    else "INCORRECT"
                ),
            )
            indexes = indexes + 1

    def predict_image_v2(self, np_frame):

        saved_model = load_model(app_constants.MODEL_NAME)
        resized_image = cv2.resize(np_frame, (224, 224))
        img_array = image.img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        class_names = ["high", "low", "medium"]

        print("Following is our prediction:")
        prediction = saved_model.predict(img_array)
        highestProb = np.argmax(prediction)

        d = prediction.flatten()
        j = d.max()
        for index, item in enumerate(d):
            if item == j:
                predicted_class_name = class_names[index]

        return np_frame, str(predicted_class_name), str(round((j * 100), 2))

        cam = GradCAM(saved_model, highestProb)
        prep = preprocess_input(np.expand_dims(np_frame, axis=0))
        heatmap = cam.compute_heatmap(prep)
        heatmap = cv2.resize(heatmap, (224, 224))

        (heatmap, prediction) = cam.overlay_heatmap(heatmap, np_frame, alpha=0.5)

        font_size = 0.5

        # Show the image.
        cv2.putText(
            np_frame,
            f"Predicted: {class_names[highestProb]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 255),
            2,
        )

        output_stacked = np.vstack([np_frame, heatmap, prediction])
        output_stacked = imutils.resize(prediction, height=700)

        cv2.imshow("window_stack", output_stacked)
        cv2.waitKey(0)

        return np_frame, str(predicted_class_name), str(round((j * 100), 2))

    def predict_image(self, np_frame):
        # Resize the input numpy array to (224, 224) if needed
        if np_frame.shape[:2] != (224, 224):
            np_frame = cv2.resize(np_frame, (224, 224))

        class_names = ["high", "low", "medium"]

        list_of_percentage = []
        iterations = 0
        saved_model = load_model(app_constants.MODEL_NAME)
        output = saved_model.predict(np.expand_dims(np_frame, axis=0))

        for x in output[0]:
            list_of_percentage.append(
                str(round((x * 100), 2)) + ":" + class_names[iterations]
            )
            print(round((x * 100), 2))
            iterations = iterations + 1

        highestProb = np.argmax(output)
        print(str(class_names[highestProb]) + ": " + str(max(output[0]) * 100))

        # return np_frame, str(class_names[highestProb]).upper(), str(max(output[0]) * 100)

        cam = GradCAM(saved_model, highestProb)
        prep = preprocess_input(np.expand_dims(np_frame, axis=0))
        heatmap = cam.compute_heatmap(prep)
        heatmap = cv2.resize(heatmap, (224, 224))
        (heatmap, output) = cam.overlay_heatmap(heatmap, np_frame, alpha=0.5)

        font_size = 0.5

        # Show the image.
        cv2.putText(
            np_frame,
            f"Predicted: {class_names[highestProb]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 255),
            2,
        )

        # cv2.imshow("Image Predicting", np_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # output = np.vstack([np_frame, heatmap, output])
        # output = imutils.resize(output, height=700)
        # cv2.imshow("Image Heatmap", output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # return np_frame, str(class_names[highestProb]).upper(), str(max(output[0]) * 100)
