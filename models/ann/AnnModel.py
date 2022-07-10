import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow.keras.backend as kb
import keras_tuner as kt

from models.xgb.XGB import print_evaluation_metrics
from preprocess.main import load_data, prepare_df_for_learning, get_x_y


def preprocess(x, y):
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.int64)

  return x, y

def create_dataset(xs, ys, n_classes=2):
  ys1 = tf.reshape(tf.one_hot(ys, depth=n_classes), [ys.shape[0], n_classes])
  return tf.data.Dataset.from_tensor_slices((xs, ys1)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)

def custom_loss(y_actual,y_pred):
 #   custom_loss= 200 if (y_actual < y_pred) else 0
    y_actual = tf.cast( y_actual, tf.float32)
    y_pred = tf.cast( y_pred, tf.float32)
    return kb.square(y_actual-y_pred)


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Reshape(target_shape=(74,), input_shape=(74,)))
    for i in range(1, hp.Int("num_layers", 2, 4)):
         model.add(keras.layers.Dense(
             units=hp.Int("units_" + str(i), min_value=4, max_value=68, step=16),
             activation='relu'))

    model.add(keras.layers.Dense(units=2, activation='softmax'))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=custom_loss,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    NUM_EPOCHS = 10
    df = load_data("ctr_dataset_train")
    df = prepare_df_for_learning(df)
    x_train, x_test, y_train, y_test, x_train_resampled, y_train_resampled, x_val, y_val = get_x_y(df)
    train_dataset = create_dataset(x_train, y_train)
    val_dataset = create_dataset(x_val, y_val)
    test_dataset = create_dataset(x_test, y_test)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner = kt.Hyperband(build_model,
                         objective="val_accuracy",
                         max_epochs=20,
                         factor=3,
                         hyperband_iterations=10,
                         directory="kt_dir",
                         project_name="kt_hyperband", )
    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early], verbose=2)

    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(
        train_dataset.repeat(),
        epochs=NUM_EPOCHS,
        steps_per_epoch=500,
        validation_data=val_dataset.repeat(),
        callbacks=[stop_early],
        validation_steps=2
    )

    train_predictions = model.predict(train_dataset)
    val_predictions = model.predict(val_dataset)
    test_predictions = model.predict(val_dataset)
    test_predictions = [numpy.argmax(i) for i in test_predictions]
    val_predictions = [numpy.argmax(i) for i in val_predictions]
    train_predictions = [numpy.argmax(i) for i in train_predictions]

    print_evaluation_metrics("Ann", model, test_predictions, train_predictions, x_train, x_test, y_train, y_test, isAnn=True)
