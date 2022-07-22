import numpy
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn import utils

from models.xgb.XGB import print_evaluation_metrics
from preprocess.main import load_data, prepare_df_for_learning, get_x_y
tf.compat.v1.disable_eager_execution()




def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Reshape(target_shape=(72,), input_shape=(72,)))
    for i in range(1, hp.Int("num_layers", 2, 6)):
        model.add(keras.layers.Dense(
            units=hp.Int("units_" + str(i), min_value=8, max_value=64, step=4),
            activation='relu'))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    NUM_EPOCHS = 30
    df = load_data("ctr_dataset_train")
    df = prepare_df_for_learning(df)
    x_train, x_test, y_train, y_test, x_train_resampled, y_train_resampled, x_val, y_val = get_x_y(df)
    x_train_resampled = x_train_resampled.to_numpy()[:,0:72]
    y_train_resampled = y_train_resampled.astype(numpy.float).to_numpy()
    x_train_resampled, y_train_resampled = utils.shuffle(x_train_resampled, y_train_resampled)

    x_test = x_test.to_numpy()[:,0:72]
    x_train = x_train.to_numpy()[:,0:72]
    x_val = x_val.to_numpy()[:,0:72]

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner = kt.Hyperband(build_model,
                         objective="val_accuracy",
                         max_epochs=20,
                         factor=3,
                         hyperband_iterations=10,
                         directory="kt_dir",
                         project_name="kt_hyperband", )
    tuner.search_space_summary()
    tuner.search(x_train_resampled, y_train_resampled, epochs=NUM_EPOCHS, validation_split=0.2, callbacks=[stop_early], verbose=2)

    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(
        x_train,
        y_train,
        shuffle=True,
        epochs = 500,
        batch_size = 64
    )

    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)
    test_predictions = [numpy.round(i) for i in test_predictions]
    val_predictions = [numpy.round(i) for i in val_predictions]
    train_predictions = [numpy.round(i) for i in train_predictions]

    print_evaluation_metrics("Ann", model, test_predictions, train_predictions, x_train, x_test, y_train, y_test, isAnn=True)
