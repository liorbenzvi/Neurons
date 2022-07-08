import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow.keras.backend as kb

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



if __name__ == '__main__':
    df = load_data("ctr_dataset_train")
    df = prepare_df_for_learning(df)
    x_train, x_test, y_train, y_test, x_train_resampled, y_train_resampled, x_val, y_val = get_x_y(df)
    train_dataset = create_dataset(x_train, y_train)
    val_dataset = create_dataset(x_val, y_val)
    test_dataset = create_dataset(x_test, y_test)
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(74,), input_shape=(74,)),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=2, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss=custom_loss,
                  metrics=['accuracy'])

    history = model.fit(
        train_dataset.repeat(),
        epochs=10,
        steps_per_epoch=500,
        validation_data=val_dataset.repeat(),
        validation_steps=2
    )

    train_predictions = model.predict(train_dataset)
    val_predictions = model.predict(val_dataset)
    test_predictions = model.predict(val_dataset)

    #print_evaluation_metrics(model, test_predictions, train_predictions, x_train, x_test, y_train, y_test)
