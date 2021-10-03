import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.initializers import Identity
import numpy as np

def apply_dropout(x, p=0.5, use_mc_dropout = False):
    if use_mc_dropout:
        return Dropout(p)(x, training=True)
    else:
        return Dropout(p)(x)

class MC_Dropout():
    def __init__(self, input_shape, dropout_p1, dropout_p2, use_mc_dropout=False):
        inputs = tf.keras.Input(input_shape)
        x = Dense(50, activation='relu', input_shape=(input_shape,), kernel_initializer=Identity())(inputs)
        x = apply_dropout(x, p=dropout_p1, use_mc_dropout=use_mc_dropout)
        x = Dense(50, activation='relu', kernel_initializer=Identity())(x)
        x = apply_dropout(x, p=dropout_p2, use_mc_dropout=use_mc_dropout)
        x = Dense(10, activation=None, kernel_initializer=Identity())(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        outputs = Dense(1, kernel_initializer=Identity())(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss = 'mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.05))
        self.model = model

    def fit(self, *args, **kwargs):

        history = self.model.fit(*args, **kwargs)
        return history

    def predict(self, X_test):
        mc_predicts = np.hstack([self.model.predict(X_test) for _ in range(10)])
        mean_prediction = np.mean(mc_predicts, 1)[:, None]
        return mean_prediction

    def predict_full(self, X_test):
        mc_predicts = np.hstack([self.model.predict(X_test) for _ in range(10)])
        mean_prediction = np.mean(mc_predicts, 1)[:, None]
        var_prediction = np.var(mc_predicts, 1)[:, None]
        return mean_prediction, var_prediction
#
# # mc model
# def get_dropout(input_tensor, p=0.5, mc=False):
#     if mc:
#         return Dropout(p)(input_tensor, training=True)
#     else:
#         return Dropout(p)(input_tensor)
#
# def get_model(mc=False, act="relu"):
#     inp = tf.keras.Input(input_shape)
#     x = Conv2D(32, kernel_size=(3, 3), activation=act)(inp)
#     x = Conv2D(64, kernel_size=(3, 3), activation=act)(x)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = get_dropout(x, p=0.25, mc=mc)
#     x = Flatten()(x)
#     x = Dense(128, activation=act)(x)
#     x = get_dropout(x, p=0.5, mc=mc)
#     out = Dense(num_classes, activation='softmax')(x)
#
#     model = Model(inputs=inp, outputs=out)
#
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adam(0.05),
#                   metrics=['accuracy'])
#     return model
#
#
# mc_model = get_model(mc=True, act="relu")
# h_mc = mc_model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=10,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
#
# import tqdm
#
# mc_predictions = []
# for i in tqdm.tqdm(range(500)):
#     y_p = mc_model.predict(x_test, batch_size=1000)
#     mc_predictions.append(y_p)
#

