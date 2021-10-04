import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, concatenate
from tensorflow.keras.initializers import Identity
import numpy as np
import shap
from shap import DeepExplainer, KernelExplainer
import copy

class DeepEnsemble(object):

    def __init__(self, input_shape, ensemble_size, heteroscedastic=True):
        self.ensemble_size = ensemble_size
        inputs = tf.keras.Input(input_shape)
        x = Dense(200, input_shape=(input_shape,))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        # x = Dense(10, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(10, activation=None, kernel_initializer=Identity())(x)
        # x = BatchNormalization()(x)
        # x = ReLU()(x)
        mean = Dense(1)(x)
        self.heteroscedastic = heteroscedastic

        if self.heteroscedastic:
            log_var = Dense(1)(x)
            outputs = concatenate([mean, log_var])

            def heteroscedastic_loss(true, pred):
                mean = pred[:, 0]
                log_var = pred[:, 1]
                precision = tf.exp(-log_var)
                log_likelihood = tf.reduce_sum(- 0.5 * precision * (true - mean) ** 2 - 0.5 * log_var, 1)
                return - tf.reduce_mean(log_likelihood, 0)

            loss = heteroscedastic_loss

        else:
            outputs = mean
            loss = "mean_squared_error"

        model = Model(inputs=inputs, outputs=outputs)

        self.model_ensemble = []
        for i in range(self.ensemble_size):
            model_i = tf.keras.models.clone_model(model)
            model_i.compile(loss = loss, optimizer=tf.keras.optimizers.Adam(0.01))
            self.model_ensemble.append(model_i)
        # self.model_ensemble = [copy.deepcopy(model) for _ in range(self.ensemble_size)]

    def fit(self, *args, **kwargs):
        ensemble_histories = []
        for i in range(self.ensemble_size):
            history = self.model_ensemble[i].fit(*args, **kwargs)
            ensemble_histories.append(history)
        return ensemble_histories

    def predict(self, X_test):
        mc_predicts = np.hstack([self.model.predict(X_test)[:,0:1] for _ in range(10)])
        mean_prediction = np.mean(mc_predicts, 1)[:, None]
        return mean_prediction

    def predict_full(self, X_test):
        if self.heteroscedastic:
            baselearner_mean_predicts = []
            baselearner_sec_moments_predicts = []
            for i in range(self.ensemble_size):
                preds = self.model_ensemble[i].predict(X_test)
                pred_mean = preds[:,0:1]
                pred_log_var = preds[:,1:2]
                baselearner_mean_predicts.append(pred_mean)
                baselearner_sec_moments_predicts.append(tf.exp(pred_log_var)+pred_mean**2)

            ensemble_mean = np.mean(np.hstack(baselearner_mean_predicts), 1)[:, None]
            ensemble_var = np.mean(np.hstack(baselearner_sec_moments_predicts), 1)[:, None] - ensemble_mean**2
        else:
            assert False
        return ensemble_mean, ensemble_var

    def compute_shap_values(self, X_test, background):


        shap.initjs()
        shap_value_list =[]
        for _ in range(10):
            e = shap.DeepExplainer(self.model, background)
            shap_values = e.shap_values(X_test)
            shap_value_list.append(shap_values)

        return shap_value_list

