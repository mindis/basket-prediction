import pandas as pd
import numpy as np
import warnings
import argparse
import os

from model import *
from prepare import *

warnings.simplefilter(action='ignore', category=FutureWarning)

I = 2000
T = 49
J = 40


class Experimenter:
    def __init__(self, dataset_folder, prediction_folder):
        self.dataset_folder = dataset_folder
        if not os.path.isdir(dataset_folder):
            raise Exception('Dataset folder does not exist!')

        self.prediction_folder = prediction_folder
        if not os.path.isdir(prediction_folder):
            os.mkdir(prediction_folder)

    def baseline_logistic(self, window_size, split_time, concat_frequencies=False):
        x_test, x_train, y_test, y_train, x_pred = self.get_datasets(split_time, window_size)

        if not concat_frequencies:
            x_train = x_train[:, :window_size]
            x_test = x_test[:, :window_size]

        loss, y_pred = logistic_model(x_test, x_train, y_test, y_train)

    def conv1d_predict(self, window_size, split_time):
        x_train, y_train, x_test, y_test, x_pred = self.get_datasets(split_time, window_size)
        prepare_global_features(window_size, split_time, concat_time=False, concat_nmf=True, concat_cnn=False,
                                concat_cnn_lstm=False, concat_multi_lstm=False)
        loss, y_pred = cnn_model_pred(x_train, y_train, x_test, y_test, x_pred, window_size, split_time)

        product_ids_pred = np.expand_dims(np.array([j for j in range(40) for i in range(2000)]), axis=-1)
        user_ids_pred = np.expand_dims(np.array([i for j in range(40) for i in range(2000)]), axis=-1)
        df_pred = pd.DataFrame(user_ids_pred, columns=['i'])
        df_pred['j'] = product_ids_pred
        df_pred['prediction'] = y_pred

        df_pred.sort_values(by=['i', 'j'], inplace=True)
        df_pred.to_csv('predictions/prediction.csv', index=False)

    def learn_embed_and_gbm_predict(self, window_size, split_time):
        x_train, y_train, x_test, y_test, x_pred = self.get_datasets(split_time, window_size)
        x_multi_train, y_multi_train, x_multi_test, y_multi_test = self.get_multi_series(split_time, window_size)

        # create cnn embeddings
        cnn_model_embed(x_train, y_train, x_test, y_test, window_size)
        cnn_lstm_embed(x_train, y_train, x_test, y_test, window_size)
        lstm_multivariate_embed(x_multi_train, y_multi_train, x_multi_test, y_multi_test, window_size, J)

        # create nmf embed
        nmf()

        # prepare features
        prepare_features(window_size, split_time)

        # gmb
        self.gbm_predict(window_size, split_time)

    @staticmethod
    def gbm_predict(window_size, split_time):
        # prepare features
        prepare_global_features(window_size, split_time, concat_time=True, concat_nmf=True,
                                concat_cnn=True, concat_cnn_lstm=False, concat_multi_lstm=False)
        # gmb
        combined_gbm(window_size, split_time)

    def get_datasets(self, split_time, window_size):
        data_path = os.path.join(self.dataset_folder, 'x_windowed_train_w%d_s%d.npy' % (window_size, split_time))
        if not os.path.exists(data_path):
            product_tensor, discount_tensor = prepare_tensors(self.dataset_folder)
            prepare_windowed_dataset(product_tensor, discount_tensor, window_size, split_time)

        x_train = np.load('data/x_windowed_train_w%d_s%d.npy' % (window_size, split_time))
        y_train = np.load('data/y_windowed_train_w%d_s%d.npy' % (window_size, split_time))
        x_test = np.load('data/x_windowed_test_w%d_s%d.npy' % (window_size, split_time))
        y_test = np.load('data/y_windowed_test_w%d_s%d.npy' % (window_size, split_time))
        x_pred = np.load('data/x_windowed_prediction_w%d_s%d.npy' % (window_size, split_time))
        return x_train, y_train, x_test, y_test, x_pred

    def get_multi_series(self, split_time, window_size):
        data_path = os.path.join(self.dataset_folder, 'x_multi_series_train_w%d_s%d.npy' % (window_size, split_time))
        if not os.path.exists(data_path):
            product_tensor, discount_tensor = prepare_tensors()
            prepare_multivariate_series(product_tensor, window_size, split_time)

        x_train = np.load('data/x_multi_series_train_w%d_s%d.npy' % (window_size, split_time))
        y_train = np.load('data/y_multi_series_train_w%d_s%d.npy' % (window_size, split_time))
        x_test = np.load('data/x_multi_series_test_w%d_s%d.npy' % (window_size, split_time))
        y_test = np.load('data/y_multi_series_test_w%d_s%d.npy' % (window_size, split_time))
        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, required=True)  # 'data'
    parser.add_argument('--prediction-folder', type=str, required=True)  # model.pkl
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--split-time', type=int, default=44)
    args = parser.parse_args()

    experimenter = Experimenter(dataset_folder=args.dataset_folder,
                                prediction_folder=args.prediction_folder)

    if args.run == 'baseline_logistic':
        experimenter.baseline_logistic(window_size=args.window_size,
                                       split_time=args.split_time)
    elif args.run == 'conv1d_predict':
        experimenter.conv1d_predict(window_size=args.window_size,
                                    split_time=args.split_time)
    elif args.run == 'learn_embed_and_gbm_predict':
        experimenter.learn_embed_and_gbm_predict(window_size=args.window_size,
                                           split_time=args.split_time)
    elif args.run == 'gbm_predict':
        experimenter.gbm_predict(window_size=args.window_size,
                                 split_time=args.split_time)
    else:
        raise Exception("Invalid experiment name!")
