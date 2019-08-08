import gc
import numpy as np
import tensorflow as tf
import lightgbm as lgb
import matplotlib.pyplot as plt
import pprint as pp
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import non_negative_factorization
from sklearn.model_selection import train_test_split
from sklearn import metrics


def cnn_model_pred(x_train, y_train, x_test, y_test, x_pred, window_size, split_time, inspect_lr=False):
    global_features_train = np.load('data/global_features_train_w%d_s%d.npy' % (window_size, split_time))
    global_features_test = np.load('data/global_features_test_w%d_s%d.npy' % (window_size, split_time))
    global_features_pred = np.load('data/global_features_pred_w%d_s%d.npy' % (window_size, split_time))

    num_global_feature = global_features_train.shape[1]

    tf.keras.backend.clear_session()

    # purchase freq + discount for j + nmf user embed + nmf prod embed
    others_j = tf.keras.layers.Input(shape=(num_global_feature,), name='others')
    time_window_j = tf.keras.layers.Input(shape=(window_size, 1), name='time')
    conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=window_size,
                                    activation="relu", input_shape=[window_size, 1])(time_window_j)
    flat_1 = tf.keras.layers.Flatten()(conv_1)
    merge_1 = tf.keras.layers.concatenate([flat_1, others_j])
    dense_2 = tf.keras.layers.Dense(50, activation='relu')(merge_1)
    dense_3 = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense_2)
    model = tf.keras.models.Model(inputs=[time_window_j, others_j], outputs=dense_3)

    x_cnn_train = np.expand_dims(x_train[:, :window_size], axis=-1)
    x_cnn_test = np.expand_dims(x_test[:, :window_size], axis=-1)
    x_cnn_pred = np.expand_dims(x_pred[:, :window_size], axis=-1)

    lr_start = 5e-3
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))

    if inspect_lr:
        lr_start = 1e-5
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_start * 10 ** (epoch / 20))
        history = model.fit([x_cnn_train, global_features_train], y_train,
                            validation_data=([x_cnn_test, global_features_test], y_test),
                            shuffle=True, epochs=60, batch_size=512, callbacks=[lr_schedule])
        np.save('predictions/loss.npy', history.history["loss"])
        np.save('predictions/lr.npy', history.history["lr"])
        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([lr_start, 1e-2, 0.05, 0.09])
        # plt.show()
    else:
        history = model.fit([x_cnn_train, global_features_train], y_train,
                            validation_data=([x_cnn_test, global_features_test], y_test),
                            shuffle=True, epochs=60, batch_size=512)

    y_test_pred = model.predict([x_cnn_test, global_features_test])
    y_pred = model.predict([x_cnn_pred, global_features_pred])

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred, pos_label=1)
    print('Expected AUC on test set:', metrics.auc(fpr, tpr))

    # loss = model.evaluate([x_cnn_pred, global_features_pred], y_pred, verbose=0)
    # loss = model.evaluate([x_cnn_test, global_features_test], y_test, verbose=0)
    loss = 0
    return loss, y_pred


def cnn_model_embed(x_train, y_train, x_test, y_test, window_size, inspect_lr=False):

    final_state_layer = 'final_state'

    tf.keras.backend.clear_session()

    time_window_j = tf.keras.layers.Input(shape=(window_size, 1), name='time')
    conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=window_size,
                                    activation="relu", input_shape=[window_size, 1])(time_window_j)
    flat_1 = tf.keras.layers.Flatten()(conv_1)
    dense_2 = tf.keras.layers.Dense(50, activation='relu', name=final_state_layer)(flat_1)
    dense_3 = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense_2)
    model = tf.keras.models.Model(inputs=time_window_j, outputs=dense_3)

    intermediate_model = tf.keras.models.Model(inputs=model.input,
                                               outputs=model.get_layer(final_state_layer).output)

    x_cnn_train = np.expand_dims(x_train[:, :window_size], axis=-1)
    x_cnn_test = np.expand_dims(x_test[:, :window_size], axis=-1)

    lr_start = 1e-3
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))

    if inspect_lr:
        lr_start = 1e-5
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_start * 10 ** (epoch / 20))
        history = model.fit(x_cnn_train, y_train,
                            validation_data=(x_cnn_test, y_test),
                            shuffle=True, epochs=60, batch_size=128, callbacks=[lr_schedule])
        np.save('predictions/loss.npy', history.history["loss"])
        np.save('predictions/lr.npy', history.history["lr"])
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([lr_start, 1e-1, 0.05, 0.09])
        plt.show()
    else:
        history = model.fit(x_cnn_train, y_train,
                            validation_data=(x_cnn_test, y_test),
                            shuffle=True, epochs=10, batch_size=128)

        cnn_features_train = intermediate_model.predict(x_cnn_train)
        cnn_features_test = intermediate_model.predict(x_cnn_test)

        np.save('predictions/cnn_features_train.npy', cnn_features_train)
        np.save('predictions/cnn_features_test.npy', cnn_features_test)


def cnn_lstm_embed(x_train, y_train, x_test, y_test, window_size, inspect_lr=False):
    final_state_layer = 'final_state'

    tf.keras.backend.clear_session()

    # purchase freq + discount for j + nmf user embed + nmf prod embed
    time_window_j = tf.keras.layers.Input(shape=(window_size, 1), name='time')
    conv_l1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal",
                                     activation="relu", input_shape=[None, 1])(time_window_j)
    lstm_l2 = tf.keras.layers.LSTM(40, return_sequences=True)(conv_l1)
    lstm_l3 = tf.keras.layers.LSTM(40)(lstm_l2)
    dense_l4 = tf.keras.layers.Dense(20, activation='relu', name=final_state_layer)(lstm_l3)
    dense_l5 = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense_l4)
    model = tf.keras.models.Model(inputs=time_window_j, outputs=dense_l5)

    intermediate_model = tf.keras.models.Model(inputs=model.input,
                                               outputs=model.get_layer(final_state_layer).output)

    print(model.summary())

    x_cnn_train = np.expand_dims(x_train, axis=-1)
    x_cnn_test = np.expand_dims(x_test, axis=-1)

    lr_start = 5e-4
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))

    if inspect_lr:
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_start * 10 ** (epoch / 10))
        history = model.fit(x_cnn_train, y_train, validation_data=(x_cnn_test, y_test),
                            shuffle=True, epochs=20, batch_size=512, callbacks=[lr_schedule])
        np.save('predictions/loss.npy', history.history["loss"])
        np.save('predictions/lr.npy', history.history["lr"])
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([lr_start, 1e-1, 0.05, 0.09])
        plt.show()
    else:
        history = model.fit(x_cnn_train, y_train, validation_data=(x_cnn_test, y_test),
                            shuffle=True, epochs=5, batch_size=512)

        lstm_features_train = intermediate_model.predict(x_cnn_train)
        lstm_features_test = intermediate_model.predict(x_cnn_test)

        np.save('predictions/lstm_final_states_train.npy', lstm_features_train)
        np.save('predictions/lstm_final_states_test.npy', lstm_features_test)


def lstm_multivariate_embed(x_concat_train, y_concat_train, x_concat_test, y_concat_test, window_size, J,
                            inspect_lr=False):
    final_state_layer = 'final_state'

    tf.keras.backend.clear_session()

    series_multivariate = tf.keras.layers.Input(shape=(window_size, J), name='time')
    lstm_l1 = tf.keras.layers.LSTM(64, input_shape=(window_size, J), return_sequences=True)(series_multivariate)
    lstm_l2 = tf.keras.layers.LSTM(64)(lstm_l1)
    dense_l3 = tf.keras.layers.Dense(20, activation='relu', name=final_state_layer)(lstm_l2)
    dense_l4 = tf.keras.layers.Dense(40, activation='sigmoid', name='output')(dense_l3)
    model = tf.keras.models.Model(inputs=series_multivariate, outputs=dense_l4)

    intermediate_model = tf.keras.models.Model(inputs=model.input,
                                               outputs=model.get_layer(final_state_layer).output)

    print(model.summary())

    lr_start = 1e-3
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
    if inspect_lr:
        lr_start = 1e-5
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: lr_start * 10 ** (epoch / 5))
        history = model.fit(x_concat_train, y_concat_train, epochs=10, batch_size=400, shuffle=True,
                            validation_data=(x_concat_test, y_concat_test), callbacks=[lr_schedule])
        np.save('predictions/loss.npy', history.history["loss"])
        np.save('predictions/lr.npy', history.history["lr"])
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([lr_start, 1e-1, 0, 0.2])
        plt.show()
    else:
        history = model.fit(x_concat_train, y_concat_train, epochs=10, batch_size=512, shuffle=True,
                            validation_data=(x_concat_test, y_concat_test))

        lstm_multi_final_states_train = intermediate_model.predict(x_concat_train)
        lstm_multi_final_states_test = intermediate_model.predict(x_concat_test)
        np.save('predictions/lstm_multi_final_states_train.npy', lstm_multi_final_states_train)
        np.save('predictions/lstm_multi_final_states_test.npy', lstm_multi_final_states_test)


def nmf():
    product_tensor = np.load('data/product_tensor.npy')
    X = product_tensor.sum(1)

    W, H, n_iter = non_negative_factorization(X, n_components=10, max_iter=500, regularization='both',
                                              init='random', solver='mu', beta_loss='kullback-leibler')
    print(n_iter, W.shape, H.shape)
    np.save('predictions/user_embed.npy', W)
    np.save('predictions/prod_embed.npy', H.T)


def combined_gbm(window_size=10, split_time=44):
    feature_names = np.load('data/feature_names.npy')
    global_features_train = np.load('data/global_features_train_w%d_s%d.npy' % (window_size, split_time))
    df_train = pd.DataFrame(global_features_train, columns=feature_names)

    y_train = np.load('data/y_windowed_train_w%d_s%d.npy' % (window_size, split_time))
    df_train['label'] = y_train
    del global_features_train, y_train  # , product_ids_train, user_ids_train
    gc.collect()

    # drop_cols = [i for i in df_train.columns
    #              if i.startswith('multi_rnn_train') or i.startswith('cnn_rnn_train')]
    # i.startswith('cnn_train') or i.startswith('cnn_rnn_train') or i.startswith('nmf') or i.startswith('nmf') or
    # df_train.drop(drop_cols, axis=1, inplace=True)

    df_train, df_val = train_test_split(df_train, train_size=.99)
    # y_train, y_val = df_train['label'].astype(int).astype(float), df_val['label'].astype(int).astype(float)
    # X_train, X_val = df_train.drop('label', axis=1), df_val.drop('label', axis=1)

    y_train = df_train['label'].astype(int).astype(float)
    y_val = np.load('data/y_windowed_test_w%d_s%d.npy' % (window_size, split_time))
    X_train = df_train.drop('label', axis=1)
    X_val = np.load('data/global_features_test_w%d_s%d.npy' % (window_size, split_time))

    del df_train, df_val
    gc.collect()

    d_train = lgb.Dataset(X_train, label=y_train, silent=True)
    d_valid = lgb.Dataset(X_val, label=y_val, silent=True)

    del X_train, X_val, y_train, y_val
    gc.collect()

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'learning_rate': .02,
        'num_leaves': 32,
        'max_depth': 12,
        'feature_fraction': 0.35,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
    }
    rounds = 5000

    valid_sets = [d_train, d_valid]
    valid_names = ['train', 'valid']
    gbdt = lgb.train(params, d_train, rounds, valid_sets=valid_sets, valid_names=valid_names, verbose_eval=20)

    del d_train, d_valid, valid_sets
    gc.collect()

    features = gbdt.feature_name()
    importance = list(gbdt.feature_importance())
    importance = zip(features, importance)
    importance = sorted(importance, key=lambda x: x[1])
    total = sum(j for i, j in importance)
    importance = [(i, float(j) / total) for i, j in importance]
    np.save('predictions/importance.npy', importance)
    # pp.pprint(importance)

    # X_test = np.load('data/global_features_test_w%d_s%d.npy' % (window_size, split_time))
    # df_test = pd.DataFrame(X_test, columns=feature_names)
    # df_test.drop(drop_cols, axis=1, inplace=True)
    # X_test = df_test.values
    # del df_test
    # gc.collect()

    # y_test = np.load('data/y_windowed_test_w%d_s%d.npy' % (window_size, split_time))
    print('Best iteration: ', gbdt.best_iteration)
    y_val_pred = gbdt.predict(X_val, num_iteration=gbdt.best_iteration)
    loss = log_loss(y_val, y_val_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_pred, pos_label=1)
    print('Loss:', loss, 'Expected AUC on test set:', metrics.auc(fpr, tpr))

    return loss, y_val_pred


def logistic_model(x_train, y_train, x_test, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)
    loss = log_loss(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print('Loss:', loss, 'Expected AUC on test set:', metrics.auc(fpr, tpr))
    return loss, y_pred


# Experimental models
def logistic_per_product(J):
    loss = 0
    for j in range(J):
        x_train = np.load('data/product_%d_train_x.npy' % j)
        y_train = np.expand_dims(np.load('data/product_%d_train_y.npy' % j), axis=1)
        x_test = np.load('data/product_%d_test_x.npy' % j)
        y_test = np.expand_dims(np.load('data/product_%d_test_y.npy' % j), axis=1)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_test)
        logistic_loss = log_loss(y_test, y_pred)
        loss += logistic_loss
        print(j, logistic_loss, loss / (j + 1))
    return loss


def lstm_bidirect_per_product(window_size, J):
    sk_loss = 0
    for j in range(1):
        x_train = np.load('data/product_%d_train_x.npy' % j)
        y_train = np.expand_dims(np.load('data/product_%d_train_y.npy' % j), axis=1)
        x_test = np.load('data/product_%d_test_x.npy' % j)
        y_test = np.expand_dims(np.load('data/product_%d_test_y.npy' % j), axis=1)
        x_train = x_train[:window_size]
        x_test = x_test[:window_size]
        tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                   input_shape=[None]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40, return_sequences=True, input_shape=(window_size, 1))),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40)),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        lr_start = 1e-3
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
        history = model.fit(x_train, y_train, epochs=5, batch_size=200, shuffle=True,
                            validation_data=(x_test, y_test))

        # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        #     lambda epoch: lr_start * 10 ** (epoch / 20))
        # history = model.fit(x_train, y_train, epochs=60, batch_size=200, shuffle=True,
        #                     validation_data=(x_test, y_test), callbacks=[lr_schedule])
        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([lr_start, 1e-2, 0, 0.3])
        # plt.show()

        y_pred = model.predict(x_test)
        incremental_loss = log_loss(y_test, y_pred)
        sk_loss += incremental_loss
        print(j, incremental_loss, sk_loss)

    return sk_loss




def lstm_per_product(window_size, J):
    sk_loss = 0
    for j in range(1):
        x_train = np.load('data/product_%d_train_x.npy' % j)
        y_train = np.expand_dims(np.load('data/product_%d_train_y.npy' % j), axis=1)
        x_test = np.load('data/product_%d_test_x.npy' % j)
        y_test = np.expand_dims(np.load('data/product_%d_test_y.npy' % j), axis=1)
        x_train = x_train[:, :window_size]
        x_test = x_test[:, :window_size]
        tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                   input_shape=[None]),
            tf.keras.layers.LSTM(40, return_sequences=True, input_shape=(window_size, 1)),
            tf.keras.layers.LSTM(40),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        print(model.summary())

        lr_start = 1e-2
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
        history = model.fit(x_train, y_train, epochs=5, batch_size=200, shuffle=True,
                            validation_data=(x_test, y_test))

        # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        #     lambda epoch: lr_start * 10 ** (epoch / 20))
        # history = model.fit(x_train, y_train, epochs=60, batch_size=200, shuffle=True,
        #                     validation_data=(x_test, y_test), callbacks=[lr_schedule])
        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([lr_start, 1e-2, 0, 0.3])
        # plt.show()

        y_pred = model.predict(x_test)
        incremental_loss = log_loss(y_test, y_pred)
        sk_loss += incremental_loss
        print(j, incremental_loss, sk_loss)

    return sk_loss


def rnn_per_product(window_size, J):
    sk_loss = 0
    for j in range(1):
        x_train = np.load('data/product_%d_train_x.npy' % j)
        y_train = np.expand_dims(np.load('data/product_%d_train_y.npy' % j), axis=1)
        x_test = np.load('data/product_%d_test_x.npy' % j)
        y_test = np.expand_dims(np.load('data/product_%d_test_y.npy' % j), axis=1)
        x_train = x_train[:, window_size]
        x_test = x_test[:, window_size]
        tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                   input_shape=[None]),
            tf.keras.layers.SimpleRNN(40, return_sequences=True),
            tf.keras.layers.SimpleRNN(40),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        lr_start = 1e-4
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
        history = model.fit(x_train, y_train, epochs=50, batch_size=200, shuffle=True,
                            validation_data=(x_test, y_test))

        # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        #     lambda epoch: lr_start * 10 ** (epoch / 20))
        # history = model.fit(x_train, y_train, epochs=60, batch_size=200, shuffle=True,
        #                     validation_data=(x_test, y_test), callbacks=[lr_schedule])
        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([lr_start, 1e-2, 0, 0.3])
        # plt.show()

        y_pred = model.predict(x_test)
        incremental_loss = log_loss(y_test, y_pred)
        sk_loss += incremental_loss
        print(j, incremental_loss, sk_loss)

    return sk_loss


def small_dnn_per_product(window_size, J):
    sk_loss = 0
    for j in range(J):
        x_train = np.load('data/product_%d_train_x.npy' % j)
        y_train = np.expand_dims(np.load('data/product_%d_train_y.npy' % j), axis=1)
        x_test = np.load('data/product_%d_test_x.npy' % j)
        y_test = np.expand_dims(np.load('data/product_%d_test_y.npy' % j), axis=1)
        tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense((window_size + 2), input_dim=(window_size + 2), activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.01, l1=0.1)),
            # tf.keras.layers.Dense(240, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.01, l1=0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.01, l1=0.01))
        ])

        lr_start = 5e-3
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
        history = model.fit(x_train, y_train, epochs=100, batch_size=200, shuffle=True,
                            validation_data=(x_test, y_test))

        # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        #     lambda epoch: lr_start * 10 ** (epoch / 20))
        # history = model.fit(x_concat_train, y_concat_train, epochs=20, batch_size=200, shuffle=True,
        #                     validation_data=(x_concat_test, y_concat_test), callbacks=[lr_schedule])
        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([lr_start, 1, 0, 100])
        # plt.show()

        y_pred = model.predict(x_test)
        incremental_loss = log_loss(y_test, y_pred)
        sk_loss += incremental_loss
        print(j, incremental_loss, sk_loss)

    return sk_loss


def multi_out_dnn(x_concat_train, y_concat_train, x_concat_test, y_concat_test, window_size, J):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense((window_size + 2) * J, input_dim=(window_size + 2) * J, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.01, l1=0.1)),
        tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.01, l1=0.01)),
        tf.keras.layers.Dense(40, activation='sigmoid',
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.01, l1=0.01))
    ])

    lr_start = 1e-4
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
    history = model.fit(x_concat_train, y_concat_train, epochs=100, batch_size=200, shuffle=True,
                        validation_data=(x_concat_test, y_concat_test))

    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: lr_start * 10 ** (epoch / 20))
    # history = model.fit(x_concat_train, y_concat_train, epochs=20, batch_size=200, shuffle=True,
    #                     validation_data=(x_concat_test, y_concat_test), callbacks=[lr_schedule])
    # plt.semilogx(history.history["lr"], history.history["loss"])
    # plt.axis([lr_start, 1, 0, 100])
    # plt.show()

    nn_loss = model.evaluate(x_concat_test, y_concat_test, verbose=0)
    y_pred = model.predict(x_concat_test)

    sk_loss = 0
    for j in range(J):
        sk_loss += log_loss(y_concat_test[:, j], y_pred[:, j])

    print(sk_loss, nn_loss)
    return sk_loss


def cnn_various_length_combined(x_train, y_train, x_test, y_test, window_size_l, window_size_m, window_size_s, split_time):
    global_features_train = np.load('data/global_features_train%d_s%d.npy' % (window_size_l, split_time))
    global_features_test = np.load('data/global_features_test%d_s%d.npy' % (window_size_l, split_time))

    num_global_feature = global_features_train.shape[1]

    tf.keras.backend.clear_session()

    # purchase freq + discount for j + nmf user embed + nmf prod embed
    others_j = tf.keras.layers.Input(shape=(num_global_feature,), name='others')
    time_window_s = tf.keras.layers.Input(shape=(window_size_s, 1), name='time_s')
    conv_l1_s = tf.keras.layers.Conv1D(filters=64, kernel_size=window_size_s,
                                       activation="relu", input_shape=[window_size_s, 1])(time_window_s)
    flat_l1_s = tf.keras.layers.Flatten()(conv_l1_s)

    time_window_m = tf.keras.layers.Input(shape=(window_size_m, 1), name='time_m')
    conv_l1_m = tf.keras.layers.Conv1D(filters=64, kernel_size=window_size_m,
                                       activation="relu", input_shape=[window_size_m, 1])(time_window_m)
    flat_l1_m = tf.keras.layers.Flatten()(conv_l1_m)

    time_window_l = tf.keras.layers.Input(shape=(window_size_l, 1), name='time_l')
    conv_l1_l = tf.keras.layers.Conv1D(filters=64, kernel_size=window_size_l,
                                       activation="relu", input_shape=[window_size_l, 1])(time_window_l)
    flat_l1_l = tf.keras.layers.Flatten()(conv_l1_l)

    # pool_1 = tf.keras.layers.MaxPool1D(pool_size=2)(conv_1)

    merge_1 = tf.keras.layers.concatenate([flat_l1_s, flat_l1_m, flat_l1_l, others_j])
    dense_2 = tf.keras.layers.Dense(50, activation='relu')(merge_1)
    dense_3 = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dense_2)
    model = tf.keras.models.Model(inputs=[time_window_s, time_window_m, time_window_l, others_j], outputs=dense_3)

    x_cnn_train_s = np.expand_dims(x_train[:, :window_size_s], axis=-1)
    x_cnn_train_m = np.expand_dims(x_train[:, :window_size_m], axis=-1)
    x_cnn_train_l = np.expand_dims(x_train[:, :window_size_l], axis=-1)
    x_cnn_test_s = np.expand_dims(x_test[:, :window_size_s], axis=-1)
    x_cnn_test_m = np.expand_dims(x_test[:, :window_size_m], axis=-1)
    x_cnn_test_l = np.expand_dims(x_test[:, :window_size_l], axis=-1)

    lr_start = 5e-4
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr_start))
    history = model.fit([x_cnn_train_s, x_cnn_train_m, x_cnn_train_l, global_features_train], y_train,
                        validation_data=([x_cnn_test_s, x_cnn_test_m, x_cnn_test_l, global_features_test], y_test),
                        shuffle=True, epochs=10, batch_size=512)

    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    #     lambda epoch: lr_start * 10 ** (epoch / 20))
    # history = model.fit([x_cnn_train_s, x_cnn_train_m, x_cnn_train_l, global_features_train], y_train,
    #                     validation_data=([x_cnn_test_s, x_cnn_test_m, x_cnn_test_l, global_features_test], y_test),
    #                     shuffle=True, epochs=60, batch_size=128, callbacks=[lr_schedule])
    # plt.semilogx(history.history["lr"], history.history["loss"])
    # plt.axis([lr_start, 1e-1, 0.05, 0.09])
    # plt.show()

    loss = model.evaluate([x_cnn_test_s, x_cnn_test_s, x_cnn_test_l, global_features_test], y_test, verbose=0)

    return loss
