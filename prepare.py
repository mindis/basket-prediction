from model import *
import pandas as pd
import gc
import os


def prepare_tensors(path):
    # load the dataset
    products = pd.read_csv(os.path.join(path, 'train.csv'),
                           dtype={'i': int,
                                  'j': int,
                                  't': int,
                                  'price': float,
                                  'advertised': int})
    products.rename(columns={'i': 'customer_id', 'j': 'product_id', 't': 'week'}, inplace=True)
    # create data set for each (i, j) pair

    I = len(products.customer_id.unique())
    J = len(products.product_id.unique())
    T = len(products.week.unique())

    product_tensor = np.zeros((I, T, J))
    discount_tensor = np.zeros((I, T, J))

    for _, row in products.iterrows():
        i = int(row['customer_id'])
        t = int(row['week'])
        j = int(row['product_id'])
        d = int(row['advertised'])
        product_tensor[i, t, j] = 1.
        discount_tensor[i, t, j] = float(d == 1)
    np.save('data/product_tensor.npy', product_tensor)
    np.save('data/discount_tensor.npy', discount_tensor)
    print('Product and discount tensors created!')
    return product_tensor, discount_tensor


def prepare_multivariate_series(product_tensor, window_size, split_time, x_axis=2, y_axis=1):
    I, T, J = product_tensor.shape
    for j in range(J):
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(I):
            series = product_tensor[i, :, j]
            for time_ in range(len(series) - window_size):
                purchase_series = series[time_: time_ + window_size]
                y = series[time_ + window_size]
                if time_ < split_time - window_size:
                    x_train.append(purchase_series.tolist())
                    y_train.append(y)
                else:
                    x_test.append(purchase_series.tolist())
                    y_test.append(y)

        x_train = np.expand_dims(np.array(x_train), axis=-1)
        x_test = np.expand_dims(np.array(x_test), axis=-1)
        y_train = np.expand_dims(np.array(y_train), axis=-1)
        y_test = np.expand_dims(np.array(y_test), axis=-1)

        x_concat_train = x_train if j == 0 else np.concatenate((x_concat_train, x_train), axis=x_axis)
        x_concat_test = x_test if j == 0 else np.concatenate((x_concat_test, x_test), axis=x_axis)
        y_concat_train = y_train if j == 0 else np.concatenate((y_concat_train, y_train), axis=y_axis)
        y_concat_test = y_test if j == 0 else np.concatenate((y_concat_test, y_test), axis=y_axis)

    u_train = [i for i in range(I) for t in range(split_time - window_size)]
    w_train = [t for i in range(I) for t in range(split_time - window_size)]
    u_test = [i for i in range(I) for t in range(split_time - window_size, len(series) - window_size)]
    w_test = [t for i in range(I) for t in range(split_time - window_size, len(series) - window_size)]

    np.save('data/x_multi_series_train_w%d_s%d.npy' % (window_size, split_time), x_concat_train)
    np.save('data/y_multi_series_train_w%d_s%d.npy' % (window_size, split_time), y_concat_train)
    np.save('data/x_multi_series_test_w%d_s%d.npy' % (window_size, split_time), x_concat_test)
    np.save('data/y_multi_series_test_w%d_s%d.npy' % (window_size, split_time), y_concat_test)
    np.save('data/weeks_multi_series_train_w%d_s%d.npy' % (window_size, split_time), np.array(w_train))
    np.save('data/weeks_multi_series_test_w%d_s%d.npy' % (window_size, split_time), np.array(w_test))
    np.save('data/user_ids_multi_series_train_w%d_s%d.npy' % (window_size, split_time), np.array(u_train))
    np.save('data/user_ids_multi_series_test_w%d_s%d.npy' % (window_size, split_time), np.array(u_test))


def prepare_windowed_dataset(product_tensor, discount_tensor, window_size, split_time):
    I, T, J = product_tensor.shape
    x_train, y_train, x_test, y_test, x_prediction = [], [], [], [], []
    product_ids_train, user_ids_train, product_ids_test, user_ids_test, weeks_train, weeks_test = [], [], [], [], [], []
    for j in range(J):
        for i in range(I):
            series = product_tensor[i, :, j]
            for time_ in range(len(series) - window_size):
                purchase_series = series[time_: time_ + window_size]
                purchase_mean = np.mean(product_tensor[i, :time_ + window_size, j])
                d = discount_tensor[i, time_ + window_size, j]
                y = series[time_ + window_size]
                if time_ < split_time - window_size:
                    x_train.append(purchase_series.tolist() + [purchase_mean] + [d])
                    y_train.append(y)
                    product_ids_train.append(j)
                    user_ids_train.append(i)
                    weeks_train.append(time_)
                else:
                    x_test.append(purchase_series.tolist() + [purchase_mean] + [d])
                    y_test.append(y)
                    product_ids_test.append(j)
                    user_ids_test.append(i)
                    weeks_test.append(time_)

                if time_ == len(series) - window_size - 1:
                    x_prediction.append(series[time_: time_ + window_size] + [y] + [purchase_mean])

    np.save('data/x_windowed_train_w%d_s%d.npy' % (window_size, split_time), np.array(x_train))
    np.save('data/y_windowed_train_w%d_s%d.npy' % (window_size, split_time), np.array(y_train))
    np.save('data/x_windowed_test_w%d_s%d.npy' % (window_size, split_time), np.array(x_test))
    np.save('data/y_windowed_test_w%d_s%d.npy' % (window_size, split_time), np.array(y_test))
    np.save('data/x_windowed_prediction_w%d_s%d.npy' % (window_size, split_time), np.array(x_prediction))
    np.save('data/product_ids_windowed_train_w%d_s%d.npy' % (window_size, split_time), np.array(product_ids_train))
    np.save('data/product_ids_windowed_test_w%d_s%d.npy' % (window_size, split_time), np.array(product_ids_test))
    np.save('data/user_ids_windowed_train_w%d_s%d.npy' % (window_size, split_time), np.array(user_ids_train))
    np.save('data/user_ids_windowed_test_w%d_s%d.npy' % (window_size, split_time), np.array(user_ids_test))
    np.save('data/weeks_windowed_train_w%d_s%d.npy' % (window_size, split_time), np.array(weeks_train))
    np.save('data/weeks_windowed_test_w%d_s%d.npy' % (window_size, split_time), np.array(weeks_test))


def create_windowed_dataset_per_product(product_tensor, discount_tensor, window_size, split_time):
    I, T, J = product_tensor.shape
    for j in range(J):
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(I):
            series = product_tensor[i, :, j]
            for time_ in range(len(series) - window_size):
                purchase_series = series[time_: time_ + window_size]
                purchase_mean = np.mean(series[:time_ + window_size])
                d = discount_tensor[i, time_ + window_size, j]
                y = series[time_ + window_size]
                if time_ < split_time - window_size:
                    x_train.append(purchase_series.tolist() + [purchase_mean] + [d])
                    y_train.append(y)
                else:
                    x_test.append(purchase_series.tolist() + [purchase_mean] + [d])
                    y_test.append(y)

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        if j == 0:
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        np.save('data/product_%d_train_x.npy' % j, x_train)
        np.save('data/product_%d_train_y.npy' % j, y_train)
        np.save('data/product_%d_test_x.npy' % j, x_test)
        np.save('data/product_%d_test_y.npy' % j, y_test)


def prepare_global_features(window_size, split_time, concat_time=True, concat_nmf=True, concat_cnn=True,
                            concat_cnn_lstm=True, concat_multi_lstm=True):

    product_ids_train = np.load('data/product_ids_windowed_train_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    product_ids_test = np.load('data/product_ids_windowed_test_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    user_ids_train = np.load('data/user_ids_windowed_train_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    user_ids_test = np.load('data/user_ids_windowed_test_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    weeks_train = np.load('data/weeks_windowed_train_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    weeks_test = np.load('data/weeks_windowed_test_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    product_ids_pred = np.expand_dims(np.array([j for j in range(40) for i in range(2000)]), axis=-1)
    user_ids_pred = np.expand_dims(np.array([i for j in range(40) for i in range(2000)]), axis=-1)

    x_train = np.load('data/x_windowed_train_w%d_s%d.npy' % (window_size, split_time))
    x_test = np.load('data/x_windowed_test_w%d_s%d.npy' % (window_size, split_time))
    x_prediction = np.load('data/x_windowed_prediction_w%d_s%d.npy' % (window_size, split_time))

    df_train = pd.DataFrame(data=product_ids_train, columns=['product_id'])
    df_train['user_id'] = user_ids_train
    df_train['week_id'] = weeks_train
    df_train['purchase_mean'] = x_train[:, -2]
    df_train['discount'] = x_train[:, -1]
    df_test = pd.DataFrame(data=product_ids_test, columns=['product_id'])
    df_test['user_id'] = user_ids_test
    df_test['week_id'] = weeks_test
    df_test['purchase_mean'] = x_test[:, -2]
    df_test['discount'] = x_test[:, -1]
    df_pred = pd.DataFrame(data=product_ids_pred, columns=['product_id'])
    df_pred['user_id'] = user_ids_pred
    df_pred['purchase_mean'] = x_prediction[:, -1]
    df_promo = pd.read_csv(os.path.join('data/promotion_schedule.csv'),
                           dtype={'j': int,
                                  'discount': float,
                                  'advertised': int})
    df_promo.rename(columns={'j': 'product_id'}, inplace=True)
    df_promo.drop('discount', axis=1, inplace=True)
    df_pred = df_pred.merge(df_promo, how='left', on='product_id')

    print('Main dataframe loaded...')

    del product_ids_train, product_ids_test, user_ids_train, user_ids_test, weeks_train, weeks_test, df_promo
    gc.collect()

    if concat_time and os.path.exists('predictions/prod_embed.npy'):

        df_train = pd.concat([df_train,
                              pd.DataFrame(x_train[:, :window_size],
                                           columns=['time_window_{}'.format(i)
                                                    for i in range(x_train[:, :window_size].shape[1])])],
                             axis=1)
        df_test = pd.concat([df_test,
                             pd.DataFrame(x_test[:, :window_size],
                                          columns=['time_window_{}'.format(i)
                                                   for i in range(x_test[:, :window_size].shape[1])])],
                            axis=1)
        df_pred = pd.concat([df_pred,
                             pd.DataFrame(x_prediction[:, :window_size],
                                          columns=['time_window_{}'.format(i)
                                                   for i in range(x_prediction[:, :window_size].shape[1])])],
                            axis=1)

    print('Time windows loaded...')

    del x_train, x_test, x_prediction
    gc.collect()

    # add nmf user and product embeddings
    if concat_nmf and os.path.exists('predictions/prod_embed.npy'):
        nmf_prod = np.load('predictions/prod_embed.npy')
        product_emb_df = pd.DataFrame(nmf_prod, columns=['nmf_product_{}'.format(i) for i in range(nmf_prod.shape[1])])
        product_emb_df['product_id'] = np.arange(nmf_prod.shape[0])
        df_train = df_train.merge(product_emb_df, how='left', on='product_id')
        df_test = df_test.merge(product_emb_df, how='left', on='product_id')
        df_pred = df_pred.merge(product_emb_df, how='left', on='product_id')

        del nmf_prod, product_emb_df
        gc.collect()
        print('NMF product embeddings loaded...')

        nmf_user = np.load('predictions/user_embed.npy')
        user_emb_df = pd.DataFrame(nmf_user, columns=['nmf_user_{}'.format(i) for i in range(nmf_user.shape[1])])
        user_emb_df['user_id'] = np.arange(nmf_user.shape[0])
        df_train = df_train.merge(user_emb_df, how='left', on='user_id')
        df_test = df_test.merge(user_emb_df, how='left', on='user_id')
        df_pred = df_pred.merge(user_emb_df, how='left', on='user_id')

        del nmf_user, user_emb_df
        gc.collect()
        print('NMF user embeddings loaded...')

    weeks_multi_series_train = np.load(
        'data/weeks_multi_series_train_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    weeks_multi_series_test = np.load(
        'data/weeks_multi_series_test_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    user_ids_multi_series_train = np.load(
        'data/user_ids_multi_series_train_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)
    user_ids_multi_series_test = np.load(
        'data/user_ids_multi_series_test_w%d_s%d.npy' % (window_size, split_time)).reshape(-1, 1)

    # add multivariate lstm features
    if concat_multi_lstm and os.path.exists('predictions/lstm_multi_final_states_train.npy'):
        multi_rnn_train = np.load('predictions/lstm_multi_final_states_train.npy')
        multi_rnn_df_train = pd.DataFrame(multi_rnn_train,
                                          columns=['multi_rnn_train_{}'.format(i) for i in range(multi_rnn_train.shape[1])])
        multi_rnn_df_train['user_id'] = user_ids_multi_series_train
        multi_rnn_df_train['week_id'] = weeks_multi_series_train
        df_train = df_train.merge(multi_rnn_df_train, how='left', on=['user_id', 'week_id'])

        multi_rnn_test = np.load('predictions/lstm_multi_final_states_test.npy')
        multi_rnn_df_test = pd.DataFrame(multi_rnn_test,
                                         columns=['multi_rnn_test_{}'.format(i) for i in range(multi_rnn_test.shape[1])])
        multi_rnn_df_test['user_id'] = user_ids_multi_series_test
        multi_rnn_df_test['week_id'] = weeks_multi_series_test
        df_test = df_test.merge(multi_rnn_df_test, how='left', on=['user_id', 'week_id'])

        del weeks_multi_series_train, weeks_multi_series_test, user_ids_multi_series_train, user_ids_multi_series_test
        del multi_rnn_df_train, multi_rnn_train, multi_rnn_df_test, multi_rnn_test
        gc.collect()

        print('Multivariate lstm features loaded...')

    # add cnn-lstm features
    if concat_cnn_lstm and os.path.exists('predictions/lstm_final_states_train.npy'):
        cnn_rnn_train = np.load('predictions/lstm_final_states_train.npy')
        df_train = pd.concat(
            [df_train,
             pd.DataFrame(cnn_rnn_train, columns=['cnn_rnn_train_{}'.format(i) for i in range(cnn_rnn_train.shape[1])])],
            axis=1)

        cnn_rnn_test = np.load('predictions/lstm_final_states_test.npy')
        df_test = pd.concat(
            [df_test,
             pd.DataFrame(cnn_rnn_test,columns=['cnn_rnn_test_{}'.format(i) for i in range(cnn_rnn_test.shape[1])])],
            axis=1)

        del cnn_rnn_train, cnn_rnn_test
        gc.collect()

        print('Cnn-lstm features loaded...')

    # add cnn features
    if concat_cnn and os.path.exists('predictions/cnn_features_train.npy'):
        cnn_train = np.load('predictions/cnn_features_train.npy')
        df_train = pd.concat([df_train,
                              pd.DataFrame(cnn_train, columns=['cnn_train_{}'.format(i) for i in range(cnn_train.shape[1])])],
                             axis=1)

        cnn_test = np.load('predictions/cnn_features_test.npy')
        df_test = pd.concat(
            [df_test,
             pd.DataFrame(cnn_test, columns=['cnn_test_{}'.format(i) for i in range(cnn_test.shape[1])])],
            axis=1)

        del cnn_train, cnn_test
        gc.collect()

        print('Cnn features loaded...')

    drop_cols = ['product_id', 'user_id', 'week_id']
    df_train.drop(drop_cols, axis=1, inplace=True)
    df_test.drop(drop_cols, axis=1, inplace=True)
    drop_cols = ['product_id', 'user_id']
    df_pred.drop(drop_cols, axis=1, inplace=True)
    global_features_train = df_train.values
    global_features_test = df_test.values
    global_features_pred = df_pred.values

    print(df_train.info())
    print(df_train.head())
    print(df_test.info())
    print(df_test.head())
    print(df_pred.info())
    print(df_pred.head())

    np.save('data/feature_names.npy', df_train.columns)
    np.save('data/global_features_train_w%d_s%d.npy' % (window_size, split_time), global_features_train)
    np.save('data/global_features_test_w%d_s%d.npy' % (window_size, split_time), global_features_test)
    np.save('data/global_features_pred_w%d_s%d.npy' % (window_size, split_time), global_features_pred)

    del df_train, global_features_train, global_features_test
    gc.collect()

    print('Loading features DONE...')


def prepare_dataset(window_size, split_time):
    # read csv and create tensors
    prepare_tensors()
    # product_tensor, discount_tensor = create_tensors()
    product_tensor = np.load('data/product_tensor.npy')
    discount_tensor = np.load('data/discount_tensor.npy')
    # prepare windowed dataset for cnn-based models cnn1d and cnn1d-lstm
    prepare_windowed_dataset(product_tensor, discount_tensor, window_size, split_time)
    # prepare multivariate series for lstm-multivariate model
    prepare_multivariate_series(product_tensor, window_size, split_time)


def prepare_features(window_size, split_time):
    # prepare global features
    prepare_global_features(window_size, split_time)
