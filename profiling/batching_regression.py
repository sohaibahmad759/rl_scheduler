import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('aggregate_profiled.csv')
new_df = pd.DataFrame()

pairs = set()

for index, row in df.iterrows():
    # print(row['Model'], row['Accel'])
    pair = (row['Model'], row['Accel'])
    pairs.add(pair)

print(len(pairs))

counter = 0
for pair in pairs:
    counter += 1
    # if counter > 5:
    #     break

    (model, accelerator) = pair
    print(f'model: {model}, accelerator: {accelerator}')

    pair_df = df.loc[(df['Model'] == model) & (df['Accel'] == accelerator)]

    # X = np.array(pair_df.loc[:, ['batchsize', '90th_pct', 'avg_latency(ms)',
    #                              'Min']])
    X = np.array(pair_df['batchsize'].to_list())
    X = X[:, None]

    y = np.array(pair_df.loc[:, '50th_pct'].to_list())
    y = y[:, None]

    # print(df)
    # print(pair_df)
    print(f'X: {X}')

    polynomial_converter = PolynomialFeatures(degree=1, include_bias=True)
    # Converter "fits" to data, in this case, reads in every X column
    # Then it "transforms" and ouputs the new polynomial data
    poly_features = polynomial_converter.fit_transform(X)
    poly_features.shape

    # X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.0, random_state=101)
    X_train = poly_features
    y_train = y

    # print(f'X_train: {poly_features}')

    poly_model = LinearRegression(fit_intercept=True)
    poly_model.fit(X_train, y_train)

    # train_pred = model.predict(X_train)
    # test_pred = model.predict(X_test)

    print(f'X_train: {X_train}')
    print(f'y_train: {y_train}')
    # print(f'train_pred: {train_pred}')
    # print(f'test_pred: {test_pred}')

    # print(f'X: {X}')
    # print(f'y: {y}')


    batch_sizes = np.array([1, 2, 4, 8, 12, 16, 20, 24, 28, 32])
    x_test = batch_sizes[:, None]
    x_test = polynomial_converter.fit_transform(x_test)
    print(f'x_test: {x_test}')

    y_test = poly_model.predict(x_test)
    print(f'y_test: {y_test}')
    # print(model.predict([[32]]))

    print(f'pair_df: {pair_df.iloc[0, :]}')
    sample_row = pair_df.iloc[0, :]

    fifty_pct_values = []
    prev_value = 0
    for y_value in y_test.ravel():
        if y_value <= prev_value:
            break
        
        fifty_pct_values.append(y_value)
        prev_value = y_value

    # print(f'fifty_pct_values: {fifty_pct_values}')

    idx = 0
    row = df.loc[(df['Model'] == model) & (df['Accel'] == accelerator) & (df['batchsize'] == 1)]
    for y_value in fifty_pct_values:
        # ((df['A'] == 2) & (df['B'] == 3)).any()
        new_row = copy.deepcopy(row)

        batch_size = batch_sizes[idx]
        new_row['batchsize'] = batch_size
        new_row['50th_pct'] = y_value

        print(f'model: {model}, accelerator: {accelerator}, batchsize: {batch_size}')
        existing = df.loc[(df['Model'] == model) & (df['Accel'] == accelerator) & (df['batchsize'] == batch_size)]
        if ((df['Model'] == model) & (df['Accel'] == accelerator) & (df['batchsize'] == batch_size)).any():
            print(f'existing: {existing}')
            new_df = new_df.append(existing, ignore_index=True)
        # continue
        else:
            new_row['90th_pct'] = None
            new_row['avg_latency(ms)'] = None
            new_row['Max'] = None
            new_row = pd.DataFrame(new_row)

            # print(f'new_row: {new_row}')
            # new_df = pd.concat([new_df, new_row], axis=0, ignore_index=True)
            new_df = new_df.append(new_row, ignore_index=True)
        idx += 1

print(f'new_df: {new_df}')
new_df.to_csv('aggregate_profiled_extended.csv')

    # test_predictions = model.predict(X_test)
    # from sklearn.metrics import mean_absolute_error,mean_squared_error
    # MAE = mean_absolute_error(y_test,test_predictions)
    # MSE = mean_squared_error(y_test,test_predictions)
    # RMSE = np.sqrt(MSE)
