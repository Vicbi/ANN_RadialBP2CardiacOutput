import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing

from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import History


def scale_data(dataset):
    """
    Scale the input dataset using Min-Max scaling.

    Parameters:
        dataset (pd.DataFrame): The dataset to be scaled.

    Returns:
        pd.DataFrame: Scaled dataset.
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(dataset.values)
    scaled_dataset = pd.DataFrame(scaled_array, columns=dataset.columns)
     
    return scaled_dataset


def rescale_values(values, prediction_variable, dataset):
    """
    Rescale values based on the specified prediction_variable and dataset.

    Parameters:
        values (numpy.ndarray): Array to be rescaled.
        prediction_variable (str): The variable being predicted.
        dataset (pandas.DataFrame): The dataset containing the prediction variable.

    Returns:
        rescaled_values (numpy.ndarray): Rescaled values.
    """
    max_prediction_variable = np.max(dataset[prediction_variable])
    min_prediction_variable = np.min(dataset[prediction_variable])
    
    rescaled_values = min_prediction_variable + (max_prediction_variable - min_prediction_variable) * values
    
    return rescaled_values


def add_random_noise(data, perc_lower, perc_upper, noise_mode):
    """
    Add random noise to the input data.

    Parameters:
        data (numpy.ndarray): Input data.
        perc_lower (float): Lower percentage for noise range.
        perc_upper (float): Upper percentage for noise range.
        noise_mode (bool): True for adding noise, False otherwise.

    Returns:
        numpy.ndarray: Noisy data if noise_mode is True, else original data.
    """
    lower = (100 - perc_lower) / 10
    upper = (100 + perc_upper) / 10
    noise = 0.1 * np.random.uniform(lower, upper, size=data.shape)

    if noise_mode:
        noisy_data = data * noise
        print('Noise added.')
    else:
        noisy_data = data
        print('No noise added.')

    return noisy_data



def select_input_dataset(norm_mode):
    """
    Load and preprocess the dataset based on normalization mode.

    Parameters:
        norm_mode (bool): Whether to use normalized dataset.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    file_suffix = "_normalized" if norm_mode else ""
    filename = f"CO_from_radialBP_resampled_repositioned_128Hz{file_suffix}.csv"
    
    pathname = 'Data/{}'.format(filename)
    dataset = pd.read_csv(pathname)
    dataset = dataset.drop(['id', 'HR', 'CT_sys', 'aoMAP'], axis=1)
    dataset = check_for_nan_values(dataset)

    return dataset


def check_for_nan_values(data):
    """
    Remove rows with NaN values.

    Parameters:
        data (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with NaN rows removed.
    """
    data = data.dropna().reset_index(drop=True)
    return data


def load_train_test_val_indices():
    """
    Load indices for training, testing, and validation sets.

    Returns:
        tuple: A tuple containing train_val_indices, train_indices, test_indices, val_indices.
    """
    with open('TrainedModels/SavedTrainValTestSplitIndices/train_val_indices', 'rb') as f:
        train_val_indices = pickle.load(f)
    with open('TrainedModels/SavedTrainValTestSplitIndices/test_indices', 'rb') as f:
        test_indices = pickle.load(f)
    with open('TrainedModels/SavedTrainValTestSplitIndices/train_indices', 'rb') as f:
        train_indices = pickle.load(f)
    with open('TrainedModels/SavedTrainValTestSplitIndices/val_indices', 'rb') as f:
        val_indices = pickle.load(f)
    
    return train_val_indices, train_indices, test_indices, val_indices


def split_train_test_val_sets(data):
    """
    Split the dataset into training, testing, and validation sets.

    Parameters:
        data (pd.DataFrame): The input dataset.

    Returns:
        tuple: A tuple containing X_train, X_test, X_val, y_train, y_test, y_val.
    """
    start = 0
    prediction = -1
    
    X = data.iloc[:, start:prediction].values
    y = data.iloc[:, prediction].values
    indices = np.arange(len(X))
    
    train_val_indices, train_indices, test_indices, val_indices = load_train_test_val_indices()

    X_train_val = X[train_val_indices]
    X_test = X[test_indices]
    y_train_val = y[train_val_indices]
    y_test = y[test_indices]

    X_train = X_train_val[train_indices]
    X_val = X_train_val[val_indices]
    y_train = y_train_val[train_indices]
    y_val = y_train_val[val_indices]

    return X_train, X_test, X_val, y_train, y_test, y_val


def artificial_neural_network(selected_batch_size, selected_epochs, X_train, X_test, y_train, y_test, verbose):
    """
    Train and evaluate an Artificial Neural Network (ANN) model.

    Parameters:
        selected_batch_size (int): Batch size for training.
        manual_epochs (int): Number of training epochs.
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Test features.
        y_train (numpy.ndarray): Training target values.
        y_test (numpy.ndarray): Test target values.
        verbose (int): Verbosity mode.

    Returns:
        Tuple: Trained model and predicted values.
    """
    numcolsX = X_train.shape[1]

    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation='relu', input_dim=numcolsX))
    # Adding the output layer
    model.add(Dense(units=1))
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size=selected_batch_size, epochs=selected_epochs, verbose=verbose)
    y_pred = model.predict(X_test)

    return model, y_pred



def find_optimal_no_epochs(batch_size, data, X_train, X_val, y_train, y_val, prediction_variable, verbose, norm_mode):
    """
    Find the optimal number of epochs for training a neural network.

    Parameters:
        batch_size (int): Batch size for training.
        data (pandas.DataFrame): Input data.
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        y_train (numpy.ndarray): Training target values.
        y_val (numpy.ndarray): Validation target values.
        prediction_variable (str): Name of the prediction variable.
        verbose (int): Verbosity mode.
        norm_mode (str): Normalization mode.

    Returns:
        int: Optimal number of epochs.
    """
    numcolsX = X_train.shape[1]

    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation='relu', input_dim=numcolsX))
    # Adding the output layer
    model.add(Dense(units=1))
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

    model_history = model.fit(X_train, y_train,
                             batch_size = batch_size,
                             epochs = 1000,
                             verbose = verbose,
                             validation_data = (X_val, y_val),
                             callbacks = [early_stopping])

    a = np.max(data[prediction_variable]) - np.min(data[prediction_variable])
    train_loss = model_history.history['loss']
    validation_loss = model_history.history['val_loss']
    train_loss_scaled = [i * (a * a) for i in train_loss]
    validation_loss_scaled = [i * (a * a) for i in validation_loss]

    # Plot the loss function
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.plot(np.sqrt(train_loss_scaled), '--', linewidth=4, color="#111111", label='Training loss')
    ax.plot(np.sqrt(validation_loss_scaled), linewidth=4, color="#111111", label='Validation loss')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    optimal_no_epochs = len(validation_loss_scaled)
    print('Epochs:', optimal_no_epochs)
    filename = 'loss_epochs_batch_size_{}_norm_{}.tiff'.format(batch_size, norm_mode)
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    return optimal_no_epochs

            
def print_results(y_test, y_pred, variable_unit):
    """
    Print various regression metrics and statistics.

    Parameters:
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        None
    """
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    nrmse = 100 * rmse / (np.max(y_test) - np.min(y_test))

    print('Mean Absolute Error:', np.round(mae, 2), variable_unit)
    print('Mean Squared Error:', np.round(mse, 2), variable_unit)
    print('Root Mean Squared Error:', np.round(rmse, 2), variable_unit)
    print('Normalized Root Mean Squared Error:', np.round(nrmse, 2), '%\n')

    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    print('Correlation:', round(r_value, 2))
    print('Slope:', round(slope, 2))
    print('Intercept:', round(intercept, 2), variable_unit)
    print('r_value:', round(r_value, 2))
    print('p_value:', round(p_value, 4))

    print('Distribution of the reference data:', round(np.mean(y_test), 1), '±', round(np.std(y_test), 1), variable_unit)
    print('Distribution of the predicted data:', round(np.mean(y_pred), 1), '±', round(np.std(y_pred), 1), variable_unit)
    


def get_and_save_train_test_val_indices(data,select_test_size,select_val_rel_size):
    
    start = 0; prediction = -1; 
    X = data.iloc[:,start:prediction].values
    y = data.iloc[:,prediction].values
    indices = np.arange(len(X))

    indices = np.arange(len(X))
    X_train_val, X_test, y_train_val, y_test, indices_train_val, indices_test = train_test_split(X, y, indices, test_size=select_test_size, random_state=42)

    indices2 = np.arange(len(X_train_val))
    X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X_train_val, y_train_val,indices2,test_size=select_val_rel_size, random_state=42) 

    train_val_indices = indices_train_val
    test_indices = indices_test
    val_indices = indices_val
    train_indices = indices_train
    # print(indices_val.shape)

    with open('TrainedModels/SavedTrainValTestSplitIndices/train_val_indices', 'wb') as f:
        pickle.dump(indices_train_val, f)
    with open('TrainedModels/SavedTrainValTestSplitIndices/test_indices', 'wb') as f:
        pickle.dump(indices_test, f)
    with open('TrainedModels/SavedTrainValTestSplitIndices/train_indices', 'wb') as f:
        pickle.dump(indices_train, f)
    with open('TrainedModels/SavedTrainValTestSplitIndices/val_indices', 'wb') as f:
        pickle.dump(indices_val, f)
    

def load_model_with_pickle(filename):
    """
    Load a machine learning model from a pickle file.

    Parameters:
        filename (str): The name of the pickle file.

    Returns:
        object: The loaded machine learning model.
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
    

