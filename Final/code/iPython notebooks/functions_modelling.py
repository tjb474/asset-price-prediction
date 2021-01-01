
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    classification_report
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    PredefinedSplit,
    TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf


class TimeSeriesSplitImproved(TimeSeriesSplit):
    """
    This is a modified version of sklearn's TimeSeriesSplit.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.

    There are 2 modifications to this class.
    1. In each split, test indices must be higher than before,
       and thus shuffling in cross validator is inappropriate.
       I.e. there is no shuffling of samples.
    2. There is now the ability to produce splits of fixed length.
       Previously, the only option was for successive training sets
       to be supersets of those that come before them. This was
       not suitable for addressing potential concept drift and
       therefore the argument 'fixed_length' was added to allow
       training sets that step forward.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide `.
    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [1] TEST: [2]
    TRAIN: [2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True,
    ...     train_splits=2):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [1 2] TEST: [3]

    Notes
    -----
    When ``fixed_length`` is ``False``, the training set has size
    ``i * train_splits * n_samples // (n_splits + 1) + n_samples %
    (n_splits + 1)`` in the ``i``th split, with a test set of size
    ``n_samples//(n_splits + 1) * test_splits``, where ``n_samples``
    is the number of samples. If fixed_length is True, replace ``i``
    in the above formulation with 1, and ignore ``n_samples %
    (n_splits + 1)`` except for the first training set. The number
    of test sets is ``n_splits + 2 - train_splits - test_splits``.
    """

    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=1, test_splits=1):
        """
        Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        fixed_length : bool, whether training sets should always have
            common length
        train_splits : positive int, for the minimum number of
            splits to include in training sets
        test_splits : positive int, for the number of splits to
            include in the test set

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)  # indexable arrays for x-val
        n_samples = _num_samples(X)  # Return number of samples in array-like x
        n_splits = self.n_splits  # number of split
        n_folds = n_splits + 1  # num folds (1 split=2 folds, 2 splits=3 folds)

        # defaults to 1 for each
        train_splits, test_splits = int(train_splits), int(test_splits)

        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) == 0 and test_splits > 0:
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))

        indices = np.arange(n_samples)  # list indices of all the examples
        split_size = (n_samples // n_folds)  # number of samples in each fold
        test_size = split_size * test_splits  # number of samples in test set
        train_size = split_size * train_splits  # num of samples in train set
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                       indices[test_start:test_start + test_size])


def return_walkforward_indices(X):
    """
    For the train and test sets, this function returns the corresponding
    indices for each separate sliding window

    Parameters
    ----------
        X (df): the original X dataframe containing all features related
            to the asset in question

    Returns
    -------
        train_indices (list of arrays): arrays containing indices of each
            sliding window training samples
            E.g. [array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]
        test_indices (list): arrays containing indices of each
            sliding window test samples
            E.g. [array([5, 6, 7]), array([10, 11, 12])]

    """
    # set up the generator function to split the dataset into 4 windows
    tscv = TimeSeriesSplitImproved()
    split = tscv.split(X, fixed_length=True, train_splits=2, test_splits=1)

    train_indices = []
    test_indices = []

    for train_index, test_index in split:
        train_indices.append(train_index)
        test_indices.append(test_index)

    return train_indices, test_indices


def walkforward_split(X, y):
    """
    This function uses the train and test indices of each sliding window
    - returned by return_walkforward_indices() - to slice the dataframe,
    creating n separate sliding windows.

    For each sliding window, there is now X_train, y_train, X_test, y_test.
    e.g. X_train_0, y_train_0, X_test_0, y_test_0 is for the first sliding
    window.

    The function also scales the inputs of each sliding window. The scaler
    is fit on each sliding window's training set, rather than the training
    set as a whole.

    Parameters
    ----------
        X (df): the original X dataframe containing all features related
            to the asset in question
        y (Series): the original labels that correspond with X

    Returns
    -------
        sets (dict of matrices): dict containing the train and test matrices
            for each sliding window.
    """
    train_indices, test_indices = return_walkforward_indices(X)
    sets = {}
    scaler = StandardScaler()

    num_windows = len(train_indices)

    # for each sliding window (which equals the sum of train and test splits)
    for i in np.arange(num_windows):

        # create sliding window 'i' (containing train and test)
        set = {'X_train_'+str(i): pd.DataFrame(X.values[train_indices[i]]),
               'y_train_'+str(i): pd.DataFrame(y.values[train_indices[i]]),
               'X_test_'+str(i): pd.DataFrame(X.values[test_indices[i]]),
               'y_test_'+str(i): pd.DataFrame(y.values[test_indices[i]])
               }
        sets.update(set)

    # scale the inputs for easier neural network learning
    for i in np.arange(0, len(sets), 4):  # for each sliding window train set

        # fit scaler on the training data of training set 'i'
        # (only the training - don't cheat!)
        scaler.fit(list(sets.values())[i])

        # apply transformation on the training set
        list(sets.values())[i].\
            update(scaler.transform(list(sets.values())[i]))
        # apply transformation on the test set
        list(sets.values())[i+2]\
            .update(scaler.transform(list(sets.values())[i+2]))

    return sets


def model_selection_sets(windows, test_size=0.3):
    """
    This function takes each sliding window training set and breaks
    it into a train and validation set for the purpose of model selection.

    Again, samples are not shuffled. It is a simple split to keep
    sequential ordering.

    Parameters
    ----------
        windows (dict of matrices): the dictionary containing the sets
            corresponding with each sliding window that is produced by
            walkforward_split().
        test_size (float): a float between 0 and 1 that determines the
            size of the proportion of samples to be assigned to the
            validation set.

    Returns
    -------
        sets_ms (dict of matrices): dict containing the train and validation
            set matrices for each sliding window, to be used for model
            selection.
            E.g. dict_keys(
                ['X_train_0', 'X_val_0', 'y_train_0', 'y_val_0', # SW0 MS sets
                 'X_train_1', 'X_val_1', 'y_train_1', 'y_val_1', # SW1 MS sets
                 .....
                 'X_train_N', 'X_val_N', 'y_train_N', 'y_val_N', # SWN MS sets
                 ])
    """

    sets_ms = {}  # ms = model selection

    for i in np.arange(0, len(windows), 4):  # for each SW train set
        X_train, X_val, y_train, y_val = train_test_split(
            list(windows.values())[i],
            list(windows.values())[i+1],
            # i = X_train for given window, i+1 = y_train for given window
            test_size=test_size, shuffle=False)

        set_ms = {list(windows.keys())[i]+'_'+'a': X_train,
                  'X_val_'+str(list(windows.keys())[i])[8]: X_val,
                  list(windows.keys())[i+1]+'_'+'a': y_train,
                  'y_val_'+str(list(windows.keys())[i])[8]: y_val}

        sets_ms.update(set_ms)

    return sets_ms


def extract_imp_feats(X_train, X_val, y_train, threshold_value, original):
    """
    Runs a random forest on the train and validation set,
    then extracts the important features using SelectFromModel() function.

    Parameters
    ----------
        threshold_value: the threshold above which a variable is deemed
            'important'
        original: the original X dataframe, from which feature names can be
            retrieved.

    Returns
    -------
        strong_features: indices of important features
        X_train_imp: X_train dataframe consisting of only important variables
        X_val_imp: X_val dataframe consisting of only important variables
    note: 'indices' can be used later to find important column names:
        X.columns[indices]
    """

    # run random forest for feature selection
    rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1)

    # select variables with an importance greater than 'threshold_value'
    sfm = SelectFromModel(rf, threshold=threshold_value)
    sfm.fit(X_train, y_train)

    # transform X to contain only 'important variables'
    X_train_imp = sfm.transform(X_train)
    X_val_imp = sfm.transform(X_val)

    # extract the important variables for use in out of sample testing
    strong_features = []  # indices of the selected features
    for feature in sfm.get_support(indices=True):
        strong_features.append(feature)

    return strong_features, X_train_imp, X_val_imp


def optimal_threshold(threshold_values, train_test_sets,
                      sets_model_selection, original_df):
    """
    Determines the optimal 'importance' threshold by selecting that which
    minimises the Random Forest out-of-bag error for the validation set.

    Parameters
    ----------
        threshold_values (list): an array of thresholds
            e.g. [0.005, ..., 0.009]
        train_test_sets (dict of matrices): dict containing the train and test
            matrices for each sliding window.
        sets_model_selection (dict of matrices): dict containing the train and
            validation sets for each sliding window.
        original_df: the original X dataframe, from which feature names can be
            retrieved

    Returns
    -------
        min_idx: index of threshold that minimises oob error
        df_errors: a dataframe containing errors for each sliding window and
            threshold setting
    """
    threshold_errors = {}  # dict to hold the oob error rates for each SW

    # return important features for each SW, for a given thresh. value
    for threshold in enumerate(threshold_values):  # for each threshold value
        indices_important = {}

        error_rates = []  # store sliding window errors each threshold

        for i in np.arange(0, len(train_test_sets), 4):

            # assign the train and validation sets for the given sliding window

            X_train = list(sets_model_selection.values())[i]
            X_val = list(sets_model_selection.values())[i+1]
            y_train = list(sets_model_selection.values())[i+2]
            y_val = list(sets_model_selection.values())[i+3]

            # feat selection using RF, returning reduced train and val sets
            feature_index, X_train, X_val = extract_imp_feats(
                X_train,
                X_val,
                y_train,
                threshold_value=threshold[1],
                original=original_df)

            # store the important indices for the sliding window in question
            _ = {'indices_' +
                 str(list(sets_model_selection.keys())[i])[8]: feature_index}

            # update the dict with the indices of important vars for each SW
            indices_important.update(_)

            # evaluate performance of these variables using a standard RF
            rf = RandomForestClassifier(n_estimators=1000, oob_score=True,
                                        n_jobs=-1)

            # X_train is now the reduced dataframe (only imp. vars)
            rf.fit(X_train, y_train)
            oob_error = 1 - rf.oob_score_
            error_rates.append(oob_error)

        # update dictionary with errors for each sliding window
        threshold_errors.update(
            {str(threshold[1])+"_threshold": error_rates})
        df_errors = pd.DataFrame(list(threshold_errors.values())).T

        # returns the index of threshold that minimises oob error for each SW
        min_idx = df_errors.idxmin(axis=1)

    return min_idx, df_errors


def extract_imp_features(X, sets, sets_ms, threshold_settings, opt_thresh_idx):
    """
    Extract important features (their indices) using the thresholds determined
    by optimal_threshold().

    E.g. if there are 4 sliding windows, each with the 4 sets required for
    model selection:

    sets_ms = dict_keys(['X_train_0_a', 'X_val_0', 'y_train_0_a', 'y_val_0',
                         'X_train_1_a', 'X_val_1', 'y_train_1_a', 'y_val_1',
                         'X_train_2_a', 'X_val_2', 'y_train_2_a', 'y_val_2',
                         'X_train_3_a', 'X_val_3', 'y_train_3_a', 'y_val_3'])

    Pass the dicts associated with each sliding window to extract_imp_feats().

    Parameters
    ----------
        X (df): the original X dataframe, from which feature names can be
            retreived
        sets (dict of dfs): Dict containing the train and test sets
            associated with each sliding window. Generated by
            walkforward_split().
        sets_ms (dict of matrices): Dict containing the train and validation
            indices for a given asset.
        threshold_settings (list): list of thresholds e.g. [0.005, ..., 0.009].
        opt_thresh_idx (int): the index of threshold settings found to be
            optimal via optimal_threshold().
    Returns
    -------
        indices (dict): a dictionary containing the lists of feature indices
            deemed to be important for each sliding window.
    """

    indices = {}

    # step through the sets 4 at a time, becauase there are 4 dicts
    # assoc. w/ each sliding window (X_train_val, X_val, y_train_val, y_train)
    for idx, i in enumerate(np.arange(0, len(sets), 4)):
        # assign the train and validation sets for the given sliding window
        X_train = list(sets_ms.values())[i]
        X_val = list(sets_ms.values())[i+1]
        y_train = list(sets_ms.values())[i+2]
        y_val = list(sets_ms.values())[i+3]  # 4th

        window_num = str(list(sets_ms.keys())[i])[8]

        # perform feature selection using RF
        feature_index, \
            sets_ms['X_train_' + window_num + '_' + 'a'], \
            sets_ms['X_val_' + window_num], \
            = extract_imp_feats(
                X_train,
                X_val,
                y_train,
                threshold_value=threshold_settings[opt_thresh_idx[idx]],
                original=X)

        imp_indices = {'indices_' + window_num: feature_index}
        indices.update(imp_indices)
    return indices


def create_model(optimizer='adam', neurons=1, learn_rate=0.01,
                 momentum=0, kernel_initializer='normal',
                 dropout_rate=0.0, input_dim=None):
    """
    Creates a simple Keras Sequential model that will be passed to
    KerasClassifier object.

    Parameters
    ----------
        optimizer (str): required to compile a Keras model. See
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
            for an exhaustive list of optimizers.
        neurons (int):
        learn_rate (float): learning rate of chosen optimizer.
        dropout_rate (float): probability that a random node will be dropped
            in each weight update cycle.
        input_dim (int): number of features in dataset
    Returns
    -------

    """
    with tf.device("/device:GPU:0"):
        # create model
        model = Sequential()
        model.add(Dense(neurons, input_dim=input_dim, activation='relu',
                        kernel_initializer=kernel_initializer))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        return model


def model_selection(model, train_test_sets, sets_model_selection,
                    param_distributions, n_jobs=1, n_iter=100, mlp=None):
    """
    Performs model selection using random grid search without cross validation.

    Parameters
    ----------
        model (obj): enter model such as RandomForestClassifier() (default)
        train_test_sets (dict of matrices): dict containing the train and test
            matrices for each sliding window.
        sets_model_selection (dict of matrices): dict containing the train and
            validation sets for each sliding window.
        param_distributions (dict): pre-defined grid to search over, specific
            to the input 'model'.
        n_iter (int): Number of parameter settings that are sampled. n_iter
            trades off runtime vs quality of the solution.
        mlp (bool): True/False input indicating whether the model is a
            multilayer perceptron. If True, prompts the building of a Keras
            classifier with the appropriate input dimensions as determined via
            feature selection.
    Returns
    -------
        optimal_models (dict): a dict containing optimal models for each
            sliding window
        optimal_params (dict): a dict containing optimal hyper-parameters for
            each optimal model
    """
    # dictionary to hold optimal models for each sliding window
    optimal_models = {}
    optimal_params = {}

    for i in np.arange(0, len(train_test_sets), 4):
        # assign the train and validation sets for the given sliding window
        X_train = list(sets_model_selection.values())[i]
        X_val = list(sets_model_selection.values())[i+1]
        y_train = list(sets_model_selection.values())[i+2]
        y_val = list(sets_model_selection.values())[i+3]

        # normalise the data if mlp
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # set up the inputs for PredefinedSplit;
        # will be input to RandomizedSearchCV
        my_train_fold = []
        for j in range(len(list(sets_model_selection.values())[i])):
            # -1 means the sample will be in the train set
            my_train_fold.append(-1)

        my_test_fold = []
        for j in range(len(list(sets_model_selection.values())[i+1])):
            # 0 means the sample will be in the validation set
            my_test_fold.append(0)

        my_fold = my_train_fold + my_test_fold

        ps = PredefinedSplit(test_fold=np.asarray(my_fold))

        # input dimensions for each MLP model will vary, depending on the
        # sliding window (due to feature selection)
        if mlp:
            input_dims = list(sets_model_selection.values())[i].shape[1]
            model = KerasClassifier(
                build_fn=create_model,
                input_dim=input_dims,
                verbose=0)

        # set up the grid search
        mdl_opt = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=ps,
            verbose=1,
            n_jobs=n_jobs)  # note, with sliding window, use 5 fold

        # Fit the random search model: parameter combinations will be trained,
        # then tested on the validation set
        mdl_opt.fit(
            np.concatenate((X_train, X_val), axis=0),
            np.concatenate((y_train.values.ravel(),
                            y_val.values.ravel()), axis=0))

        # return the optimal parameters
        mdl = {'optimal_model_sw' +
               str(list(sets_model_selection.keys())[i])[8]:
               mdl_opt.best_estimator_}
        optimal_models.update(mdl)

        if mlp:
            mdl_opt.best_estimator_.\
                model.save('mlp_model_sw' +
                           str(list(sets_model_selection.keys())[i])[8] +
                           '_fs.h5')

        param = {'optimal_model_sw' +
                 str(list(sets_model_selection.keys())[i])[8]:
                 mdl_opt.best_params_}
        optimal_params.update(param)

        # send the optimal model to the 'optimal models' dictionary
        optimal_models.update(mdl)
        optimal_params.update(param)

        # save the optimal mlp models
        if mlp:
            # clear keras session to speed up next round of random search
            K.clear_session()
            tf.reset_default_graph()

    return optimal_models, optimal_params


def return_validation_metrics(optimal_models, train_test_sets):
    """
    Returns the performance measure(s) of optimal models on their
    respective sliding window.

    Models are evaluated on the validation sets if passed train/validation
    sets. E.g. 'sets_for_model_selection_spx'

    Parameters
    ----------
        optimal_models (dict): the optimal models for the respective
            classifier/asset combination
        train_test_sets (dict of matrices): dict containing the train and test
            matrices for each sliding window.

    Returns
    -------
        all_metrics (dict of dicts): A dictionary of dictionaries containing
            performance measures
                E.g. metrics_0 : acc, auc... , metrics_1 : auc, acc... ,
                     metrics_2 : auc, acc...
        y_tests (dict of dicts): A dictionary of dictionaries containing test
            set values
        y_preds (dict of dicts): A dictionary of dictionaries containing preds
        y_scores (dict of dicts): A dictionary of dictionaries containing
            scores
    """

    all_metrics = {}
    y_tests = {}
    y_preds = {}
    y_scores = {}

    for i in np.arange(0, len(train_test_sets), 4):  # for each sliding window
        # define the relevant training and test sets, and model
        X_train = list(train_test_sets.values())[i]
        X_test = list(train_test_sets.values())[i+1]
        y_train = list(train_test_sets.values())[i+2]
        y_test = list(train_test_sets.values())[i+3]

        # normalise the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf_number = int(str(list(train_test_sets.keys())[i])[8])

        # extract the relevant model from optimal model dictionary
        clf = list(optimal_models.values())[clf_number]

        # fit the model
        clf.fit(X_train, y_train.values.ravel())

        # make predictions
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1]

        # output performance metrics
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test,
            y_score,
            drop_intermediate=False,
            pos_label=1)
        acc = metrics.accuracy_score(y_test, y_pred)  # accuracy
        auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test,
            y_score)
        f1 = metrics.f1_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # http://scikit-learn.org/stable/modules/classes.html
        perf_metrics = {'metrics_sw' +
                        str(list(train_test_sets.keys())[i][8]): {
                            'acc': acc,
                            'auc': auc,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'cr': class_report,
                            'tpr': tpr,
                            'fpr': fpr}
                        }
        y_test_temp = {'y_test_' +
                       str(list(train_test_sets.keys())[i][8]): y_test}
        y_pred_temp = {'y_pred_' +
                       str(list(train_test_sets.keys())[i][8]): y_pred}
        y_score_temp = {'y_score_' +
                        str(list(train_test_sets.keys())[i][8]): y_score}

        all_metrics.update(perf_metrics)
        y_tests.update(y_test_temp)
        y_preds.update(y_pred_temp)
        y_scores.update(y_score_temp)

    return all_metrics, y_tests, y_preds, y_scores


def return_validation_metrics_VC(sets_model_selection, train_test_sets,
                                 optimal_weights, rf_optimal, mlp_optimal,
                                 xgb_optimal):
    """
    Returns the performance measure(s) of optimal models on their respective
    sliding window. Models are evaluated on the test sets.

    Pass in the optimal models for the respective classifier/asset combination.

    Parameters
    ----------
        sets_model_selection (dict): dict containing the train and validation
            sets for each sliding window.
        train_test_sets (sets):
        optimal_weights (dict): a dictionary w/ four vectors of optimal
            weights, one for each sliding window.
        [rf/mlp/xgb]_optimal: optimal models for a given asset
            E.g. 'rf_spx_optimal_models'

    Returns
    -------
        all_metrics (dict of dicts): A dictionary of dictionaries containing
            performance measures
                E.g. metrics_0 : acc, auc... , metrics_1 : auc, acc... ,
                     metrics_2 : auc, acc...
        y_tests (dict of dicts): A dictionary of dictionaries containing test
            set values
        y_preds (dict of dicts): A dictionary of dictionaries containing preds
        y_scores (dict of dicts): A dictionary of dictionaries containing
            scores
    """
    all_metrics = {}
    y_tests = {}
    y_preds = {}
    y_scores = {}

    for i in np.arange(0, len(sets_model_selection), 4):  # for each SW
        sliding_window = int(str(list(train_test_sets.keys())[i])[8])
        clf1 = list(rf_optimal.values())[sliding_window]  # RFClassifier
        clf2 = list(mlp_optimal.values())[sliding_window]  # KerasClassifier
        clf3 = list(xgb_optimal.values())[sliding_window]  # XGBoostClassifier

        # define the relevant training and test sets, and model
        X_train = list(sets_model_selection.values())[i]
        X_test = list(sets_model_selection.values())[i+1]
        y_train = list(sets_model_selection.values())[i+2]
        y_test = list(sets_model_selection.values())[i+3]

        # normalise the data since MLP is a voter
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # extract optimal weights for given sliding window
        vc_weights = list(optimal_weights.values())[sliding_window]

        eclf = VotingClassifier(estimators=[('rf', clf1),
                                            ('mlp', clf2),
                                            ('xgb', clf3)],
                                voting='soft',
                                weights=vc_weights,
                                n_jobs=1)

        # fit the model
        eclf.fit(X_train, y_train.values.ravel())

        # make predictions
        y_pred = eclf.predict(X_test)
        y_score = eclf.predict_proba(X_test)[:, 1]

        # output performance metrics
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test,
            y_score,
            drop_intermediate=False,
            pos_label=1)
        acc = metrics.accuracy_score(y_test, y_pred)  # accuracy
        auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test,
            y_score)
        f1 = metrics.f1_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # http://scikit-learn.org/stable/modules/classes.html
        perf_metrics = {'metrics_sw' +
                        str(list(train_test_sets.keys())[i][8]): {
                            'acc': acc,
                            'auc': auc,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'cr': class_report,
                            'tpr': tpr,
                            'fpr': fpr}
                        }

        y_test_temp = {'y_test_' +
                       str(list(train_test_sets.keys())[i][8]): y_test}
        y_pred_temp = {'y_pred_' +
                       str(list(train_test_sets.keys())[i][8]): y_pred}
        y_score_temp = {'y_score_' +
                        str(list(train_test_sets.keys())[i][8]): y_score}

        all_metrics.update(perf_metrics)
        y_tests.update(y_test_temp)
        y_preds.update(y_pred_temp)
        y_scores.update(y_score_temp)

    return all_metrics, y_tests, y_preds, y_scores


def tune_weights_ms(train_test_sets, sets_model_selection,
                    rf_optimal, mlp_optimal, xgb_optimal):
    """
    This determines the optimal weight for each classifier in
    the voting classifier ensemble.

    Assumes that optimal RF, MLP and XGB models are already saved in
    their respective dictionaries such as 'rf_spx_optimal_models',
    'mlp_spx_optimal_models', and 'xgb_spx_optimal_models'. These
    dictionaries are created using model_selection().

    Parameters
    ----------
        train_test_sets: the dictionary containing the train/test split
            for each window E.g. 'set_spx'
        sets_model_selection: dict containing the train and validation
            sets for each sliding window.
                (e.g. 'sets_for_model_selection_us5yr')
        [rf/mlp/xgb]_optimal: optimal models for a given asset
            E.g. 'rf_spx_optimal_models'

    Returns
    -------
        optimal weights: a dictionary that contains four vectors of optimal
            weights, one for each sliding window
    """
    # set up list of weights
    weights = []

    for w1 in range(1, 4):
        for w2 in range(1, 4):
            for w3 in range(1, 4):
                weights.append([w1, w2, w3])

    # dictionary to hold optimal models for each sliding window
    optimal_weights = {}

    for i in np.arange(0, len(train_test_sets), 4):  # i.e. for each SW
        # set up the voting classifier to be tuned
        sliding_window = int(str(list(train_test_sets.keys())[i])[8])
        clf1 = list(rf_optimal.values())[sliding_window]  # RFClassifier
        clf2 = list(mlp_optimal.values())[sliding_window]  # KerasClassifier
        clf3 = list(xgb_optimal.values())[sliding_window]  # XGBoostClassifier

        acc_all = []

        for w1 in range(1, 4):
            for w2 in range(1, 4):
                for w3 in range(1, 4):
                    # assign the train and validation sets for the given SW
                    X_train = list(sets_model_selection.values())[i]
                    X_val = list(sets_model_selection.values())[i+1]
                    y_train = list(sets_model_selection.values())[i+2]
                    y_val = list(sets_model_selection.values())[i+3]

                    # normalise the data since MLP is a voter
                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_val = scaler.transform(X_val)

                    eclf = VotingClassifier(
                        estimators=[('rf', clf1),
                                    ('mlp', clf2),
                                    ('xgb', clf3)],
                        # prediction based on the argmax(sums pred probs)
                        voting='soft',
                        weights=[w1, w2, w3],
                        n_jobs=1)

                    eclf = eclf.fit(X_train, y_train.values.ravel())
                    y_pred = eclf.predict(X_val)
                    acc = metrics.accuracy_score(y_val, y_pred)  # accuracy
                    acc_all.append(acc)

        # identify the optimal VC weights for each sliding window
        weight_vector = {
            'optimal_weights_sw' +
            str(list(sets_model_selection.keys())[i])[8]:
                # get weight mix associated with highest accuracy
                weights[acc_all.index(max(acc_all))]}

        # send the optimal model to the 'optimal models' dictionary
        optimal_weights.update(weight_vector)

    return optimal_weights


def filter_imp_vars(sets, important_cols):
    """
    When passed a dict containing X_train and X_test sets
    of the sliding windows, this will remove features that
    aren't important, as determined by extract_imp_features().

    Parameters
    ----------
        sets (dict of matrices): dict containing the train
            and test matrices for each sliding window.
            These sets are generated by walkforward_split().
        important_cols (dict): a dictionary containing the
            lists of feature indices deemed to be important
            for each sliding window.

    Returns
    -------
        sets_filtered (dict of matrices): dict containing the
            train and test matrices for each sliding window,
            with only important features included.
    """

    sets_for_filtering = []

    # get X_train and X_test for each sliding window
    for set in np.arange(0, len(sets), 2):
        sets_for_filtering.append(list(sets.keys())[set])

    # cycle through the X_train and X_tests,
    # leaving the important variables for given sliding window.
    for set in enumerate(sets_for_filtering):
        # take the number from the end of the set's name e.g. "X_train_0" = 0
        # this tells us which key from dict "indices" to use
        indicie_to_use = int(str(set[1])[-1])

        # access only important variables for sliding window
        sets[set[1]] = sets[set[1]]\
            .iloc[:, list(important_cols.values())[indicie_to_use]]

    return sets


def return_final_metrics(optimal_models, train_test_sets, mlp=None):
    """
    Returns the performance measure(s) of optimal models on their
    respective sliding window. Models are evaluated on the test sets.

    Parameters
    ----------
        optimal_models (dict of objects): dict containing the optimal
            models for the respective classifier/asset combination.
        train_test_sets (dict of matrices): dict containing the train
            and test matrices for each sliding window.

    Returns
    -------
        all_metrics (dict of dicts): A dictionary of dictionaries containing
            performance measures
                E.g. metrics_0 : acc, auc... , metrics_1 : auc, acc... ,
                     metrics_2 : auc, acc...
        y_tests (dict of dicts): A dictionary of dictionaries containing test
            set values
        y_preds (dict of dicts): A dictionary of dictionaries containing preds
        y_scores (dict of dicts): A dictionary of dictionaries containing
            scores
    """

    all_metrics = {}
    y_tests = {}
    y_preds = {}
    y_scores = {}

    # for each sliding window
    for i in np.arange(0, len(train_test_sets), 4):

        # define the relevant training and test sets, and model
        X_train = list(train_test_sets.values())[i]
        X_test = list(train_test_sets.values())[i+2]
        y_train = list(train_test_sets.values())[i+1]
        y_test = list(train_test_sets.values())[i+3]

        # normalise the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf_number = int(str(list(train_test_sets.keys())[i])[8])

        # extract the relevant model from optimal model dictionary
        clf = list(optimal_models.values())[clf_number]

        # fit the model
        clf.fit(X_train, y_train.values.ravel())

        # make predictions
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1]

        # output performance metrics
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test,
            y_score,
            drop_intermediate=False,
            pos_label=1)
        acc = metrics.accuracy_score(y_test, y_pred)  # accuracy
        auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = metrics.\
            precision_recall_curve(y_test, y_score)
        f1 = metrics.f1_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # http://scikit-learn.org/stable/modules/classes.html
        # create a dict to hold metrics for the current sliding window
        perf_metrics = {'metrics_sw' +
                        str(list(train_test_sets.keys())[i][8]): {
                            'acc': acc,
                            'auc': auc,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'cr': class_report,
                            'tpr': tpr,
                            'fpr': fpr}
                        }
        y_test_temp = {'y_test_' +
                       str(list(train_test_sets.keys())[i][8]): y_test}
        y_pred_temp = {'y_pred_' +
                       str(list(train_test_sets.keys())[i][8]): y_pred}
        y_score_temp = {'y_score_' +
                        str(list(train_test_sets.keys())[i][8]): y_score}

        all_metrics.update(perf_metrics)
        y_tests.update(y_test_temp)
        y_preds.update(y_pred_temp)
        y_scores.update(y_score_temp)

    return all_metrics, y_tests, y_preds, y_scores


def return_final_metrics_VC(train_test_sets, optimal_weights, rf_optimal,
                            mlp_optimal, xgb_optimal):
    """
    Modified version of return_final_metrics_VC() which returns
    the performance measure(s) of the optimal VC model (which itself is
    made up of the optimal rf, mlp and xgb models determined via model
    selection). Metrics are returned for each sliding window.
    Models are evaluated on the test sets.

    Parameters
    ----------
        train_test_sets (dict of matrices): dict containing the train
            and test matrices for each sliding window.
        optimal_weights (dict of dicts): the optimal weights for the voting
            classifier for each sliding window. From tune_weights_ms().
        rf_optimal (dict of objects): optimal RFs for each SW, from model
            selection
        mlp_optimal (dict of objects): optimal MLPs for each SW, from model
            selection
        xgb_optimal (dict of objects): optimal XGBs for each SW, from model
            selection

        optimal_models (dict of objects): dict containing the optimal
            models for the respective classifier/asset combination.

    Returns
    -------
        all_metrics (dict of dicts): A dictionary of dictionaries containing
            performance measures
                E.g. metrics_0 : acc, auc... , metrics_1 : auc, acc... ,
                     metrics_2 : auc, acc...
        y_tests (dict of dicts): A dictionary of dictionaries containing test
            set values
        y_preds (dict of dicts): A dictionary of dictionaries containing preds
        y_scores (dict of dicts): A dictionary of dictionaries containing
            scores
    """

    all_metrics = {}
    y_tests = {}
    y_preds = {}
    y_scores = {}

    for i in np.arange(0, len(train_test_sets), 4):  # for each sliding window

        sliding_window = int(str(list(train_test_sets.keys())[i])[8])
        clf1 = list(rf_optimal.values())[sliding_window]  # RRFClassifier
        clf2 = list(mlp_optimal.values())[sliding_window]  # KerasClassifier
        clf3 = list(xgb_optimal.values())[sliding_window]  # XGBoostClassifier

        # define the relevant training and test sets, and model
        X_train = list(train_test_sets.values())[i]
        X_test = list(train_test_sets.values())[i+2]
        y_train = list(train_test_sets.values())[i+1]
        y_test = list(train_test_sets.values())[i+3]

        # normalise the data since MLP is a voter
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # extract optimal weights for given sliding window
        vc_weights = list(optimal_weights.values())[sliding_window]

        eclf = VotingClassifier(estimators=[('rf', clf1),
                                            ('mlp', clf2),
                                            ('xgb', clf3)],
                                voting='soft',
                                weights=vc_weights,
                                n_jobs=1)

        # fit the model
        eclf.fit(X_train, y_train.values.ravel())

        # make predictions
        y_pred = eclf.predict(X_test)
        y_score = eclf.predict_proba(X_test)[:, 1]

        # output performance metrics
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test,
            y_score,
            drop_intermediate=False,
            pos_label=1)
        acc = metrics.accuracy_score(y_test, y_pred)  # accuracy
        auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test, y_score)
        f1 = metrics.f1_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # http://scikit-learn.org/stable/modules/classes.html
        perf_metrics = {'metrics_sw' +
                        str(list(train_test_sets.keys())[i][8]): {
                            'acc': acc,
                            'auc': auc,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'cr': class_report,
                            'tpr': tpr,
                            'fpr': fpr}
                        }

        y_test_temp = {'y_test_' +
                       str(list(train_test_sets.keys())[i][8]): y_test}
        y_pred_temp = {'y_pred_' +
                       str(list(train_test_sets.keys())[i][8]): y_pred}
        y_score_temp = {'y_score_' +
                        str(list(train_test_sets.keys())[i][8]): y_score}

        all_metrics.update(perf_metrics)
        y_tests.update(y_test_temp)
        y_preds.update(y_pred_temp)
        y_scores.update(y_score_temp)
    return all_metrics, y_tests, y_preds, y_scores
