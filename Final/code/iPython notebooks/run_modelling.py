from datetime import datetime
from functions_modelling import (
    walkforward_split,
    model_selection_sets,
    optimal_threshold,
    extract_imp_features,
    model_selection,
    return_validation_metrics,
    return_validation_metrics_VC,
    tune_weights_ms,
    filter_imp_vars,
    return_final_metrics,
    return_final_metrics_VC
)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.backend import set_session
import pickle
from scipy import stats
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import expon
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from numpy import random
import tensorflow as tf
from xgboost.sklearn import XGBClassifier

# import the data from preprocessing
X_path = 'X_spx.sav'
y_path = 'X_spx.sav'
X_spx = pickle.load(open(X_path, 'rb'))
y_spx = pickle.load(open(y_path, 'rb'))

# 3 splits = 4 folds/sliding windows
sets_spx = walkforward_split(
    X=X_spx,
    y=y_spx)

# take each sliding window training set and split it
# into a train and validation set for the purpose of model selection
sets_for_model_selection_spx = model_selection_sets(sets_spx)

# find optimal thresholds for feature selection, for sliding window/asset combo
threshold_settings = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                      0.007, 0.008, 0.009, 0.01, 0.011]

# determine optimal threshold
min_idx_spx, errors_spx = optimal_threshold(
    threshold_values=threshold_settings,
    train_test_sets=sets_spx,
    sets_model_selection=sets_for_model_selection_spx,
    original_df=X_spx)

# use optimal thresh val to keep only important feats for each sliding window
indices_spx = extract_imp_features(
    X=X_spx,
    sets=sets_spx,
    sets_ms=sets_for_model_selection_spx,
    threshold_settings=threshold_settings,
    opt_thresh_idx=min_idx_spx)

# configure the tensorflow session for mlp model
config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=1.0),
    device_count={'GPU': 1},
    log_device_placement=True
)

sess_1 = tf.compat.v1.Session(config=config)

# PARAMETER GRIDS FOR RANDOM SEARCH
# MLP parameter grid
optimizers = ['SGD', 'rmsprop', 'adam']
init = ['glorot_uniform', 'lecun_uniform', 'normal', 'uniform']
lr = expon(0.001, 0.2)
mom = randint(0.0, 0.9)
epochs = randint(5, 40)
batches = randint(30, 150)
dropout_rate = uniform(0.0, 0.9)
neurons = randint(1, 50)
random_grid_mlp = dict(
    epochs=epochs,
    batch_size=batches,
    kernel_initializer=init,
    dropout_rate=dropout_rate,
    learn_rate=lr,
    neurons=neurons)

# XGBoost parameter grid
random_grid_xgb = {"n_estimators": randint(1, 1000),
                   "max_depth": randint(1, 15),
                   "min_child_weight": randint(1, 6),
                   "gamma": uniform(0.01, 0.2)}

# Random Forest parameter grid
n_estimators = randint(1, 5000)  # num trees
max_features = ['auto', 'sqrt']  # feats to consider at each split
max_depth = randint(5, 110)  # max num levels in tree
min_samples_split = randint(2, 10)  # min num of samples req. to split node
min_samples_leaf = randint(1, 10)  # min num samples req. at each leaf in node
bootstrap = [True, False]  # Method of selecting samples for training each tree

random_grid_rf = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
    }

# set num. iterations of random search
iterations = 400

# perform model selection for random forest
rf_spx_optimal_models, rf_spx_optimal_params = model_selection(
    RandomForestClassifier(random_state=0),
    train_test_sets=sets_spx,
    sets_model_selection=sets_for_model_selection_spx,
    param_distributions=random_grid_rf,
    n_iter=iterations,
    n_jobs=-1)

# perform model selection for neural network
mlp_spx_optimal_models, mlp_spx_optimal_params = model_selection(
    mlp=True,
    model=None,
    train_test_sets=sets_spx,
    sets_model_selection=sets_for_model_selection_spx,
    param_distributions=random_grid_mlp,
    n_iter=iterations)

# perform model selection for XGBoost
xgb_spx_optimal_models, xgb_spx_optimal_params = model_selection(
    model=XGBClassifier(random_state=0),
    train_test_sets=sets_spx,
    sets_model_selection=sets_for_model_selection_spx,
    param_distributions=random_grid_xgb,
    n_iter=iterations,
    n_jobs=-1)

# VALIDATION METRICS FOR SPX
# RF
rf_spx_val_metrics_fs, rf_y_tests_val_fs, \
    rf_y_preds_val_fs, rf_y_scores_val_fs = return_validation_metrics(
        rf_spx_optimal_models,
        sets_for_model_selection_spx)

# MLP
mlp_spx_val_metrics_fs, mlp_y_tests_val_fs, \
    mlp_y_preds_val_fs, mlp_y_scores_val_fs = return_validation_metrics(
        mlp_spx_optimal_models,
        sets_for_model_selection_spx)

# XGB
xgb_spx_val_metrics_fs, xgb_y_tests_val_fs, \
    xgb_y_preds_val_fs, xgb_y_scores_val_fs = return_validation_metrics(
        xgb_spx_optimal_models,
        sets_for_model_selection_spx)

# Voting Classifier
# tune the weights for the voting classifier (VC) for SPX
optimal_weights_spx = tune_weights_ms(
    train_test_sets=sets_spx,
    sets_model_selection=sets_for_model_selection_spx,
    rf_optimal=rf_spx_optimal_models,
    mlp_optimal=mlp_spx_optimal_models,
    xgb_optimal=xgb_spx_optimal_models)

# use optimal weight vector to run a VC model and return the validation metrics
vc_spx_val_metrics_fs, vc_spx_y_tests_val_fs, \
    vc_spx_y_preds_val_fs, vc_spx_y_scores_val_fs = return_validation_metrics_VC(
        sets_for_model_selection_spx,
        sets_spx,
        optimal_weights_spx,
        rf_spx_optimal_models,
        mlp_spx_optimal_models,
        xgb_spx_optimal_models)

# remove non-important features from train and test sets of SWs
sets_spx = filter_imp_vars(sets=sets_spx, important_cols=indices_spx)

# get test set performance across the sliding windows by carrying
# optimal models forward, and for each sliding window retrain on
# original 70%, and assess performance on test set (30%)

# RF
rf_spx_all_metrics_fs, rf_spx_y_tests_fs, \
    rf_spx_y_preds_fs, rf_spx_y_scores_fs = return_final_metrics(
        optimal_models=rf_spx_optimal_models,
        train_test_sets=sets_spx)

# MLP
mlp_spx_all_metrics_fs, mlp_spx_y_tests_fs, \
    mlp_spx_y_preds_fs, mlp_spx_y_scores_fs = return_final_metrics(
        optimal_models=mlp_spx_optimal_models,
        train_test_sets=sets_spx)

# XGB
xgb_spx_all_metrics_fs, xgb_spx_y_tests_fs, \
    xgb_spx_y_preds_fs, xgb_spx_y_scores_fs = return_final_metrics(
        optimal_models=xgb_spx_optimal_models,
        train_test_sets=sets_spx)

# VC
vc_spx_all_metrics_fs, vc_spx_y_tests_fs, \
    vc_spx_y_preds_fs, vc_spx_y_scores_fs = return_final_metrics_VC(
        optimal_weights=optimal_weights_spx,
        train_test_sets=sets_spx,
        rf_optimal=rf_spx_optimal_models,
        mlp_optimal=mlp_spx_optimal_models,
        xgb_optimal=xgb_spx_optimal_models)

# save the models
filename = 'rf_spx_optimal_models_fs.sav'
pickle.dump(rf_spx_optimal_models, open(filename, 'wb'))
filename = 'xgb_spx_optimal_models_fs.sav'
pickle.dump(xgb_spx_optimal_models, open(filename, 'wb'))
filename = 'vc_spx_optimal_models_fs.sav'
pickle.dump(vc_spx_optimal_models, open(filename, 'wb'))