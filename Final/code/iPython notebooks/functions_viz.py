import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_charts(data, dim1, dim2, fig_size, y_label=None):
    """
    Generic plotting function to plot multiple features as
    line charts within a facet plot.
    """
    numericals = data.select_dtypes(include=['float64']).columns
    number_of_numericals = data.select_dtypes(
        include=['float64']).columns.shape[0]

    fig, axs = plt.subplots(dim1, dim2, figsize=fig_size)
    fig.tight_layout(pad=1, w_pad=4, h_pad=4)
    axs = axs.ravel()

    for i in range(number_of_numericals):
        axs[i].plot(data[numericals[i]], color='steelblue')
        axs[i].set_title(numericals[i], size=10)
        axs[i].set_xlabel("Weeks")
        axs[i].set_ylabel(y_label)


def plot_class_frequencies(y):
    """
    Plots the class frequencies in a bar chart
    """
    plt.figure(figsize=(8, 6))
    plt.hist(y, color='steelblue', bins=2, rwidth=0.5)
    plt.xticks((0.25, 0.75), ['0', '1'])
    plt.title('Class Frequencies')
    plt.xlabel('Class')
    plt.ylim((0, 1000))
    pct_up = "{0:.1%}".format(round(y.mean(), 3))
    pct_down = "{0:.1%}".format(round(1-y.mean(), 3))
    pct_up_yloc = y.sum() + 30
    pct_down_yloc = (y.count() - y.sum() + 30)
    plt.annotate(pct_down, xy=(0.15, pct_down_yloc),
                 xytext=(0.22, pct_down_yloc), fontsize=15)
    plt.annotate(pct_up, xy=(0.15, pct_up_yloc),
                 xytext=(0.72, pct_up_yloc), fontsize=15)


def plot_oob_scores(df_errors, thresholds, title):
    fig = plt.figure(figsize=(12, 4), dpi=100)
    sns.heatmap(data=df_errors, xticklabels=thresholds, annot=True,
                fmt='.3f', linewidths=0.05)
    fig.suptitle(title, fontsize=20, x=0.44)
    plt.ylabel("Sliding Window", fontsize=15)
    plt.xlabel("Threshold", fontsize=15)


# plot a count of variables in each sliding window
def plot_imp_vars(df, indices, asset_label):
    """
    Plots frequency at which a variable was deemed important
    over the N sliding windows.

    inputs:
        df (df): input dataframe for asset class in question
        indices (dict of dicts): contains the important variable
            indices determined by extract_imp_features().

    returns:
        Bar chart displaying the number of sliding windows
        over which each variable was deemed to be important.
    """

    df = df.drop(['Date', 'Movement'], axis=1)
    df_0 = df.columns[indices['indices_0']]
    df_1 = df.columns[indices['indices_1']]
    df_2 = df.columns[indices['indices_2']]
    df_3 = df.columns[indices['indices_3']]

    df_0 = [(var, 1) for idx, var in enumerate(df_0)]
    df_1 = [(var, 1) for idx, var in enumerate(df_1)]
    df_2 = [(var, 1) for idx, var in enumerate(df_2)]
    df_3 = [(var, 1) for idx, var in enumerate(df_3)]

    df_0 = pd.DataFrame(df_0).T
    df_1 = pd.DataFrame(df_1).T
    df_2 = pd.DataFrame(df_2).T
    df_3 = pd.DataFrame(df_3).T

    df_0.columns = df_0.iloc[0]
    df_0 = df_0[1:]
    df_1.columns = df_1.iloc[0]
    df_1 = df_1[1:]
    df_2.columns = df_2.iloc[0]
    df_2 = df_2[1:]
    df_3.columns = df_3.iloc[0]
    df_3 = df_3[1:]

    dfs = pd.concat([df_0, df_1, df_2, df_3], axis=0)
    dfs = dfs.reset_index(drop=True)
    dfs_count = [np.sum(dfs)[i] for i in np.arange(len(dfs.T))]
    dfs_count = pd.Series(dfs_count, index=dfs.T.index)
    dfs_count = dfs_count.sort_values(ascending=False)

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(23, 5))
    dfs_count.plot.bar(width=1, alpha=0.4, color='steelblue')
    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    plt.yticks((1, 2, 3, 4))
    plt.title(asset_label + " - No. of sliding windows for"
              + "which each variable meets importance threshold")
    plt.xlabel("Variable")


def plot_accuracy(rf_metrics, mlp_metrics, xgb_metrics, vc_metrics):
    """
    Plots a heatmap of accuracies of dimensions
        (# classifiers x # sliding windows).
    """
    acc = []

    no_windows = len(rf_metrics.keys())

    for i in range(0, no_windows):
        acc.append(round(rf_metrics['metrics_sw'+str(i)]['acc'], 3))
    for i in range(0, no_windows):
        acc.append(round(mlp_metrics['metrics_sw'+str(i)]['acc'], 3))
    for i in range(0, no_windows):
        acc.append(round(xgb_metrics['metrics_sw'+str(i)]['acc'], 3))
    for i in range(0, no_windows):
        acc.append(round(vc_metrics['metrics_sw'+str(i)]['acc'], 3))

    acc = np.reshape(acc, (4, 4))
    models = ['Random Forest', 'Neural Network',
              'XGBoost', 'Voting Classifier']

    fig = plt.figure(figsize=(12, 4))
    sns.heatmap(data=acc, yticklabels=models, annot=True, fmt='.3f',
                linewidths=0.05, cmap='RdYlGn', annot_kws={"size": 20})
    fig.suptitle('Accuracy', fontsize=20, x=0.44)
    plt.ylabel("Model", fontsize=15)
    plt.xlabel("Test Set / Sliding Window", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)


def plot_auc(rf_metrics, mlp_metrics, xgb_metrics, vc_metrics):
    """
    Plots a heatmap of AUC of dimensions (# classifiers x # sliding windows).
    """
    auc = []

    no_windows = len(rf_metrics.keys())

    for i in range(0, no_windows):
        auc.append(round(rf_metrics['metrics_sw'+str(i)]['auc'], 3))
    for i in range(0, no_windows):
        auc.append(round(mlp_metrics['metrics_sw'+str(i)]['auc'], 3))
    for i in range(0, no_windows):
        auc.append(round(xgb_metrics['metrics_sw'+str(i)]['auc'], 3))
    for i in range(0, no_windows):
        auc.append(round(vc_metrics['metrics_sw'+str(i)]['auc'], 3))

    auc = np.reshape(auc, (4, 4))
    models = ['Random Forest', 'Neural Network',
              'XGBoost', 'Voting Classifier']

    fig = plt.figure(figsize=(12, 4))
    sns.heatmap(data=auc, yticklabels=models, annot=True, fmt='.3f',
                linewidths=0.05, cmap='RdYlGn', annot_kws={"size": 20})
    fig.suptitle('Receiver Operator Characteristic, Area Under Curve (AUC)',
                 fontsize=20, x=0.44)
    plt.ylabel("Model", fontsize=15)
    plt.xlabel("Test Set / Sliding Window", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)


def plot_ROC(metrics, asset_label, clf_label):
    """
    Plots an ROC curve for each sliding window.

    Inputs:
        metrics (dict): the performance metrics for a given model. i.e. the
            output from the function return_final_metrics().
        asset_label (str): Name of the asset performance being visualised.
            E.g. 'S&P500'
        clf_labels (str): Name of the classifier performance being visualised.
            E.g. 'Random Forest'

    Returns:
        A matplotlib figure displaying the ROC curve of the asset/classifier
        combination specified in the inputs.
    """
    plt.figure(1, figsize=(8, 8))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.plot(metrics['metrics_sw0']['fpr'],
             metrics['metrics_sw0']['tpr'],
             label='Optimal ' + clf_label +
             ' - Sliding window 1: ' +
             str(float(round(metrics['metrics_sw0']['auc'], 3))))

    plt.plot(metrics['metrics_sw1']['fpr'],
             metrics['metrics_sw1']['tpr'],
             label='Optimal ' + clf_label +
             ' - Sliding window 2: ' +
             str(float(round(metrics['metrics_sw1']['auc'], 3))))

    plt.plot(metrics['metrics_sw2']['fpr'],
             metrics['metrics_sw2']['tpr'],
             label='Optimal ' +
             clf_label + ' - Sliding window 3: ' +
             str(float(round(metrics['metrics_sw2']['auc'], 3))))

    plt.plot(metrics['metrics_sw3']['fpr'],
             metrics['metrics_sw3']['tpr'],
             label='Optimal ' + clf_label +
             ' - Sliding window 4: ' +
             str(float(round(metrics['metrics_sw3']['auc'], 3))))

    plt.legend(loc=4, prop={'size': 11})
    plt.plot([0, 1], [0, 1], color='black')
    plt.title(clf_label + " (" + asset_label + ")")
    plt.show()
