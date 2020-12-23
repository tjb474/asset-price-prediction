import matplotlib.pyplot as plt


def plot_charts(data, dim1, dim2, fig_size, y_label=None):
    """
    Generic plotting function to plot multiple features as
    line charts within a facet plot.
    """
    numericals = data.select_dtypes(include=['float64']).columns
    number_of_numericals = data.select_dtypes(
        include=['float64']).columns.shape[0]

    fig, axs = plt.subplots(dim1, dim2, figsize= fig_size)
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
    plt.figure(figsize =(8,6))
    plt.hist(y, color='steelblue', bins=2, rwidth=0.5)
    plt.xticks((0.25, 0.75), ['0', '1'])
    plt.title('Class Frequencies')
    plt.xlabel('Class')
    plt.ylim((0, 1000))
    
    pct_up = "{0:.1%}".format(round(y.mean(), 3))
    pct_down = "{0:.1%}".format(round(1-y.mean(), 3))

    pct_up_yloc = y.sum() + 30
    pct_down_yloc = (y_spx.count() - y.sum() + 30)
    
    plt.annotate(pct_down, xy=(0.15, pct_down_yloc),
                 xytext=(0.22, pct_down_yloc), fontsize=15)
    plt.annotate(pct_up, xy=(0.15, pct_up_yloc),
                 xytext=(0.72, pct_up_yloc), fontsize=15)