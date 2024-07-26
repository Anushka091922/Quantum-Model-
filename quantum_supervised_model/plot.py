import seaborn as sns
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_roc(labels_, Y_pred_, n_classes, title, fig_name):
    # Function to plot the ROC curve for multi-class classification
    from scipy import interp
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc

    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    b = np.zeros((np.array(labels_).size, np.array(labels_).max() + 1))
    b[np.arange(np.array(labels_).size), np.array(labels_)] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(b[:, i], Y_pred_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(b.ravel(), Y_pred_.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.show()

def show_values(pc, fmt="%.2f", **kw):
    '''
    Display values in each cell of the heatmap
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        # Set text color based on the background color
        if np.all(color[:3] > 0.5:
            color = (0.0, 0.0, 0.0)  # Black text on light background
        else:
            color = (1.0, 1.0, 1.0)  # White text on dark background
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Convert figure size from centimeters to inches for matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def heatmap(AUC, fig_name, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Create a heatmap for AUC values
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''
    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # Put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # Set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # Set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # Resize figure
    fig.set_size_inches(cm2inch(figure_width, figure_height))
    plt.show()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()

def plot_classification_report(classification_report, fig_name, title='Classification report', cmap='RdBu'):
    '''
    Plot scikit-learn classification report as a heatmap
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), fig_name, title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

def plot_confusion_matrix(cm, fig_name, classes,
                           normalize=False,
                           title='Confusion matrix',
                           ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()

def plot_loss(epochs, loss, label, title):
    # Function to plot loss over epochs
    plt.semilogy(epochs, loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()

def plot_losses(epochs, losses, labels, title, fig_name):
    # Function to plot multiple loss curves over epochs
    fig = plt.figure()
    for i in losses:
        plt.semilogy(epochs, i[0], label=i[1])
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()

def plot_silhouette(X, y, title='Silhouette Analysis'):
    # Function to plot silhouette analysis
    unique_labels = np.unique(y)
    n_clusters = unique_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y)
    y_lower = 10

    for i in unique_labels:
        # Aggregate silhouette scores for samples belonging to cluster i, and sort them
        ith_silhouette_vals = silhouette_vals[y == i]
        ith_silhouette_vals.sort()

        size_i = ith_silhouette_vals.shape[0]
        y_upper = y_lower + size_i

        color = plt.cm.viridis(i / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, (y_lower + y_upper) / 2, str(i))
        y_lower = y_upper + 10  # 10 for the space between clusters

    plt.title(title)
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Cluster label')
    plt.axvline(x=silhouette_score(X, y), color="red", linestyle="--")
    plt.show()
    plt.close()

def plot_cluster(X, y, title, n_clusters):
    # Function to plot clustered data
    unique_labels = np.unique(y)
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for i, color in zip(unique_labels, colors):
        plt.scatter(X[y == i, 0], X[y == i, 1], color=color, label=f'Cluster {i}')

    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()
