#!/usr/bin/env python3

"""Plotting of trained BinaryStringClassifier models"""

import logging
import importlib.resources as pkg_resources

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from model import ModelLoader
from sklearn.metrics import roc_auc_score, roc_curve


PKG_STYLESHEET = pkg_resources.open_binary('BinaryStringClassifier', 'mrdark.mplstyle').name


class ModelPlotter(object):
    """Create plots for a given model,
    
    Attributes:
        logger (logging.Logger, optional): Standard Python logger
        model (ModelLoader): Loaded BinaryStringClassifier model class
    """
    def __init__(self, model: ModelLoader, logger:logging.Logger=logging.getLogger('ModelPlotter'), plot_style=PKG_STYLESHEET):
        """Summary
        
        Args:
            model (ModelLoader): Loaded BinaryStringClassifier model class
            logger (logging.Logger, optional): Standard Python logger
            plot_style (str, optional): mplstyle name or configuration file
        """

        plt.style.use(plot_style)
        self.model = model
        self.logger = logger

    def loss_by_epoch(self, show:bool=False, path_plot:str='loss_by_epoch.png'):
        """Plot loss over epoch
        
        Args:
            show (bool, optional): Whether to display figure in GUI after creation
            path_plot (str, optional): Path to save figure to
        """
        plt.clf()
        loss = self.model.history['loss']
        val_loss = self.model.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        self.logger.info(f"Saving plot to {path_plot}")
        plt.savefig(path_plot)
        if show:
            plt.show()

    def accuracy_by_epoch(self, show:bool=False, path_plot:str='accuracy_by_epoch.png'):
        """Plot accuracy over epoch
        
        Args:
            show (bool, optional): Whether to display figure in GUI after creation
            path_plot (str, optional): Path to save figure to
        """
        plt.clf()
        acc = self.model.history['accuracy']
        val_acc = self.model.history['val_accuracy']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, label='Training acc')
        plt.plot(epochs, val_acc, label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        self.logger.info(f"Saving plot to {path_plot}")
        plt.savefig(path_plot)
        if show:
            plt.show()


def plot_confusion_matrix_proportional(confusion_matrix: np.array, path_plot:str='confusion_matrix.png', logger:logging.Logger=logging.getLogger('ModelPlotter'), plot_style=PKG_STYLESHEET):
    """Visualise a confusion matrix as proportions of prediction categories
    
    Args:
        confusion_matrix (np.array): Output of sklearn.metrics.confusion_matrix
        path_plot (str, optional): Path to write figure to
        logger (logging.Logger, optional): Standard Python logger
        plot_style (TYPE, optional): Matplotlib style argument or file
    """
    conf_matrix_prop = confusion_matrix / confusion_matrix.astype(np.float).sum(axis=0)
    plt.clf()
    plt.title('Confusion Matrix (Proportional by Prediction)')
    plt.xlabel('Prediction')
    plt.ylabel('Reality')
    plt.xticks([0,1], ['Non-SMILES', 'SMILES'])
    plt.yticks([0,1], ['Non-SMILES', 'SMILES'])
    plt.yticks(rotation=90, va='center')
    plt.imshow(conf_matrix_prop, cmap='Blues')
    plt.grid(False)
    plt.colorbar()
    logger.info(f"Saving plot to {path_plot}")
    plt.savefig(path_plot)


def plot_roc_curves(model: tuple, predictions: pd.DataFrame, path_plot: str = "roc_curve.png"):
    """Create a ROC curve for each model against the provided feature & target set.
    
    Args:
        model (tuple): Trained BinaryStringClassifier model
        predictions (pd.DataFrame): Test data features, target, model predictions and probabilities
        filename (str, optional): Name of output file
    """
    plt.figure()
    logit_roc_auc = roc_auc_score(predictions['Category'], predictions['Probability'])
    fpr, tpr, thresholds = roc_curve(predictions['Category'], predictions['Probability'])
    plt.plot(fpr, tpr, marker='.', label=f"AUC: {logit_roc_auc:.3}")
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(path_plot)
