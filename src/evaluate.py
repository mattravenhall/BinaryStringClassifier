#!/usr/bin/env python3

"""Evaluation of trained BinaryStringClassifier models
"""

import logging

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd

from model import ModelLoader
from plotting import plot_confusion_matrix_proportional, plot_roc_curves


class ModelEvaluator(object):
    """Evaluate a trained BinaryStringClassifier model
    
    Attributes:
        final_validation_accuracy (TYPE): Description
        logger (logging.Logger, optional): Standard Python logger
        model (ModelLoader): Loaded BinaryStringClassifier model class
        predictions (TYPE): Description
        test_data (TYPE): Description
    """
    def __init__(self, model: ModelLoader, test_data:pd.DataFrame, logger:logging.Logger=logging.getLogger('ModelPlotter')):
        """Summary
        
        Args:
            model (ModelLoader): Loaded BinaryStringClassifier model class
            test_data (pd.DataFrame): Description
            logger (logging.Logger, optional): Standard Python logger
        """

        self.model = model
        self.logger = logger

        # Get predictions
        self.test_data = pd.read_csv(test_data)
        self.predictions = self.get_predictions()

        # Training Accuracy
        self.final_validation_accuracy = self.model.history['val_accuracy'][-1]

    def evaluate(self):
        """Run full evaluation suite
        """
        self.calc_metrics()
        self.logger.info(f"Final Validation Accuracy: {self.final_validation_accuracy:.3f}")
        self.find_misclassifications()

    def get_predictions(self) -> pd.DataFrame:
        """Create a DataFrame detailing model predictions & probabilities for self.test_data
        
        Returns:
            pd.DataFrame: Predictions DataFrame containing String, Category, Prediction, and Probability columns
        """
        predictions = []
        for i, entry in self.test_data.iterrows():
            smiles_probability = self.model.predict(entry['String'])

            predictions.append({
                "String": entry["String"],  # Feature
                "Category": entry["Category"],  # Target
                "Prediction": int(smiles_probability > 0.5),
                "Probability": smiles_probability,
            })

        df_predictions = pd.DataFrame(predictions)
        return df_predictions

    def calc_metrics(self):
        """Calculate & plot various metrics related to model performance (e.g. accuracy, F score, ROC curves).
        """
        # Accuracy
        accuracy = accuracy_score(self.predictions['Category'], self.predictions['Prediction'])

        # Confusion matrix
        conf_matrix = confusion_matrix(self.predictions['Category'], self.predictions['Prediction'])
        plot_confusion_matrix_proportional(conf_matrix, logger=self.logger)

        # P, R, F
        precision, recall, fscore, _ = precision_recall_fscore_support(
            self.predictions['Category'], 
            self.predictions['Prediction'],
            average = "binary"
        )

        # ROC Curve
        plot_roc_curves(self.model, predictions=self.predictions)

        # Log metrics
        self.logger.info(f"Accuracy: {accuracy:.3f}")
        self.logger.info(f"Precision: {precision:.3f}")
        self.logger.info(f"Recall: {recall:.3f}")
        self.logger.info(f"F Score: {fscore:.3f}")

    def find_misclassifications(self):
        """Identify misclassifications and write to txt files
        """
        # Write misclassifications to disk
        misclassified_as_smiles = self.predictions[(self.predictions['Category'] == 0) & (self.predictions['Prediction'] == 1)]
        misclassified_as_nonsmiles = self.predictions[(self.predictions['Category'] == 1) & (self.predictions['Prediction'] == 0)]

        if misclassified_as_smiles.shape[0]:
            misclassified_as_smiles['String'].to_csv('misclassified_as_smiles.txt', index=False, header=None)
        if misclassified_as_nonsmiles.shape[0]:
            misclassified_as_nonsmiles['String'].to_csv('misclassified_as_nonsmiles.txt', index=False, header=None)
