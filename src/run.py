#!/usr/bin/env python3

"""Entrypoint for model training & evaluation"""

__version__ = "0.1.0"

import argparse
import logging
from pathlib import Path
import sys

# Silence TF warnings
if True:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.utils import set_random_seed
import pandas as pd
import matplotlib.pyplot as plt

from model import ModelTrainer, ModelLoader
from plotting import ModelPlotter, PKG_STYLESHEET
from evaluate import ModelEvaluator
from utils import _dir_exists, _file_exists, _valid_output_path

NAME = 'BSC-Model' # No spaces!


# Create argparse components
# Common debug arguments
parser_debug = argparse.ArgumentParser(add_help=False)
parser_debug.add_argument('--debug', action='store_true', help=f'Whether to write the {NAME}.debug file.')
parser_debug.add_argument('--verbose', action='store_true', help=f'Set console logger level to DEBUG.')

# Primary/root parser
parser_root = argparse.ArgumentParser(description=f"{NAME}. Entrypoint for model training & evaluation.", parents=[parser_debug], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser_root.add_subparsers(dest="mode", required=True, title="Modes")

# Subparsers & components
# Shared parser for model loaders
parser_loader = argparse.ArgumentParser(add_help=False)
parser_loader.add_argument(dest='model_to_load', help='Path to exported model for loading.', type=_dir_exists)

# Subparser: train - Model trainer
help_train = "Train a BinaryStringClassifier model."
parser_train = subparsers.add_parser("train", parents=[parser_debug], description=help_train, help=help_train)
parser_train.add_argument(dest='dataset', help='Path to training dataset.', type=_file_exists)  # TODO: Verify dataset here? i.e. contains expected columns
parser_train.add_argument('-d', '--model_directory', help='Directory to save trained model to.', type=_valid_output_path, default='trained_model')
parser_train.add_argument('-t', '--tokenizer_file', help=argparse.SUPPRESS, type=str, default='tokenizer.json')
parser_train.add_argument('-s', '--seed', help='Seed for model reproducibility.', type=int, default=None)

# Subparser: evaluate - Model evaluator
help_plot = "Evaluate a trained BinaryStringClassifier model."
parser_plot = subparsers.add_parser("evaluate", parents=[parser_debug, parser_loader], description=help_plot, help=help_plot)
parser_plot.add_argument('-d', '--test_dataset', help='Test dataset to evaluate against.', type=_file_exists, required=True)
parser_plot.add_argument('-s', '--style', help='Matplotlib style for plots.', type=str, default=PKG_STYLESHEET)

# Subparser: predict
help_predict = "Get SMILES probabilities with a trained BinaryStringClassifier model."
parser_predict = subparsers.add_parser("predict", parents=[parser_debug, parser_loader], description=help_predict, help=help_predict)
parser_predict.add_argument('-t', '--tokenizer_to_load', help='Path to exported tokenizer.', type=_file_exists, default='tokenizer.json')

# Support strings for prediction from EITHER the command line or a provided file
predict_string_input = parser_predict.add_mutually_exclusive_group(required=True)
predict_string_input.add_argument('-s', '--predict_strings', help='Strings for SMILES probability calculation.', type=str, nargs='+')
predict_string_input.add_argument('-f', '--predict_file', help='Files containing strings for SMILES probability calculation.', type=_file_exists)

args = parser_root.parse_args()


# Set up logger
logger = logging.getLogger(NAME)
logger.setLevel(logging.DEBUG)

# Config debug file
if args.debug:
    formatter_logfile = logging.Formatter("[%(levelname)s] %(message)s")
    log_file = logging.FileHandler(f"{NAME}.debug")
    log_file.setLevel(logging.DEBUG)
    log_file.setFormatter(formatter_logfile)
    logger.addHandler(log_file)

# Config console logger
formatter_console = logging.Formatter("[%(levelname)s] %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO if not args.verbose else logging.DEBUG)
console_handler.setFormatter(formatter_console)
logger.addHandler(console_handler)


# Running from commandline
def main():
    logger.debug(args)
    if args.mode == "train":
        # TODO: Verify this, allow user to set
        if args.seed is not None:
            set_random_seed(args.seed)

        model = ModelTrainer(pd.read_csv(args.dataset), logger=logger) #pd.read_csv('../data/combined.csv'))
        model.export_model(args.model_directory)
    elif args.mode == "evaluate":
        plt.style.use(args.style)
        model = ModelLoader(args.model_to_load, logger=logger)
        plotter = ModelPlotter(model, logger=logger, plot_style=args.style)
        plotter.accuracy_by_epoch()
        plotter.loss_by_epoch()

        evaluator = ModelEvaluator(model, test_data=args.test_dataset, logger=logger)
        evaluator.evaluate()
    elif args.mode == "predict":
        model = ModelLoader(args.model_to_load, logger=logger, path_tokenizer=args.tokenizer_to_load)

        # Parse strings for prediction
        if args.predict_strings is not None:
            input_strings = args.predict_strings
        elif args.predict_file is not None:
            input_strings = [line.rstrip() for line in open(args.predict_file).readlines()]
        else:
            raise ValueError(f"No valid strings or file provided for prediction.")

        # Verification
        if not isinstance(input_strings, list):
            raise ValueError(f"Provided input strings is an expected type '{type(input_strings)}'")

        # Prediction loop
        for input_string in input_strings:
            prediction = model.predict(input_string)
            logger.info(f"SMILES Probability for '{data}': {np.round(100*prediction,1)}%")
    else:
        logger.critical(f"Unknown mode '{args.mode}'")

if __name__ == "__main__":
    main()
