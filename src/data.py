#!/usr/bin/env python3

"""Prepare and evaluate datasets for the BinaryStringClassifier"""

__version__ = "0.1.0"

from pathlib import Path
from typing import List
import argparse
import logging
import random
import string

import pandas as pd
from sklearn.model_selection import train_test_split

import utils


NAME = 'BSC-Data'   # No spaces!


def generate_random_strings(
    output_file:str = 'generated_dataset.txt',
    n_strings:int = 100,
    string_length:List[int] = (2, 70),
    valid_characters:list = set(set(string.printable) - set(string.whitespace))
    ):
    """Create a file containing one random string per line
    
    Args:
        output_file (str, optional): Path of output file
        n_strings (int, optional): Description
        string_length (List[int], optional): Description
        valid_characters (list, optional): List of characters that strings may contain
    
    Deleted Parameters:
        n_words (int, optional): Number of words (lines) to generate
        word_length (Tuple[int], optional): Minimum and maximum word length
    """
    if not isinstance(output_file, Path):
        output_path = utils._valid_output_path(output_file)
    else:
        output_path = output_file

    if isinstance(valid_characters, set):
        valid_characters = list(valid_characters)

    with open(output_path, 'w') as f:
        for _ in range(n_strings):
            word = ''.join([random.choice(valid_characters) for _ in range(random.randint(*string_length))])
            f.write(word+'\n')


def combine_datasets(files_nonsmiles:List[Path], files_smiles:List[Path], output_file:str='combined.csv'):
    """Combine multiple text files into a model-ready dataset
    
    Args:
        files_nonsmiles (List[Path]): Paths to txt files containing non-SMILES strings
        files_smiles (List[Path]): Paths to txt files containing SMILES strings
        output_file (str, optional): Path to write combined dataset to
    """
    strings_nonsmiles = utils.flatten_list_of_lists(map(utils.file_to_list, files_nonsmiles))
    strings_smiles = utils.flatten_list_of_lists(map(utils.file_to_list, files_smiles))

    output_path = utils._valid_output_path(output_file)

    combined = pd.DataFrame()
    combined['String'] = strings_smiles + strings_nonsmiles
    combined['Category'] = [1]*len(strings_smiles) + [0]*len(strings_nonsmiles)

    combined.to_csv(output_path, index=False)


def summary_statistics(dataframe: pd.DataFrame, logger:logging.Logger=logging.getLogger(NAME)):
    """Generate summary statistics for a given DataFrame
    
    Args:
        dataframe (pd.DataFrame): DataFrame to get statistics for
        logger (logging.Logger, optional): Standard Python logger
    """
    cr_threshold = 2.5

    # Entry counts
    logger.info(f"Total entries: {dataframe.shape[0]:,}")

    # Category counts
    counts_categories = dataframe['Category'].value_counts()
    logger.info(f"Count (Category 0): {counts_categories[0]:,} ({100*counts_categories[0]/counts_categories.sum():.1f}%)")
    logger.info(f"Count (Category 1): {counts_categories[1]:,} ({100*counts_categories[1]/counts_categories.sum():.1f}%)")

    # Flag up possible flag inbalances
    class_ratio = max(counts_categories) / min(counts_categories)
    if class_ratio > cr_threshold:
        logger.warning(f"Category ratio ({class_ratio:.1f}) exceeds threshold ({cr_threshold})")


def evaluate_datasets(datasets:List[Path], logger:logging.Logger=logging.getLogger(NAME)):
    """Provide summary statistics for (and validate) BSC datasets
    
    Args:
        datasets (List[Path]): Datasets to provide summary statistics for
        logger (logging.Logger, optional): Standard Python logger
    """

    EXPECTED_COLUMNS = {'String', 'Category'}
    EXPECTED_CATEGORIES = {0, 1}

    # Serial process datasets
    for dataset in datasets:
        logger.info(f"Generating summary statistics for '{dataset}'")
        # Attempt to load dataset
        try:
            dataframe = pd.read_csv(dataset)
        except:
            logger.critical(f"Failing to load dataset at: '{dataset}'")
            continue

        # Validate dataset
        # Ensure that all expected columns are present
        missing_columns = EXPECTED_COLUMNS - set(dataframe.columns)
        if missing_columns:
            logger.critical(f"{len(missing_columns)} missing column{'s' if len(missing_columns) > 1 else ''}: {', '.join(sorted(missing_columns))}")
            continue

        # Ensure that only valid categories are present
        invalid_categories = set(dataframe['Category'].unique()) - EXPECTED_CATEGORIES
        if invalid_categories:
            logger.critical(f"{len(invalid_categories)} invalid categorie{'s' if len(invalid_categories) > 1 else ''}: {', '.join(sorted(invalid_categories))}")
            continue

        # Identify strings present in both categories
        uniq_strings_by_category = dataframe.groupby('Category').agg(set)['String']
        multiclass_strings = uniq_strings_by_category[0].intersection(uniq_strings_by_category[1])
        if multiclass_strings:
            logger.critical(f"{len(multiclass_strings)} string{'s' if len(multiclass_strings) > 1 else ''} present in both categories: {', '.join(sorted(multiclass_strings))}")
            continue

        summary_statistics(dataframe, logger)


def split_dataset(dataset:Path, logger:logging.Logger=logging.getLogger(NAME), test_size=0.3, seed=None, prefix=None):
    """Perform a train/test split on a given dataset
    
    Args:
        dataset (Path): Dataset for train/test splitting
        logger (logging.Logger, optional): Standard Python logger
    """

    if prefix is None:
        prefix = dataset.with_suffix('')
        logger.warning(f"Output prefix set to '{prefix}'")

    dataframe = pd.read_csv(dataset).drop_duplicates()  # Getting unique to prevent shared train/test values
    X = dataframe['String']
    y = dataframe['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    df_train = pd.DataFrame({'String': X_train, 'Category': y_train})
    df_test = pd.DataFrame({'String': X_test, 'Category': y_test})

    logger.info(f"Generating summary statistics for train dataset")
    summary_statistics(df_train)
    logger.warning(f"Writing train dataset to '{prefix}_train.csv'")
    df_train.to_csv(f"{prefix}_train.csv", index=False)

    logger.info(f"Generating summary statistics for test dataset")
    summary_statistics(df_test)
    logger.warning(f"Writing test dataset to '{prefix}_test.csv'")
    df_test.to_csv(f"{prefix}_test.csv", index=False)


# Parse in arguments
# Common debug arguments
parser_debug = argparse.ArgumentParser(add_help=False)
parser_debug.add_argument('--debug', action='store_true', help=f'Whether to write the {NAME}.debug file.')
parser_debug.add_argument('--verbose', action='store_true', help=f'Set console logger level to DEBUG.')

# Primary/root parser
parser_root = argparse.ArgumentParser(description=f"{NAME}. Entrypoint for dataset creation and curation.", parents=[parser_debug], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser_root.add_subparsers(dest="mode", required=True, title="Modes")

## Subparser for CREATE functionality
help_create = "Create randomly generated datasets within given parameters."
parser_create = subparsers.add_parser("create", parents=[parser_debug], description=help_create, help=help_create)
parser_create.add_argument('-n', '--num_strings', help='Number of strings to generate.', type=int, default=100)
parser_create.add_argument('-l', '--string_length', help='Length of strings to generate (min, max).', nargs=2, type=int, default=[2, 70])
parser_create.add_argument('-c', '--valid_characters', help='Characters that strings may contain.', type=set, default=list(set(string.printable) - set(string.whitespace)))
parser_create.add_argument('-o', '--output_path', help='Path to save component dataset to.', type=utils._valid_output_path, default='generated_dataset.txt')

## Subparser for COMBINE functionality
help_combine = "Combine multiple components datasets into one ready for training."
parser_combine = subparsers.add_parser("combine", parents=[parser_debug], description=help_combine, help=help_combine)
parser_combine.add_argument('-f', '--files_nonsmiles', required=True, nargs='+', help='Path to non-SMILES datasets.', type=utils._file_exists)
parser_combine.add_argument('-s', '--files_smiles', required=True, nargs='+', help='Path to SMILES datasets.', type=utils._file_exists)
parser_combine.add_argument('-o', '--output_path', help='Path to save combined dataset to.', type=utils._valid_output_path, default='combined.csv')

## Subparser for EVALUATE functionality
help_evaluate = "Generate summary information for given a dataset."
parser_evaluate = subparsers.add_parser("evaluate", parents=[parser_debug], description=help_evaluate, help=help_evaluate)
parser_evaluate.add_argument('-d', '--datasets', required=True, nargs='+', help='Path to dataset(s) for evaluation.', type=utils._file_exists)

## Subparser for SPLIT functionality
help_split = "Perform train/test split for a given dataset."
parser_split = subparsers.add_parser("split", parents=[parser_debug], description=help_split, help=help_split)
parser_split.add_argument('-d', '--dataset', required=True, help='Path to dataset for train/test split.', type=utils._file_exists)
parser_split.add_argument('-t', '--test_size', help='Proportion of dataset to assign to test.', default=0.3, type=float)
parser_split.add_argument('-s', '--seed', help='Seed for data split reproducibility.', type=int, default=None)
parser_split.add_argument('-p', '--prefix', help='Prefix of output files.', type=str, default=None)


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
    if args.mode == "create":
        generate_random_strings(
            output_file = args.output_path,
            n_strings = args.num_strings,
            string_length = args.string_length,
            valid_characters = args.valid_characters
            )
    elif args.mode == "combine":
        combine_datasets(
            files_nonsmiles = args.files_nonsmiles,
            files_smiles = args.files_smiles,
            output_file = args.output_path
            )
    elif args.mode == "evaluate":
        evaluate_datasets(
            datasets = args.datasets,
            logger = logger
            )
    elif args.mode == "split":
        split_dataset(
            dataset = args.dataset,
            logger = logger,
            test_size = args.test_size,
            seed = args.seed,
            prefix = args.prefix
            )
    else:
        logger.critical(f"Unknown mode '{args.mode}'")

if __name__ == "__main__":
    main()
