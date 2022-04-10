#!/usr/bin/env python3

"""General utility functions"""

from pathlib import Path
import logging
import sys


def _file_exists(path: str, logger:logging.Logger=logging.getLogger("utils")):
    """Check whether a provided file path exists, raise error and fail fast if not.
    
    Args:
        path (str): File path for verification
        logger (logging.Logger, optional): Logger object
    
    Returns:
        pathlib.Path: pathlib.Path representation of file path, if valid. Otherwise sys.exit is invoked.
    """

    path = Path(path)

    if not path.exists():
        logger.critical(f"Provided file '{path}' does not exist.")
        sys.exit(-1)
    else:
        return path


def _dir_exists(dir_path:str, logger:logging.Logger=logging.getLogger("utils")):
    """Check whether a provided path is a valid directory.
    
    Args:
        dir_path (str): Directory path for verification
        logger (logging.Logger, optional): Logger object
    
    Returns:
        pathlib.Path: pathlib.Path representation of directory, if valid. Otherwise sys.exit is invoked.
    """

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        logger.critical(f"Provided path '{dir_path}' is not a directory.")
        sys.exit(-1)
    else:
        return dir_path


def _valid_output_path(path:str, logger:logging.Logger=logging.getLogger("utils")):
    """Check whether a given output path is valid (its directory exists), prompts if a file already exists.
    
    Args:
        path (str): Output file path for verification
        logger (logging.Logger, optional): Logger object
    
    Returns:
        pathlib.Path: pathlib.Path representation of output file path, if valid. Otherwise sys.exit is invoked.
    """
    path = Path(path)
    if path.exists():
        logger.warning(f"Output location already exists: {path}")
        overwrite_check = input('Overwrite existing path? [y/N] ')
        if overwrite_check.lower() != 'y':
            logger.critical(f"Exiting due to declined overwrite.")
            sys.exit(-1)
        else:
            return path
    elif not path.absolute().parents[0].is_dir():
        logger.critical(f"Invalid output location provided: {path.absolute()}")        
        sys.exit(-1)
    else:
        return path


def file_to_list(path:str):
    """Read a given file to create a list containing its contents
    
    Args:
        file (str): Path to file
    
    Returns:
        list: List representation of file
    """
    path = _file_exists(path)
    with open(path, 'r') as r:
        lines = [x.rstrip() for x in r.readlines()]
    return(lines)


def flatten_list_of_lists(list_of_lists:list):
    """Flatten a list of lists into a simple list
    
    Args:
        list_of_lists (list): Two dimension list
    
    Returns:
        list: One dimension list
    """
    return [item for sublist in list_of_lists for item in sublist]
