#!/usr/bin/env python3

"""Train and load BinaryStringClassifier models
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import json
import logging
import string

# Silence TF warnings
if True:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

from utils import _file_exists, _valid_output_path


class BinaryStringClassifier(object):
    """Core class for encapsulating Binary String Classifier models.
    
    Attributes:
        logger (logging.Logger, optional): Standard Python logger
        model (keras.Sequential): Keras LSTM model
    """
    
    def __init__(self, logger:logging.Logger=logging.getLogger('BinaryStringClassifier')):
        """Core class for training and loading Binary String Classifiers.
        
        Args:
            logger (logging.Logger, optional): Standard Python logger
        """
        self.model = None
        self.logger = logger

    def _encode_string_list(self, list_of_strings: list, max_input_size: Union[int, None]=None) -> np.array:
        """Encode a given list of strings for this model's tokeniser.
        
        Args:
            list_of_strings (list): List containing strings to be encoded
            max_input_size (Union[int, None], optional): Maximum input size of strings to tokeniser
        
        Returns:
            np.array: Numpy array with shape (len(list_of_strings), max_input_size)
        
        Raises:
            ValueError: If no tokenizer is found at self.tokenizer
        """
        if max_input_size is None:
            max_input_size = self.model.input_shape[1]
        if self.tokenizer is None:
            raise ValueError(f"Tokenizer model not loaded.")

        tokens = self.tokenizer.texts_to_sequences(list_of_strings)
        tokens_padded = pad_sequences(tokens, maxlen=max_input_size)
        return tokens_padded

    def predict(self, data: str):
        """Get SMILES probability for a given string.
        
        Args:
            data (str): String to provide SMILES probability for
        
        Raises:
            ValueError: If data is not a str
        """

        if isinstance(data, str):
            data = [data]
        else:
            raise ValueError(f"Expected a string, received {type(data)}")

        data_encoded = self._encode_string_list(data)
        prediction = self.model.predict(data_encoded)[0][1]
        return prediction

    def export_model(self, path: str) -> str:
        """Save a trained model & components to disk.
        
        Components include:
            - history.json
            - tokenizer.json
        
        Args:
            path (str): Path to write model directory to
        """
        path_model = Path(path)
        path_history = Path(path, 'history.json')
        path_tokenizer = Path(path, 'tokenizer.json')

        self.logger.info(f"Saving trained model to {path_model}")
        self.model.save(path_model)

        self.logger.info(f"Saving model history to '{path_history}'")
        json.dump(self.history.history, open(path_history, 'w'))

        self.logger.info(f"Saving model tokenizer to '{path_tokenizer}'")
        json_tokenizer = self.tokenizer.to_json()
        with open(path_tokenizer, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_tokenizer)) #, ensure_ascii=False))


class ModelLoader(BinaryStringClassifier):
    """Create an BinaryStringClassifier object by loading an existing model.
    
    Attributes:
        history (dict): Model training history
        model (keras.Sequential): Loaded model
        tokenizer (Tokenizer): Tokenizer for model
    """
    def __init__(self, path_model: str, path_tokenizer: str = 'tokenizer.json', *args, **kwargs):
        """Summary
        
        Args:
            path_model (str): Path to trained model for loading
            path_tokenizer (str, optional): Path to tokenizer for loading
            *args: Handles any other input
            **kwargs: Handles any other input
        """
        super(ModelLoader, self).__init__(*args, **kwargs)
        self.model = self._load_model(path_model)
        self.history = self._load_history(Path(path_model, 'history.json'))
        self.tokenizer = None if path_tokenizer is None else self._load_tokenizer(path_tokenizer)


    def _load_model(self, path_model: str) -> Sequential:
        """Load a keras model from the given location
        
        Args:
            path_model (str): Path to the root directory holding a keras model
        
        Returns:
            Sequential: Loaded model object
        """
        path_model = _file_exists(path_model)
        return load_model(path_model)

    def _load_tokenizer(self, path_tokenizer: str) -> Tokenizer:
        """Load tokenizer from the given location
        
        Args:
            path_tokenizer (str): Path to the tokenizer json
        
        Returns:
            Tokenizer: Loaded tokenizer
        """
        with open(path_tokenizer) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return tokenizer

    def _load_history(self, path_history: str) -> dict:
        """Load model history JSON
        
        Args:
            path_history (str): Path to model history JSON
        
        Returns:
            dict: Loaded model history
        """
        return json.load(path_history.open())


class ModelTrainer(BinaryStringClassifier):
    """Create an BinaryStringClassifier object by training a model on a given dataset.
    
    Attributes:
        dataset (pd.DataFrame): Combined dataset containing String and Category columns (where 1 indicates a SMILES string)
        history (dict): Model training history
        max_num_inputs (int): Maximum length of input vectors
        model (keras.Sequential): Loaded model
        tokenizer (Tokenizer): Tokenizer for model
    """
    def __init__(self, dataset: pd.DataFrame, *args, **kwargs):
        """Create an BinaryStringClassifier object by training a model on a given dataset.
        
        Args:
            dataset (pd.DataFrame): Combined dataset containing String and Category columns (where 1 indicates a SMILES string)
            *args: Handles any other input
            **kwargs: Handles any other input
        """
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.tokenizer = Tokenizer(char_level=True, filters=None)
        self.tokenizer.fit_on_texts(list(string.printable))
        self.max_input_size: int = self.dataset['String'].str.len().max() #train['String'].apply(lambda p: len(p.split())).max()
        self.max_num_inputs = 8192
        self.x_train, self.y_train = self._tokenize()
        self.model = self._build_model()
        self._train_model()

    def _tokenize(self) -> (np.array, np.array):
        """Tokenise training data
        
        Returns:
            np.array, np.array: Features of training data, Categories of training data
        
        """
        x_train = self.tokenizer.texts_to_sequences(self.dataset['String'])
        x_train = pad_sequences(x_train, maxlen=self.max_input_size)

        # Encode class vector as a binary class matrix
        y_train = to_categorical(self.dataset['Category'])

        return x_train, y_train

    def _build_model(self) -> Sequential:
        """Construct the LSTM model 
        
        Returns:
            Sequential: LSTM model
        """

        # TODO: Allow user to change this structure?
        # Source: https://towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc

        self.logger.info('Building model')
        model = Sequential()
        model.add(Embedding(input_dim=self.max_num_inputs, output_dim=256, input_length=self.max_input_size))
        model.add(SpatialDropout1D(0.3))
        model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )
        return model

    def _train_model(self):
        """Train the LSTM model
        """

        # TODO: Allow user to change hyperparameters?

        batch_size = 512
        epochs = 20

        self.logger.info('Training model')
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size
        )
