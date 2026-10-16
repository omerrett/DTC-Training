"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)

def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.
    NaN values are ignored, normalised to 0.
    Negative values rounded to 0.
    """
    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid ='ignore', divide = 'ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0

    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')

    return normalised

class Patient:
    def __init__(self,id,data):
        self.id = id
        self.data = data

    def data_mean(self):
        """Calculate the overall mean of a patients inflammation data."""
        return np.mean(self.data)

    def data_max(self):
        """Calculate the overall max of a patients inflammation data."""
        return np.max(self.data)

    def data_min(self):
        """Calculate the overall min of a patients inflammation data."""
        return np.min(self.data)

class Trial:
    def __init__ (self, filename, id):
        self.id = id
        self.data = self.load_csv(filename)

    @staticmethod
    def load_csv(filename):
        """Load a Numpy array from a CSV

        :param filename: Filename of CSV to load
        """
        return np.loadtxt(fname=filename, delimiter=',')

    def daily_mean(self):
        """Calculate the daily mean of a 2d inflammation data array."""
        return np.mean(self.data, axis=0)


    def daily_max(self):
        """Calculate the daily max of a 2d inflammation data array."""
        return np.max(self.data, axis=0)


    def daily_min(self):
        """Calculate the daily min of a 2d inflammation data array."""
        return np.min(self.data, axis=0)

    def patient_normalise(self):
        """
        Normalise patient data from a 2D inflammation data array.
        NaN values are ignored, normalised to 0.
        Negative values rounded to 0.
        """
        if np.any(self.data < 0):
            raise ValueError('Inflammation values should not be negative')
        if not isinstance(self.data, np.ndarray):
            raise TypeError('data input should be ndarray')
        if len(self.data.shape) != 2:
            raise ValueError('inflammation array should be 2-dimensional')
        max_data = np.nanmax(self.data, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            normalised = self.data / max_data[:, np.newaxis]
        normalised[np.isnan(normalised)] = 0
        normalised[normalised < 0] = 0
        return normalised

    def get_patient(self, row):
        """ Gets a patient object by data row. The id of the object is the row number."""
        return Patient(row, self.data[row,:])

filename = "data/inflammation-01.csv"
trial_group_01 = Trial(filename, "Group01")
patient_0 = trial_group_01.get_patient(0) # Create a Patient with id 0