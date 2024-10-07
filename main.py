#by @orangebird3
#10/05/2024 (mm/dd/yyyy)
#small data synthesizer project i was working on

import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

#base abstract generator class
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, n):
        pass

#numerical data generator
class NumericalGenerator(BaseGenerator):
    def __init__(self, mean=0, std_dev=1, min_value=None, max_value=None):
        self.mean = mean
        self.std_dev = std_dev
        self.min_value = min_value
        self.max_value = max_value

    def generate(self, n):
        data = np.random.normal(self.mean, self.std_dev, n)
        if self.min_value is not None:
            data = np.maximum(data, self.min_value)
        if self.max_value is not None:
            data = np.minimum(data, self.max_value)
        return data

#categorical data generator
class CategoricalGenerator(BaseGenerator):
    def __init__(self, categories, probabilities=None):
        self.categories = categories
        self.probabilities = probabilities

    def generate(self, n):
        return random.choices(self.categories, weights=self.probabilities, k=n)

#time-series data generator
class TimeSeriesGenerator(BaseGenerator):
    def __init__(self, start_date, end_date, frequency='1D'):
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

    def generate(self, n):
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.frequency)
        data_length = min(n, len(date_range))
        return np.random.choice(date_range, size=data_length, replace=False)

#DataSynthesizer to manage and synthesize datasets
class DataSynthesizer:
    def __init__(self):
        self.generators = {}

    def add_generator(self, name, generator):
        if not isinstance(generator, BaseGenerator):
            raise ValueError(f"Generator for {name} must be an instance of BaseGenerator")
        self.generators[name] = generator

    def generate_data(self, n):
        synthesized_data = {}
        for name, generator in self.generators.items():
            synthesized_data[name] = generator.generate(n)
        return pd.DataFrame(synthesized_data)

#testing
if __name__ == "__main__":
    synthesizer = DataSynthesizer()

    #generators for numerical, categorical, and time-series data
    synthesizer.add_generator('Age', NumericalGenerator(mean=30, std_dev=10, min_value=18, max_value=65))
    synthesizer.add_generator('Income', NumericalGenerator(mean=50000, std_dev=15000, min_value=20000, max_value=100000))
    synthesizer.add_generator('Gender', CategoricalGenerator(categories=['Male', 'Female'], probabilities=[0.5, 0.5]))
    synthesizer.add_generator('SignUpDate', TimeSeriesGenerator(start_date='2022-01-01', end_date='2024-01-01', frequency='1D'))

    #100 samples
    synthetic_data = synthesizer.generate_data(100)
    print(synthetic_data)
