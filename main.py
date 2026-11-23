"""
STEP 1 of the project: load data, clean it, and make charts.

Run:  python main.py
"""

import os

from src.datahandler import DataHandler
from src.datavisualizer import DataVisualizer

Data_Set = "data/Synthetic_Financial_datasets_log.csv"
if not os.path.exists(Data_Set):          # fall back to the fake test data
    Data_Set = "data/sample_data.csv"

handler = DataHandler(Data_Set)

data = handler.load_data()
handler.inspect_data(data)                # prints rows, columns, nulls, duplicates
data = handler.clean_data(data)           # store the cleaned data back in data

obj = DataVisualizer(output_dir="reports")
obj.visualize_all(data)
