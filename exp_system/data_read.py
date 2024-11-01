"""
File: data_read.py
Author: Koustav Mallick
Date: 26/10/2024

Description: [Add a brief description of the file here]
"""
from collections import defaultdict
import pandas as pd
from math import log, tanh
import json

class DataFactory:
    def __init__(self):
        self.path = None
        self.record_dict = None
        self.knowledge = None
    
    def read_predictions(self, path):
        """
        Reads the predictions from a file and returns a list of predictions.

        Args:
            path (str): The path to the predictions file.

        Returns:
            list: A list of predictions.
        """
        records = pd.read_csv(path, header=0)
        record_dict = defaultdict(lambda:defaultdict(dict))

        for _, row in records.iterrows():
            filename = row['filename']
            for key, value in row.items():
                if key != 'filename':
                    record_key = key.split('_')
                    record_dict[filename][record_key[0]][record_key[1]] = value

        self.record_dict = dict(record_dict)

    def load_kb(self, path):
        """
        Loads the knowledge base from a file and returns a dictionary.

        Args:
            path (str): The path to the knowledge base file.

        Returns:
            dict: A dictionary of knowledge base data.
        """
        with open(path, 'r') as f:
            self.knowledge = json.load(f)

    def __str__(self): return json.dumps(self.record_dict, indent=4)

    def strength_builder(self):
        
        for itemKey, item in self.record_dict.items():
            for key, value in item.items():
                try:
                    confidence = self.knowledge[key]["confidence_threshold"]
                    label = value.get('label')
                    score = value.get('score')
                    score = max(min(score, 1 - 1e-15), 1e-15)
                    log_odds = log(score / (1 - score))
                    self.record_dict[itemKey][key]["strength"] = confidence * (tanh(log_odds) if label == 1 else tanh(-log_odds))
                except KeyError:
                    print(f"{key}: {value}")

    def mismatched_records(self):
        
        mismatched_records = {
            filename: {key: value for key, value in content.items() 
               if (value.get('label') == 1 and value.get('score') == 0) or
                  (value.get('label') == 0 and value.get('score') == 1)}
            for filename, content in self.record_dict.items()
        }

        return {k: v for k, v in mismatched_records.items() if v}


def main():
    import os
    df = DataFactory()
    path = os.path.join(os.getcwd(), "exp_system") + "/"
    df.read_predictions(path+"predictions.csv")
    df.load_kb(path+"knowledge.json")
    df.strength_builder()
    print(df.mismatched_records())
    # print(df)
    pass

if __name__ == "__main__":
    main()
