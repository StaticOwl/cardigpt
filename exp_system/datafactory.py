"""
File: datafactory.py
Author: Koustav Mallick
Date: 26/10/2024

Description:
This module contains the DataFactory class which is responsible for loading and processing the data.
"""
from collections import defaultdict
import pandas as pd
from math import log, tanh
import json
from engine import Engine

class DataFactory:
    """
    The DataFactory class is responsible for loading and processing the data.

    Attributes:
        record_dict (dict): A dictionary containing the records.
        knowledge (dict): A dictionary containing the knowledge base data.
        engine (Engine): An instance of the Engine class.
    """

    def __init__(self):
        """
        Initializes the DataFactory class.
        """
        self.record_dict = None
        self.knowledge = None
        self.engine = Engine(self)

    def read_predictions(self, path, is_csv=True):
        """
        Reads the predictions from a file and returns a list of predictions.

        Args:
            path (str): The path to the predictions file.
            is_csv (bool): A flag indicating whether the file is a csv file or not. Defaults to True.

        Returns:
            list: A list of predictions.
        """
        if is_csv:
            records = pd.read_csv(path, header=0)
    
            record_dict = defaultdict(lambda:defaultdict(dict))

            for _, row in records.iterrows():
                filename = row['filekey']
                for key, value in row.items():
                    if key != 'filekey':
                        record_key = key.split('_')
                        record_dict[filename][record_key[0]][record_key[1]] = value

            self.record_dict = dict(record_dict)
        else:
            with open(path, 'r') as f:
                self.record_dict = json.load(f)

    def __str__(self): return json.dumps(self.record_dict, indent=4)

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
        
        self.knowledge = {self.process_key(key):value for key, value in self.knowledge.items()}

    def strength_builder(self):
        """
        Builds the strength of the predictions based on the knowledge base data.
        """
        if self.knowledge is None:
            print("Knowledge base not loaded")
            return
        else:
            print("Knowledge base loaded")
        print("Building strength")
        for itemKey, item in self.record_dict.items():
            for key, value in item.items():
                if key != 'filekey':
                    try:
                        priority = self.knowledge.get(int(key)).get("impact")
                        confidence = self.engine.fuzz_dfuzz(term=priority)
                        label = value.get('label')
                        score = value.get('score')
                        score = max(min(score, 1 - 1e-15), 1e-15)
                        score = score ** 0.4
                        log_odds = log(score / (1 - score))
                        strength = (tanh(log_odds) if label == 1 else tanh(-log_odds))
                        self.record_dict[itemKey][key]["condition"] = self.engine.fuzz_dfuzz(label=label, strength=strength)
                    except KeyError as e:
                        print(f"{key} not found in knowledge base")
                        print(f"{key}: {value}")
    
    def process_key(self, key):
        try:
            return int(key)
        except ValueError:
            return key


def main():
    import os,json
    df = DataFactory()
    path = os.path.join(os.getcwd(), "exp_system") + "/"
    df.read_predictions(path+"results_E03065.json", is_csv=False)
    df.load_kb(path+"knowledge.json")
    df.strength_builder()
    # for key, _ in df.knowledge.items():
    #     df.load_snomed(key)
    # print(df.knowledge)
    chatbot_input="exp_system/chatbot_input"
    os.makedirs(chatbot_input,exist_ok=True)
    with open(f"{chatbot_input}/results_E03065_output.json", "w") as outfile:
        json.dump(df.record_dict, outfile, indent=4)
    pass

if __name__ == "__main__":
    main()

