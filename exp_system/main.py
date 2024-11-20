"""
File: main.py
Author: Koustav Mallick
Date: 19/11/2024

Description: [Add a brief description of the file here]
"""

from datafactory import DataFactory
import os
import json

def main():
    path = os.path.join(os.getcwd(), "exp_system") + "/"
    df = DataFactory()
    df.load_kb(path+"knowledge.json")
    df.read_predictions(path+"results_E03065.json", is_csv=False)
    df.strength_builder()
    df.engine.clean_data()
    df.engine.load_knowledge()

    print(df)
    with open(path+"results_E03065_output.json", 'w') as f:
        json.dump(df.record_dict, f, indent=4)
    pass

if __name__ == '__main__':
    main()
