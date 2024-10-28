"""
File: knowledge.py
Author: Koustav Mallick
Date: 27/10/2024

Description: [Add a brief description of the file here]
"""

import yaml

class Conditions:
    def __init__(self, name, description, confidence_threshold, recommendations, snowmed):
        self.name = name
        self.description = description
        self.confidence_threshold = confidence_threshold
        self.recommendations = recommendations
        self.snowmed = snowmed
        self.parents = []

    def load_parents(self):
        #TODO: Use snowmed data to load parent conditions.
        pass
    def get_explanation(self):
        return f"{self.name}: {self.description}. Treatments include {', '.join(self.recommendations)}."


def load_knowledge():
    with open("knowledge.yaml", "r") as f:
        data = yaml.safe_load(f)
    
    conditions = {}
    for cond in data["conditions"]:
        name = cond["name"]
        description = cond["description"]
        confidence_threshold = cond["confidence_threshold"]
        recommendations = cond["recommendations"]
        snowmed = cond["snowmed"]
        conditions[snowmed] = Conditions(name, description, confidence_threshold, recommendations, snowmed)
    
    return conditions

def main():
    pass

if __name__ == '__main__':
    main()
