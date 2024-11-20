"""
File: engine.py
Author: Koustav Mallick
Date: 19/11/2024

Description: This module contains the Engine class which is responsible for cleaning and processing the data.
"""

from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
import wikipedia

class Engine:
    """
    The Engine class is responsible for cleaning and processing the data.
    """
    def __init__(self, data):
        """
        Initializes the Engine class.

        Args:
            data (dict): The data to be processed.
        """
        default_world.set_backend(filename="pym.sqlite3")
        try:
            PYM = get_ontology("http://PYM/").load()
        except:
            import_umls("umls-2024AA-metathesaurus-full.zip", terminologies=["SNOMEDCT_US"])
            default_world.save()
            PYM = get_ontology("http://PYM/").load()
        self.snowmed = PYM["SNOMEDCT_US"]
        self.data = data
        self.negligible_records = []

    def clean_data(self):
        """
        Removes records with a condition of 'negligible' from the data.
        """
        print("Cleaning data")
        for filename, content in list(self.data.record_dict.items()):
            negligible_for_file = {}
            filtered_content = {}
            for key, value in content.items():
                if key == 'filekey' or value.get('condition') != 'negligible':
                    filtered_content[key] = value
                else:
                    negligible_for_file[key] = value

            self.data.record_dict[filename] = filtered_content
            if negligible_for_file:
                self.negligible_records.append({filename: negligible_for_file})

    def load_knowledge(self):
        """
        Loads the knowledge base data for each record.
        """
        print("Loading knowledge base data")
        for filekey, data in self.data.record_dict.items():
            for key, _ in data.items():
                if key != 'filekey':
                    condition = self.data.record_dict[filekey][key].get('condition')
                    self.data.record_dict[filekey][key] = {
                        **self.data.record_dict[filekey][key],
                        "recommendation": self.data.knowledge.get(int(key)).get("recommendations").get(condition),
                        "knowledge": {
                            **self.load_snomed(key)
                        }
                    }

    def load_snomed(self, ct_code):
        """
        Loads the SNOMED data for a given CT code.

        Args:
            ct_code (str): The CT code.

        Returns:
            dict: The SNOMED data.
        """
        print(f"Loading snowmed knowledge for {ct_code}")
        ct_code = self.data.process_key(ct_code)
        concept = self.snowmed[ct_code]
        label = concept.label.first().split(",")[0].strip()
        synonyms = [str(label).strip() for label in concept.synonyms]
        relevant_ancestors = [str(label).split('#')[1].strip() for label in concept.ancestor_concepts()[:3]][1:]
        interpretation = [str(label).split('#')[1].strip() for label in concept.interprets]
        term_type = concept.term_type
        search_results = wikipedia.search(label)
        try:
            wiki_summary = wikipedia.summary(search_results[0])
        except:
            wiki_summary = None
        return {
            "name":self.data.knowledge[ct_code].get("name"),
            "description":self.data.knowledge[ct_code].get("description"),
            "label": label,
            "synonyms": synonyms,
            "relevant_ancestors": relevant_ancestors,
            "interpretation": interpretation,
            "term_type": term_type,
            "wiki_summary": wiki_summary
        }

    def mismatched_records(self):
        """
        Returns a dictionary of records where the label and score mismatch.
        """
        mismatched_records = {
            filename: {key: value for key, value in content.items() 
               if (value.get('label') == 1 and value.get('score') == 0) or
                  (value.get('label') == 0 and value.get('score') == 1)}
            for filename, content in self.data.record_dict.items()
        }

        return {k: v for k, v in mismatched_records.items() if v}
    
    def fuzz_dfuzz(self, label=None, strength=None, term=None, alpha=0.7):
        """
        Classifies the condition based on the label and strength.

        Args:
            label (int): The label of the prediction.
            strength (float): The strength of the prediction.

        Returns:
            str: The classified condition.
        """
        term_ranges = {
            "severe": {
                "label": 1,
                "score": (0.9, 1.0)
            },
            "high": {
                "label": 1,
                "score": (0.79, 0.9)
            },
            "medium": {
                "label": None,
                "score": (0.0, 0.79)
            },
            "low": {
                "label": 0,
                "score": (0.79, 0.9)
            },
            "negligible": {
                "label": 0,
                "score": (0.9, 1.0)
            }
        }

        if term:
            lower, upper = term_ranges.get(term, {}).get("score")
            return alpha * lower + (1 - alpha) * upper
        
        if strength is not None and label is not None:
            for condition, properties in term_ranges.items():
                label_match = properties["label"] is None or properties["label"] == label
                lower, upper = properties["score"]
                if label_match and lower <= strength <= upper:
                    return condition
                
        return None
    
    def centroid_defuzzifier(self, data):
        pass

    
def main():
    pass

if __name__ == '__main__':
    main()

