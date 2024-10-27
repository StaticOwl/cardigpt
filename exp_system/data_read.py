"""
File: data_read.py
Author: Koustav Mallick
Date: 26/10/2024

Description: [Add a brief description of the file here]
"""
import defaultdict

def parse_conditions(record):
    confident_conditions = defaultdict(list)

    for column in record.index:
        if '_label' in column and record[column] == 1:
            score_column = column.replace('_label', '_score')
            condition_code = column.split('_')[0]
            score = record[score_column] if score_column in record else 0.0
            confident_conditions[condition_code] = score

    return confident_conditions

def expert_system_solution(confident_conditions):
    explanation = []
    solution_confidence = 0.0
    
    # Example Rule 1: If certain conditions are present, suggest a potential cardiovascular issue
    if '111975006' in confident_conditions and '164890007' in confident_conditions:
        explanation.append("Indication of potential ischemic heart disease.")
        solution_confidence += (confident_conditions['111975006'] + confident_conditions['164890007']) / 2

    # Example Rule 2: If certain arrhythmia-related codes are detected, suggest monitoring for arrhythmia
    if '426627000' in confident_conditions and '427172004' in confident_conditions:
        explanation.append("Signs of arrhythmia. Recommend further cardiac monitoring.")
        solution_confidence += (confident_conditions['426627000'] + confident_conditions['427172004']) / 2

    # Example Rule 3: If more than 3 high-confidence conditions, suggest a thorough cardiovascular examination
    if len(confident_conditions) > 3:
        explanation.append("Multiple conditions detected. Suggest a comprehensive cardiovascular examination.")
        solution_confidence += sum(confident_conditions.values()) / len(confident_conditions)

    # Normalize the solution confidence score
    solution_confidence = min(solution_confidence, 1.0)
    
    return {
        "explanation": " | ".join(explanation) if explanation else "No significant condition detected.",
        "solution_confidence": solution_confidence
    }

# Define an enhanced expert system function with more refined rules and thresholds
def enhanced_expert_system_solution(confident_conditions):
    explanation = []
    solution_confidence = 0.0
    
    # Rule 1: Threshold-based rule for ischemic heart disease indication
    if '111975006' in confident_conditions and confident_conditions['111975006'] > 0.8:
        explanation.append("High confidence for ischemic heart disease.")
        solution_confidence += confident_conditions['111975006']

    # Rule 2: Combination of arrhythmia and conduction disorder with threshold
    if '426627000' in confident_conditions and '427172004' in confident_conditions:
        if confident_conditions['426627000'] > 0.6 and confident_conditions['427172004'] > 0.6:
            explanation.append("Detected arrhythmia with conduction disorder, indicating possible rhythm instability.")
            solution_confidence += (confident_conditions['426627000'] + confident_conditions['427172004']) / 2

    # Rule 3: Cardiovascular examination recommendation for multiple high-confidence conditions
    high_confidence_conditions = [code for code, score in confident_conditions.items() if score > 0.7]
    if len(high_confidence_conditions) > 3:
        explanation.append("Multiple high-confidence cardiovascular indicators detected. Suggest a comprehensive cardiovascular examination.")
        solution_confidence += sum(confident_conditions[code] for code in high_confidence_conditions) / len(high_confidence_conditions)
    
    # Rule 4: Additional explanatory suggestions for tiered conditions based on severity
    if len(explanation) > 2:
        explanation.append("Given the combination of detected conditions, a specialist consultation may be beneficial.")
    
    # Normalize the solution confidence score to a maximum of 1.0
    solution_confidence = min(solution_confidence / len(explanation) if explanation else 0, 1.0)
    
    return {
        "explanation": " | ".join(explanation) if explanation else "No significant condition detected.",
        "solution_confidence": solution_confidence
    }

# Test the enhanced expert system function on the sample conditions
enhanced_solution_sample = enhanced_expert_system_solution(confident_conditions_sample)

enhanced_solution_sample


def main():
    pass

if __name__ == '__main__':
    main()
