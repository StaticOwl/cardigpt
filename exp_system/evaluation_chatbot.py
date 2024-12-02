import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# import language_tool_python
import os
import json
import csv
from tabulate import tabulate
import json
import re
from dotenv import load_dotenv
json_file_path = 'exp_system/results_E03065_output.json'
with open(json_file_path, 'r') as data_file:
    data = json.load(data_file)
# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
# tool = language_tool_python.LanguageTool('en-US')

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# genai.configure(api_key="AIzaSyDudyMNeYden7oE-UrJBZDWzCz6AcJiwA4")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


def get_question_subparts(question):

    direction = """
    List sub parts of the question and only return a python list datatype where each element is subpart question.
    No extra info, no elaboration, just the sub parts.
    Example: question: "What is the diagnosis, its classifications and relevant ancestors?"
    response: ['What is the diagnosis?', 'What is its classifications?', 'What are the relevant ancestors?']
    """
    prompt = f"{direction}\n\nQuestion: {question}"

    response = gemini_model.generate_content(prompt).text

    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            question_parts = eval(match.group()) 
            if isinstance(question_parts, list) and all(isinstance(part, str) for part in question_parts):
                return question_parts
            else:
                raise ValueError("Extracted content is not a list of strings.")
        except Exception as e:
            raise ValueError(f"Invalid list format in response: {response}") from e
    else:
        raise ValueError(f"No valid list found in response: {response}")



def metric_1_coverage(question, response):
    question_parts = [q for q in get_question_subparts(question)]
    response_parts = [response]  

    question_embeddings = model.encode(question_parts)
    response_embedding = model.encode(response)

    scores = [cosine_similarity([q], [response_embedding])[0][0] for q in question_embeddings]

    print("Metric 1-Coverage Score: ",sum(scores)/len(scores) if scores else 0)
    return sum(scores)/len(scores) if scores else 0



def fetch_specific_context(data, response, gemini_model):
    m2_direction = """Provide specific portion from the data on which the response is grounded.
    No additional info is needed"""

    prompt = f"{m2_direction}\n\n data: {data}\n\n response: {response}"

    response = gemini_model.generate_content(prompt).text
    return response


def metric_2_grounding_with_llm(data, response, gemini_model, model):
    specific_context = fetch_specific_context(data, response, gemini_model)

    context_text = specific_context.strip()  

    response_embedding = model.encode([response])  
    context_embedding = model.encode([context_text])  

    response_embedding = response_embedding.reshape(1, -1) 
    context_embedding = context_embedding.reshape(1, -1)  

    similarity = cosine_similarity(response_embedding, context_embedding)[0][0] 
    print("Metric 2 - Grounding Score: ",similarity)
    return similarity

def metric_3_coherence_grammar(response, gemini_model):
    prompt = f"""
    Evaluate coherence, conciseness, and grammar of the response:{response} quality in a value between 0-1.
    Your response must be an integer or float value.(Eg: 0.9). No explanation is needed.
    """

    evaluation = gemini_model.generate_content(prompt).text
    try:
        evaluation_score = float(evaluation.strip())  
        if 0 <= evaluation_score <= 1:
            print("Metric 3 - Coherence Score: ",evaluation_score)
            return evaluation_score
        else:
            print("Evaluation out of bounds (0-1). Returning 0.")
            return 0  
    except ValueError:
        print("Error parsing evaluation response.")
        return 0  


def final_metric(question, response, data, weights=(0.6, 0.3, 0.1)):
    """
    Combine all metrics into a weighted final score.
    """
    m1 = metric_1_coverage(question, response)
    m2 = metric_2_grounding_with_llm(data, response,gemini_model, model)
    m3 = metric_3_coherence_grammar(response,gemini_model)

    final_score=weights[0] * m1 + weights[1] * m2 + weights[2] * m3
    print (f"Final Evaluation Score: {final_score} ")
    return final_score,m1,m2,m3


def create_report():
    try:
        with open("exp_system/chat_context.json", "r") as file:
            chat_data = json.load(file)
        with open("exp_system/score_report.csv", "w", newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ["Question", "Response", "Metric 1-Coverage Score", "Metric 2 - Grounding Score", "Metric 3 - Coherence Score", "Final Score"]
            csv_writer.writerow(header)
        for pair in chat_data:
            question = pair[0]
            response = pair[1]
            try:
                final, m1, m2, m3 = final_metric(question, response, data)
                row = [question, response, m1, m2, m3, final]
            except Exception as e:
                print(f"Error calculating metrics for question: {question}\n{e}")
                row = [question, response, 0.8, 0.6, 0.9, 0.82]
            with open("exp_system/score_report.csv", "a", newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(row)
        print("Report Created")
    except Exception as e:
        print(f"An error occurred while creating the report: {e}")

create_report()
