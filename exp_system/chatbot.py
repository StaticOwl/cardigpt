import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
from datafactory import main

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

create_input_to_chatbot=main()
# Load data.json
json_file_path = 'exp_system/chatbot_input/results_E03065_output.json'
with open(json_file_path, 'r') as data_file:
    data = json.load(data_file)
 
# Summary question and global directions
summary_question = """
An overall summary of the JSON FILE describing details of relevant medical condition(or diagnosis), their meaning, their severity and overall recommendations.
"""
global_directions = """
You are given a JSON FILE with following info.
"E03065" is the name of patient's file.
"39732003" is the code for a medical condition(or diagnosis)
"label" is binary indicating whether the condition applies to the patient or not.
"score" IGNORE THIS. DO NOT FACTOR THIS INTO YOUR ANALYSIS.
"condition" indicates severity of the diagnosis.
"recommendation" tell what is the prescription for the given condition.
"knowledge" reflects more data about that condition like its name, detailed description, synonyms of the condition, its relevant medical ancestors,
its interpretation, term type and a brief summary of the condition based on wikipedia.
 
YOUR RESPONSE MUST HAVE THE FOLLOWING:
1. Specific and complete Response to only what the user has asked.
2. A tone that indicates that you are the doctor and you are communicating your findings to another doctor or user.
 
YOUR RESPONSE MUST NOT CONTAIN THE FOLLOWING:
1. Phrases like "Based on the provided data" which reveal that you have been given a specific context and your response is built from that context.
2. Probabilities or score. Your analysis must rely on "label": 0/1 and "condition": "low"/"medium"/"high".
3. Meta data or any note of caution like "Important Notes:  This analysis is based solely on the model's output.  The model's confidence scores are not a definitive diagnosis.."
4. Anything outside of what is asked. Including any suggestion of precautionary note, any advice etc.
5. Any mention, howsoever subtle, of the file, filename or the context provided. The user should not be given any idea that your responses are coming from a file or context.
"""
 
# List to maintain chat context (up to the last 10 exchanges)
chat_context = []
 
# Function to query the chatbot with the combined context
def query_chatbot(question):
    # Create a context string from the last 10 exchanges
    context = "\n".join([f"User: {q}\nChatbot: {r}" for q, r in chat_context])
    prompt = f"{global_directions}:JSON FILE:{json.dumps(data, indent=2)}\n\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text
 
# Function to process user updates to the knowledge base
def update_kb(update_request):
    global data
    try:
        if "change" in update_request and "to" in update_request:
            parts = update_request.split("change", 1)[1].split("to", 1)
            path = parts[0].strip()
            value = parts[1].strip()
 
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
 
            # Traverse and update the JSON
            keys = path.split('.')
            ref = data
            for key in keys[:-1]:  
                ref = ref[key]
            ref[keys[-1]] = value  
 
            
            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
 
            return f"The KB has been updated: {path} is now {value}."
        else:
            return "Invalid update format. Use 'Update KB: change <key> to <value>'."
    except Exception as e:
        return f"Error processing update: {str(e)}"
 
 
 
is_first_time = True
 
while True:
    if is_first_time:
        summary_response = query_chatbot(summary_question)
        print("Chatbot Summary Response:", summary_response)
        is_first_time = False
 
    question = input("Ask a question or type 'exit' to quit: ")
    if question.lower() == 'exit':
        break
 
 
    if question.lower().startswith("update kb:"):
        update_request = question[len("update kb:"):].strip()
        update_feedback = update_kb(update_request)
        print("Chatbot Response:", update_feedback)
        continue
 
 
    response = query_chatbot(question)
    print("Chatbot Response:", response)
 
 
    chat_context.append((question, response))
 
 
    if len(chat_context) > 10:
        chat_context.pop(0)