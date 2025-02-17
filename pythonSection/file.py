from openai import OpenAI
import os, json
import requests
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(
    base_url = "https://api.together.xyz/v1",
    api_key = os.environ['TOGETHER_API_KEY'],
)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo"

def get_chatbot_response(client,model_name,messages,temperature=0):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2000,
    ).choices[0].message.content
    
    return response

messages = [{'role':'user','content':"What's the capital of Italy?"}]
response = get_chatbot_response(client,model_name,messages)
print(response)


embedding_client = OpenAI(
        
    )


def get_embedding(embedding_client,model_name,text_input):
    output = embedding_client.embeddings.create(input = text_input,model=model_name)
    
    embedings = []
    for embedding_object in output.data:
        embedings.append(embedding_object.embedding)

    return embedings



user_prompt = """What's new in iphone 16?"""
user_prompt_embeddings = get_embedding(embedding_client,model_name,user_prompt)




def get_embedding(text_input):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Correct data format for Hugging Face API
    data = {
        "inputs": text_input,  # String input, not a list
        "parameters": {}  # Some models require this field
    }

    response = requests.post(EMBEDDING_URL, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()  # Extract embeddings from JSON response
    else:
        print("Error:", response.text)  # Print error message if request fails
        return None

# Example usage:
user_prompt = "What's new in iPhone 16?"
user_prompt_embeddings = get_embedding(user_prompt)
print(user_prompt_embeddings)  # Check output


API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": "Bearer "}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
	"source_sentence": "That is a happy person",
	"sentences": [
		"That is a happy dog",
		"That is a very happy person",
		"Today is a sunny day"
	]
},
})


output