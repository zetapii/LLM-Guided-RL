import ollama

# Initialize the Ollama client
client = ollama.Client()

# Define the model and your prompt
model = 'llama3.2'  # Use 'llama3.2:1b' for the 1B model
prompt = ''

stream = client.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], stream=True)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

