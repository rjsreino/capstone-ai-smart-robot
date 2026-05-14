from ollama import chat

print("Starting test...")

response = chat(
    model='phi3',
    messages=[
        {'role': 'user', 'content': 'Hello, who are you?'}
    ]
)

print("Response received:")
print(response.message.content)