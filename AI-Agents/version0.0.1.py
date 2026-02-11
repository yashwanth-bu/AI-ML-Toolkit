import ollama
from datetime import datetime

# Initialize conversation history
conversation = []

# Function to build the prompt with conversation history
def build_prompt(user_input):
    conversation.append(f"User: {user_input}")
    return "\n".join(conversation[-6:])


# Define banned words for safety
BANNED_WORDS = ["rm -rf", "ignore previous", "hack"]

# Function to check if the input is safe
def is_safe(text):
    return not any(word in text.lower() for word in BANNED_WORDS)


# Define system prompt with tool call instructions
SYSTEM_PROMPT = """
You are an AI assistant.

If the user asks for the current time, respond ONLY with:
TOOL_CALL:get_time

Do not add any extra text when calling a tool.
Otherwise respond normally.
"""

# Function to ask the model and get a response
def ask_model(question):
    response = ollama.generate(
        model="mistral",
        prompt=question,
        stream=False
    )
    return response["response"]


# Example tool function to get the current time
def get_time():
    return datetime.now().strftime("Current time is %H:%M") 


# Main loop to interact with the user
while True:
    user_input = input("Ask the model (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    if not is_safe(user_input):
        print("Input contains banned words. Please try again.")
        continue

    prompt = build_prompt(user_input)

    answer = ask_model(SYSTEM_PROMPT + "\n" + prompt).strip()
    conversation.append(f"Assistant: {answer}")
    # Check if the model is calling a tool
    if answer.splitlines()[0].strip() == "TOOL_CALL:get_time":
        result = get_time()
        print("Tool:", result)
    else:
        print("AI:", answer)
