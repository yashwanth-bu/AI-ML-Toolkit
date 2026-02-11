import ollama
import os
import json

# ==============================
# Configuration
# ==============================

SANDBOX_DIR = "sandbox"
ALLOWED_EXTENSIONS = [".txt", ".md", ".py", ".json", ".js"]
BANNED_WORDS = ["rm -rf", "ignore previous", "hack"]

SYSTEM_PROMPT = """
You are a helpful assistant.

If the user asks to create a file, respond ONLY in valid JSON format like this:

{
  "tool": "create_file",
  "path": "filename.ext",
  "content": "file content here"
}

Rules:
- Respond with ONLY valid JSON when calling a tool.
- Do NOT include explanations.
- Do NOT wrap JSON in markdown.
- Do NOT include triple backticks.
- Only call the tool if explicitly asked to create a file.
- Otherwise respond normally with text.
"""

# ==============================
# Safety Utilities
# ==============================

def is_safe(text):
    return not any(word in text.lower() for word in BANNED_WORDS)


def clean_code_blocks(content):
    """
    Removes markdown code fences like ```javascript ... ```
    """
    content = content.strip()

    # Remove triple backticks (any variation)
    content = content.replace("```javascript", "")
    content = content.replace("```js", "")
    content = content.replace("```json", "")
    content = content.replace("```python", "")
    content = content.replace("```", "")

    return content.strip()


def create_file(path, content):
    try:
        # Prevent directory traversal
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1]

        if ext not in ALLOWED_EXTENSIONS:
            return "Error: file type not allowed"

        # Ensure sandbox exists
        os.makedirs(SANDBOX_DIR, exist_ok=True)

        # Clean markdown fences
        content = clean_code_blocks(content)

        full_path = os.path.join(SANDBOX_DIR, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File created at {full_path}"

    except Exception as e:
        return f"Error creating file: {str(e)}"


# ==============================
# Ollama Adapter
# ==============================

class OllamaAdapter:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_response(self, prompt, system=None):
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            system=system,
            stream=False
        )
        return response["response"]


# ==============================
# Conversation Memory
# ==============================

conversation = []

def build_prompt(user_input):
    conversation.append(f"User: {user_input}")
    return "\n".join(conversation[-6:])


# ==============================
# Main Agent Loop
# ==============================

def main():
    adapter = OllamaAdapter("mistral")

    print("Local Agent Started (type 'exit' to quit)\n")

    while True:
        user_input = input("User: ")

        if user_input.lower() == "exit":
            break

        if not is_safe(user_input):
            print("Input contains banned words. Please try again.")
            continue

        prompt = build_prompt(user_input)

        try:
            response = adapter.generate_response(prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            print(f"Model error: {str(e)}")
            continue

        # ==============================
        # Try Parsing JSON Tool Call
        # ==============================

        try:
            data = json.loads(response)

            if data.get("tool") == "create_file":
                filename = data.get("path")
                content = data.get("content")

                if not filename or content is None:
                    print("Malformed tool response.")
                    continue

                result = create_file(filename, content)
                conversation.append(f"Assistant: {result}")
                print(result)
                continue

        except json.JSONDecodeError:
            pass  # Not a tool call → normal response

        # ==============================
        # Normal Response
        # ==============================

        conversation.append(f"Assistant: {response}")
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
