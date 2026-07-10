import os
import re
from openai import OpenAI
import requests
import json 

def read_file(filepath):
    """Reads the current state of the file."""
    with open(filepath, 'r') as f:
        return f.read()

def write_file(filepath, content):
    """Overwrites the file with the new code."""
    with open(filepath, 'w') as f:
        f.write(content)

def edit_code(filepath, instruction):
    print(f"Reading {filepath}...")
    current_code = read_file(filepath)
    
    # 2. The System Prompt 
    # This is critical. We must force the LLM to output ONLY raw code
    # so we don't accidentally write "Here is your code:" into your files.

    system_prompt = f"""
    You are an expert AI coding agent. 
    You will be provided with the current code of a file and an instruction on how to modify it.
    You must output ONLY the entirely rewritten code.
    You must include clear comments inside the code explaining the modifications you made.
    Do not include any conversational text, explanations, or markdown formatting blocks.
    Just output the raw, modified code so it can be directly written to the file.
    Here is the code you need:{current_code}
    """

    
    print("Thinking...")
    # 3. Call OpenRouter
    

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "poolside/laguna-xs-2.1:free",
            "messages": [
                {
                "role": "user",
                "content": system_prompt
                }
            ],
            "reasoning": {"enabled": True}
        })
        )

    # Extract the assistant message with reasoning_details
    response = response.json()
    new_code = response['choices'][0]['message']    
    print(new_code.keys)
    # Fallback cleanup: If the LLM disobeys and wraps the code in markdown (```python ... ```), 
    # this regex strips those tags out before saving.
    try:
        if new_code.startswith("```"):
            new_code = re.sub(r"^```[a-zA-Z]*\n", "", new_code)
            new_code = re.sub(r"\n```$", "", new_code)
    except:
        print("Caught some useless error")
    # 4. Act
    write_file(filepath, new_code)
    print(f"Successfully updated {filepath}! Check VS Code.")

# --- Run the Agent ---
if __name__ == "__main__":
    # Example usage:
    target_file = "codingAgent/test.py" # Ensure this file exists in your directory
    user_instruction = "Refactor this code to use modern ES6 syntax and add error handling."
    
    edit_code(target_file, user_instruction)