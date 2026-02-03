import os
from google import genai
from google.genai import types  

client = genai.Client(
    api_key="AIzaSyBnucDnuP2L0w5uIHv_Y2HXJ29hJgN25yY"
)

def generate_answer(query, context):
    prompt = f"""
You are an AI tutor that helps users understand
PROGRAMMING SYNTAX clearly.

Answer format:
- Language:
- Syntax:
- Explanation:
- Example:
- Common mistakes:

If the question is comparative:
- Must Provide a comparison table
- Show syntax 
- Give one example for each


Use ONLY the context below.
Do NOT invent APIs or syntax.
Context:
{context}

Question:
{query}

Answer:
"""

    try:
        # Use the correct model string without leading whitespace
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                 max_output_tokens=15000
            )
        )
        return response.text.strip()

    except Exception as e:
        return f"Gemini error: {e}"
