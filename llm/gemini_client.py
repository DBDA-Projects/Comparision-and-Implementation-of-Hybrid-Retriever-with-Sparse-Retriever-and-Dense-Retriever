import streamlit as st
from google import genai
from google.genai import types  

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

def generate_answer(query, context):

    if not context or len(context.strip()) < 20:
        return "No relevant documentation found."

    prompt = f"""
You are an AI tutor that helps users understand PROGRAMMING SYNTAX.

Answer format:
- Language:
- Syntax:
- Explanation:
- Example:
- Common mistakes:

If the question is comparative:
- Provide a comparison table
- Show syntax
- Give one example for each

Use ONLY the context below.
Do NOT invent APIs or syntax.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=1024
            )
        )

        if response.text:
            return response.text.strip()
        else:
            return response.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        return f"Gemini error: {e}"
