"""
RAG Pipeline Step 4: The LLM / Generation Layer
══════════════════════════════════════════════════════════════════════════════
This file contains the final step of our RAG (Retrieval-Augmented Generation) 
pipeline. After we have retrieved relevant code chunks from ChromaDB (Step 3),
we need to pass them to a Large Language Model (LLM) to generate a human-readable 
answer.

This module is responsible for:
1. Prompt Engineering: Building a carefully structured prompt that combines the 
   user's question with the retrieved code context.
2. API Communication: Sending this prompt to the Gemini API and returning the 
   generated answer.

🔗 CONNECTS TO:
- app.py: Calls these functions to get the final answer to show to the user.
- embedder.py: Provides the 'chunks' that are passed into build_prompt.
"""

import os
from google import genai

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING SECTION
# ══════════════════════════════════════════════════════════════════════════════
def build_prompt(question: str, chunks: list[dict], directory_tree: str, chat_history: list[dict] = None) -> str:
    """Constructs the full prompt sent to Gemini.
    Format:
      SYSTEM: You are a code expert...
      CONVERSATION HISTORY: (last N Q&A pairs for follow-up context)
      REPO STRUCTURE: (directory tree)
      CODE CONTEXT: (retrieved chunks with file names and line numbers)
      QUESTION: (user's question)
    """
    prompt_parts = []
    
    # 📚 CONCEPT: System Prompt & Grounding
    # This first part acts like a "System Prompt". It sets the persona and rules.
    # 💡 WHY: We explicitly tell it to answer "ONLY based on the provided code context". 
    # This is called "grounding" and it's the core idea of RAG! It prevents the LLM 
    # from hallucinating (making up code that doesn't exist in the repo).
    prompt_parts.append("You are an expert code analyst. Answer questions about a codebase based ONLY on the provided code context. Be specific — reference file names and line numbers.")
    
    # [1] Add Conversation History (for follow-up questions)
    # 📚 CONCEPT: Conversation Memory
    # Without this, if a user asks "What is RAG?" and then "Explain in detail",
    # the LLM has NO IDEA what "it" refers to. By including the last few Q&A pairs,
    # the LLM can understand follow-up questions naturally.
    # 💡 WHY only last 5? Including too much history wastes tokens (which cost money
    # and slow down responses). 5 pairs is enough context for most follow-ups.
    if chat_history:
        # Keep only the last 5 Q&A pairs (10 messages) to avoid exceeding token limits
        recent_history = chat_history[-10:]
        prompt_parts.append("\nCONVERSATION HISTORY (for follow-up context):")
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append("(End of conversation history)\n")
    
    # [2] Add Repository Structure
    # Providing the folder tree helps the LLM understand the big picture (architecture),
    # which is often lost when we only look at small chunks of code.
    if directory_tree:
        prompt_parts.append("\nREPOSITORY STRUCTURE:")
        prompt_parts.append(directory_tree)
    
    # [3] Inject Retrieved Context (The 'R' in RAG)
    # 📚 CONCEPT: Context Augmentation
    # Here we take the chunks retrieved by vector search and inject them into the prompt.
    prompt_parts.append("\nRELEVANT CODE (retrieved from vector search):")
    if not chunks:
        prompt_parts.append("No relevant code chunks found.")
    else:
        for chunk in chunks:
            # We include metadata like file_path and line numbers.
            # 💡 WHY: This allows the LLM to cite its sources accurately in the answer.
            meta = chunk['metadata']
            file_path = meta.get('file_path', 'unknown')
            start_line = meta.get('start_line', '?')
            end_line = meta.get('end_line', '?')
            
            prompt_parts.append(f"---\n📄 File: {file_path} | Lines {start_line}-{end_line}")
            prompt_parts.append(chunk['content'])
            
    # [4] Add User Question
    # Finally, we append the actual user question at the end.
    prompt_parts.append(f"---\n\nQUESTION: {question}")
    
    # [5] Output Rules
    # 📚 CONCEPT: Prompt Constraints
    # Reiterating rules at the end of a long prompt helps the LLM remember them 
    # (LLMs can suffer from "lost in the middle" syndrome).
    prompt_parts.append("\nRules:\n- Reference specific files and line numbers\n- If the context doesn't contain enough info, say so\n- Use markdown code blocks for code examples\n- For architecture questions, use the repository structure above\n- If the user's question refers to a previous answer, use the conversation history above for context")
    
    return "\n".join(prompt_parts)

# ══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION SECTION
# ══════════════════════════════════════════════════════════════════════════════
def ask_gemini(prompt: str) -> str:
    """Calls Gemini 3.5 Flash via google-genai SDK (free tier). 
    Includes retry with exponential backoff for rate limits.
    Returns the generated text answer."""
    
    # 💡 WHY: API keys should never be hardcoded. We load them from environment 
    # variables (.env file) for security.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Error: GEMINI_API_KEY is not set. Please add it to your .env file."
        
    # Initialize the official Google GenAI SDK client
    # 📚 CONCEPT: SDK (Software Development Kit)
    # This handles all the complex HTTP requests to Google's servers for us.
    client = genai.Client(api_key=api_key)
    
    try:
        # 📚 CONCEPT: Context Window and Token Limits
        # Every LLM has a limit to how much text (tokens) it can read at once.
        # The prompt we built above must fit within this limit. Gemini Flash has a large 
        # context window, making it great for RAG.
        # Note: In a more advanced setup, you can also control the "temperature" parameter 
        # here. Temperature controls creativity (0 = strict/deterministic, 1 = creative). 
        # For code RAG, a lower temperature is usually better to prevent hallucinations.
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error calling Gemini API: {str(e)}"
