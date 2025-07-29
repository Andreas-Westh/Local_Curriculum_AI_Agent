from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from src.vectorseach import retriever
from src.spinner import start_spinner, stop_spinner
import os

model = OllamaLLM(model='llama3.2')

template = """
You are an expert AI and Data Science teacher with a talent for making complex topics feel simple. 
Your main goal is to explain concepts with absolute clarity.

Here are your rules:
1.  **Introduce and Explain:** When you use a technical term (like "embedding" or "backpropagation"), you MUST immediately follow it with a simple, down-to-earth explanation.
2.  **Explain how it actually works:** When you explain a concept, you MUST provide a clear, step-by-step explanation of how it works, not just the general theory, explain how the theory connects with the real world.
3.  **Be Direct:** Get straight to the point. Keep the explanation as short and concise as possible without losing the meaning.

Based on the provided curriculum text, answer the student's question.

---
Curriculum:
{curriculum}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True: 
    print("\n-=-=-=-=-=-=-=-=-")
    question = input("Enter your question (q to quit): ")
    print("\n")
    if question.lower() == "q":
        break

    # Start the spinner to show the AI is working.
    start_spinner()
    
    curriculum = retriever.invoke(question)
    
    response_stream = chain.stream({
        "curriculum": curriculum,
        "question": question 
    })
    
    # This flag helps us know when to stop the spinner.
    first_chunk = True
    for chunk in response_stream:
        # As soon as we get the first piece of the answer, stop the animation.
        if first_chunk:
            stop_spinner()
            first_chunk = False
        
        print(chunk, end="", flush=True)
    
    # Just in case the spinner is still running (e.g., if there's no response).
    stop_spinner()
    
    # Display sources
    print("\n\n===============")
    print("Sources:")
    
    unique_sources = set()
    for doc in curriculum:
        source_file = os.path.basename(doc.metadata['source'])
        
        if source_file == 'schedule.txt':
            unique_sources.add(f"{source_file} (Course schedule)")
        else:
            page_number = doc.metadata.get('page', -1)
            if page_number >= 0:
                unique_sources.add(f"{source_file} (Page {page_number + 1})")
            else:
                unique_sources.add(f"{source_file}")
    
    for source_info in sorted(unique_sources):
        print(f"\n- {source_info}")
    
    print("===============\n")