from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vectorseach import retriever
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
    print("\n\n")
    if question.lower() == "q":
        break

    curriculum = retriever.invoke(question)
    
    response_stream =chain.stream({
        "curriculum": curriculum,
        "question": question 
    })
    
    # printing each chunk of the response stream
    for chunk in response_stream:
        print(chunk, end="", flush=True)
    
    
    # Display sources
    print("\n\n===============")
    print("Sources:")
    
    unique_sources = set()
    for doc in curriculum:
        source_file = os.path.basename(doc.metadata['source'])
        page_number = doc.metadata['page']
        if page_number > 0:
            unique_sources.add(f"{source_file} (Page {page_number})")
        else:
            unique_sources.add(f"{source_file}")
    
    for source_info in unique_sources:
        print(f"\n- {source_info}")
    
    print("===============\n")