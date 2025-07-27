from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vectorseach import retriever
import os

model = OllamaLLM(model='llama3.2')

template = """
You are a Data Science and ML/AI Enginner teacher, who is helpful in teaching their students in a way that they will actually understand the concepts.

Here is your current curriculim:
{curriculum}

Here is the question the student has asked:
{question}

It is important to keep your response short and concise.
Use daily speech, and talk in a down to earth way.
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
    result = chain.invoke({
    "curriculum": curriculum,
    "question": question
    })
    print(result)
    
    # Display sources
    sources = set(doc.metadata['source'] for doc in curriculum)
    
    for source_file in sources:
        print(f"- {os.path.basename(source_file)}")