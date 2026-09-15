
from rag import (
    load_doc,
    normalize_text,
    lemmatize_text,
    create_chunk,
    create_embedding_model,
    create_vector_db,
    retrieval_chunk,
    generate_answer
)

# calling function from rag.py

data = load_doc('data.txt')

data = normalize_text(data)

data = lemmatize_text(data)

chunks = create_chunk(data)

embedding_model = create_embedding_model()

vectordb = create_vector_db(chunks,embedding_model)

# asking query from user

print("-----------------Hey! Throw your questions at me, let’s see what we can find.-----------------")
while True:
    query = input("\nquery:")
    if query.lower() == "exit":
        print("Catch ya later! Let me know when you wanna dig into more docs.")
        break

    retrieval_text = retrieval_chunk(vectordb,query,k=3)

    response = generate_answer(retrieval_text,query)

    print(response)
   
    print("\n\nType 'exit' to terminate.")

