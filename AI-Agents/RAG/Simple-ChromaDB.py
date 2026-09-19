import chromadb
from pypdf import PdfReader
import ollama

# Client() function creates temporary chromadb in memory, in-memory client. Your data can disappear when the program/session ends.
client = chromadb.Client()

collection = client.create_collection(
    name="my_collection"
)

# chromadb operations
    # add() - adds the content with other metadata, embed only documents
    # get, update, delete, search, filter, query, vector ranking

# Saved on disk
#         ↓
# Available after restarting
# client = chromadb.PersistentClient(
#     path="./chroma_db"
# )

# collection = client.get_or_create_collection(
#     name="my_collection"
# )

def load_pdf(path):
    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

    return pages

def chunk_text(text, chunk_size=200, overlap=40):

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            continue

        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

    return chunks

def create_chunks(pages):

    chunks = []

    for page in pages:

        page_chunks = chunk_text(page["text"])

        for chunk_index, chunk in enumerate(page_chunks):

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "chunk_index": chunk_index
            })

    return chunks

if __name__ == '__main__':

    pdf_path = "documents/Artificial_Intelligence_Overview.pdf"

    pages = load_pdf(pdf_path)

    print(f"Loaded {len(pages)} pages")

    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks")

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = (
            f"page_{chunk['page']}"
            f"_chunk_{chunk['chunk_index']}"
        )

        ids.append(chunk_id)

        documents.append(chunk['text'])

        metadatas.append({
            "source": pdf_path,
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )


    print(f"Added {len(ids)} chunks to ChromaDB")

    while True:

        user_question = str(input("question : "))

        relevent_documents = collection.query(
            query_texts=[user_question],
            n_results=3
        )

        context = "\n".join(relevent_documents['documents'][0])

        prompt = f"""
            Answer the question using only the provided context.

            Context:
            {context}

            Question:
            {user_question}

            If the answer cannot be found in the context,
            say that the information is not available.
        """

        answer = ollama.chat(
            model="qwen2.5-coder:7b",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            stream=True
        )

        for chunk in answer:
             print(chunk.message.content, end="", flush=True)
