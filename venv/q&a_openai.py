import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
from flask import Flask, request, jsonify
from pydantic import BaseModel
from flask_pydantic import validate

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = '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'
df = pd.read_csv('dataset_dg_token_terbaru.csv')
df = df.set_index(["kategori", "tanggal_publikasi"])
df = df.loc[~df.index.duplicated(keep='first')]
def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def load_embeddings(fname: str):
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "kategori" and c != "tanggal_publikasi"])
    return {
           (r.kategori, r.tanggal_publikasi): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

document_embeddings = load_embeddings("berita_tokens_embeddings6_newest.csv")

def vector_similarity(x, y) -> float:
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, contexts):
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.judul_berita.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    header = """Jawab pertanyaan sejujur mungkin menggunakan konteks yang disediakan, dan jika jawabannya tidak terdapat dalam teks di bawah ini, katakan "Saya tidak tahu."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

prompt = construct_prompt(
    "ada apa di jakarta",
    document_embeddings,
    df
)

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(query: str,df: pd.DataFrame,document_embeddings,show_prompt: bool = False) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

"""
example :

query = "dimanakah lampu LED dipasang?"
answer = answer_query_with_context(query, df, document_embeddings)

print(f"\nQ: {query}\nA: {answer}")

"""

app = Flask(__name__)

class Question(BaseModel):
    question : str

@app.route('/get_question', methods=['POST'])
@validate()
def get_answer(query : Question):
    answer = answer_query_with_context(query, df, document_embeddings)
    return jsonify({'answer': answer})

app.run(host='0.0.0.0', port=8080)