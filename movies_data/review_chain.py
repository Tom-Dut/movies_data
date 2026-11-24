import os

from operator import itemgetter

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain_core.output_parsers import StrOutputParser

import movies_data.user_consts as consts

REVIEWS_CSV_PATH = os.path.join(
    consts.IN_FOLDER,
    "movie",
    "movie_reviews.csv"
)

REVIEWS_CHROMA_PATH = os.path.join(
    consts.OUT_FOLDER,
    "chroma_movie_reviews"
)

embeddings = OllamaEmbeddings(
    model=consts.OLLAMA_TEXT_SIMILARITY_MODEL,
    base_url=consts.OLLAMA_BASE_URL,
    client_kwargs={
        "headers": {
            "Authorization":
                "Bearer " + consts.API_KEY
        }
    }
)

if not os.path.exists(REVIEWS_CHROMA_PATH):

    loader = CSVLoader(
        file_path=REVIEWS_CSV_PATH,
        source_column="movie_review",
        metadata_columns=[
            "movie_name",
            "review_sentiment",
            "movie_rating"
        ]
    )

    reviews = loader.load()

    review_vector_db = Chroma.from_documents(
        reviews,
        embeddings,
        persist_directory=REVIEWS_CHROMA_PATH
    )

review_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embeddings
)

reviews_retriever = review_vector_db.as_retriever(
    search_kwargs={"k": 10}
)

review_template_str = """
Task:
Your job is to use movie reviews to answer questions
about users' opinions of movies.

Use only the following context to answer the question.
Do not make up information that is not present in the context.
If you do not know the answer, say you do not know.

Context:
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
)

review_prompt_template = ChatPromptTemplate(
    input_variables=[
        "context",
        "question"
    ],
    messages=[
        review_system_prompt,
        review_human_prompt
    ]
)

chat_model = ChatOllama(
    model=consts.OLLAMA_MODEL,
    base_url=consts.OLLAMA_BASE_URL,
    temperature=0,
    client_kwargs={
        "headers": {
            "Authorization":
                "Bearer " + consts.API_KEY
        }
    }
)

review_chain = (
    {
        "context":
            itemgetter("question")
            | reviews_retriever,

        "question":
            itemgetter("question")
    }
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

if __name__ == "__main__":

    question = """
    What do users think about Stanley & Iris?
    """

    response = review_chain.invoke(
        {
            "question": question
        }
    )

    print(response)