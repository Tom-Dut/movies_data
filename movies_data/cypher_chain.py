import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

import movies_data.user_consts as consts

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from operator import itemgetter

from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain_core.output_parsers import StrOutputParser

from movies_data.neo4j_utils_py2neo import construct_neo4j_graph_schema, extract_cypher, correct_cypher_query



def format_cypher_examples(documents: list[Document]) -> str:

    result = ""

    for document in documents:

        question = (
            document.page_content
            .replace("question: ", "")
            .strip()
        )

        cypher = document.metadata["cypher"]

        result += "Question:\n"
        result += question
        result += "\nCypher:\n"
        result += cypher
        result += "\n\n"

    return result


CYPHER_EXAMPLES_CSV_PATH = os.path.join(
    consts.IN_FOLDER,
    "movie",
    "movie_example_cypher.csv"
)

CYPHER_EXAMPLES_CHROMA_PATH = os.path.join(
    consts.OUT_FOLDER,
    "chroma_movie_cypher_examples"
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


if not os.path.exists(
    CYPHER_EXAMPLES_CHROMA_PATH
):

    loader = CSVLoader(
        file_path=CYPHER_EXAMPLES_CSV_PATH,
        source_column="question",
        metadata_columns=["cypher"]
    )

    examples = loader.load()

    cypher_examples_vector_db = Chroma.from_documents(
        examples,
        embeddings,
        persist_directory=CYPHER_EXAMPLES_CHROMA_PATH
    )


cypher_examples_vector_db = Chroma(
    persist_directory=CYPHER_EXAMPLES_CHROMA_PATH,
    embedding_function=embeddings
)


cypher_examples_retriever = (
    cypher_examples_vector_db.as_retriever(
        search_kwargs={"k": 3}
    )
)

neo4j_schema = construct_neo4j_graph_schema(
    include_types=[],
    exclude_types=[]
)

cypher_template_str = """
Task:
Your job is to generate a Cypher query from a user's question.

Use only the Neo4j graph schema below.

Neo4j graph schema:
{schema}

Here are examples of questions and their corresponding Cypher queries:

{examples}

Instructions:
Return only the Cypher query.
Do not explain your answer.
Do not invent labels, relationships or properties.
Respect the relationship directions given in the schema.
Use valid Cypher syntax.
Place ORDER BY and LIMIT after RETURN or WITH.
"""

cypher_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=[
            "schema",
            "examples"
        ],
        template=cypher_template_str
    )
)

cypher_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
)

cypher_prompt_template = ChatPromptTemplate(
    input_variables=[
        "schema",
        "examples",
        "question"
    ],
    messages=[
        cypher_system_prompt,
        cypher_human_prompt
    ]
)

cypher_chat_model = ChatOllama(
    model=consts.MOVIE_CYPHER_MODEL,
    base_url=consts.OLLAMA_BASE_URL,
    temperature=0,
    client_kwargs={
        "headers": {
            "Authorization":
                "Bearer " + consts.API_KEY
        }
    }
)

cypher_chain = (
    {
        "examples":
            itemgetter("question")
            | cypher_examples_retriever
            | RunnableLambda(
                format_cypher_examples
            ),

        "schema":
            itemgetter("schema"),

        "question":
            itemgetter("question")
    }

    | cypher_prompt_template
    | cypher_chat_model
    | StrOutputParser()
    | RunnableLambda(extract_cypher)
    | RunnableLambda(correct_cypher_query)
)




if __name__ == "__main__":

        
    question = "What are the movies directed by Martin Ritt?"

    cypher = cypher_chain.invoke(
        {
            "question": question,
            "schema": neo4j_schema
        }
    )

    print("Question :")
    print(question)

    print("\nCypher généré :")
    print(cypher)