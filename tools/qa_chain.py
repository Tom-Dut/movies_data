from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import movies_data.user_consts as consts
from movies_data.neo4j_utils_py2neo import (
    run_predicted_cypher_query
)

qa_template_str = """
Task:
Your job is to answer the user's question using only the result
returned by the Neo4j database.

Use only the Neo4j result below.
Do not use external knowledge.
Do not add corrections, comments, notes or information that is
not explicitly present in the Neo4j result.
Do not make up information.
If the result is empty, say that no result was found.

Neo4j result:
{context}
"""
qa_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=qa_template_str
    )
)

qa_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
)

qa_prompt_template = ChatPromptTemplate(
    input_variables=[
        "context",
        "question"
    ],
    messages=[
        qa_system_prompt,
        qa_human_prompt
    ]
)

qa_chat_model = ChatOllama(
    model=consts.MOVIE_QA_MODEL,
    base_url=consts.OLLAMA_BASE_URL,
    temperature=0,
    client_kwargs={
        "headers": {
            "Authorization":
                "Bearer " + consts.API_KEY
        }
    }
)

qa_chain = (
    {
        "context":
            itemgetter("cypher")
            | RunnableLambda(
                run_predicted_cypher_query
            ),

        "question":
            itemgetter("question")
    }

    | qa_prompt_template
    | qa_chat_model
    | StrOutputParser()
)

if __name__ == "__main__":

    question = "What is the most recent movie?"

    cypher = """
    MATCH (m:Movie)
    RETURN m.title AS title, m.released AS released
    ORDER BY m.released DESC
    LIMIT 1
    """

    response = qa_chain.invoke(
        {
            "question": question,
            "cypher": cypher
        }
    )

    print(response)

