from operator import itemgetter
from langchain_classic.agents import Tool
from tools.cypher_chain import (
    cypher_chain,
    neo4j_schema
)
from tools.qa_chain import qa_chain

cypher_qa_chain = (
    {
        "question":
            itemgetter("question"),

        "cypher":
            cypher_chain
    }

    | qa_chain
)


def answer_graph_question(question: str) -> str:
    return cypher_qa_chain.invoke({
        "question": question,
        "schema": neo4j_schema,
    })


graph_qa_tool = Tool(
    name="MovieDatabaseQA",
    func=answer_graph_question,
    description=(
        "Répond aux questions sur les données du graphe Neo4j, par exemple "
        "les films, acteurs, réalisateurs ou notes. Entrée : la question complète."
    ),
    return_direct=True,
)


if __name__ == "__main__":

    question = "What is the average rating value for the movie Hollow Man?"

    response = cypher_qa_chain.invoke(
        {
            "question": question,
            "schema": neo4j_schema
        }
    )

    print(response)