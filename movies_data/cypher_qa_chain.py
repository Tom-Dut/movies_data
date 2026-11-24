from operator import itemgetter

import movies_data.user_consts as consts

from movies_data.cypher_chain import (
    cypher_chain,
    neo4j_schema
)

from movies_data.qa_chain import qa_chain

cypher_qa_chain = (
    {
        "question":
            itemgetter("question"),

        "cypher":
            cypher_chain
    }

    | qa_chain
)

if __name__ == "__main__":

    question = "What is the most recent movie?"

    response = cypher_qa_chain.invoke(
        {
            "question": question,
            "schema": neo4j_schema
        }
    )

    print(response)