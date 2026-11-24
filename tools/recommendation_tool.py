from langchain_classic.agents import Tool
from langchain_neo4j import Neo4jGraph
import movies_data.user_consts as consts

graph = Neo4jGraph(
    url=consts.NEO4J_URI,
    username= consts.NEO4J_USERNAME,
    password=consts.NEO4J_PASSWORD,
    refresh_schema=False
)

def recommend_movie_for_user(user_name: str) -> str:
    """
    Return popular highly rated movies the user has not rated yet.
    """
    user_name = user_name.strip().strip("'\"")
    if not user_name:
        return "Please provide the user's full name."

    user_result = graph.query(
        "MATCH (u:User {name: $user_name}) RETURN u.name AS name LIMIT 1",
        params={"user_name": user_name},
    )
    if not user_result:
        return f"No user named '{user_name}' was found."

    query = """
    MATCH (u:User {name: $user_name})
    OPTIONAL MATCH (u)-[:RATED]->(rated:Movie)
    WITH u, collect(DISTINCT rated) AS rated_movies
    MATCH (other:User)-[r:RATED]->(m:Movie)
    WHERE other <> u
      AND r.rating > 3.0
      AND NOT (m IN rated_movies)
    RETURN m.title AS title, count(DISTINCT other) AS positive_users
    ORDER BY positive_users DESC, title ASC
    LIMIT 5
    """
    result = graph.query(query, params={"user_name": user_name})
    if not result:
        return (
            f"No popular highly rated movies were found that {user_name} "
            "has not already rated."
        )

    recommendations = "\n".join(
        f"- {row['title']} ({row['positive_users']} users rated it above 3.0)"
        for row in result
    )
    return (
        f"Popular recommendations for {user_name} "
        "(not personalized, based on positive ratings):\n"
        f"{recommendations}"
    )


recommendation_tool = Tool(
    name="MovieRecommendation",
    func=recommend_movie_for_user,
    description="""Returns up to five popular movies not yet rated by a user, based on the number of other users rating them above 3.0. This is popularity-based, not personalized to the user's tastes.
    Input must be the user's full name. For instance, if the prompt is "Which popular movies has Robert Brooks not rated?",
    the input should be "Robert Brooks".""",
)