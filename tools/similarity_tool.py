from langchain_classic.agents import Tool
from langchain_neo4j import Neo4jGraph
import movies_data.user_consts as consts

graph = Neo4jGraph(
    url=consts.NEO4J_URI,
    username= consts.NEO4J_USERNAME,
    password=consts.NEO4J_PASSWORD,
    refresh_schema=False
)


def get_user_review_similarity(users_str: str) -> str:
    """
    Compare users by Jaccard similarity over the movies they rated.
    """
    users = [user.strip().strip("'\"") for user in users_str.split(",")]
    if len(users) != 2 or not all(users):
        return "Please provide exactly two non-empty user names separated by a comma."
    u1, u2 = users

    query = """
    OPTIONAL MATCH (u1:User {name: $u1})
    OPTIONAL MATCH (u2:User {name: $u2})
    OPTIONAL MATCH (u1)-[:RATED]->(m1:Movie)
    WITH u1, u2, collect(DISTINCT m1) AS movies1
    OPTIONAL MATCH (u2)-[:RATED]->(m2:Movie)
    WITH u1, u2, movies1, collect(DISTINCT m2) AS movies2
    WITH u1, u2, movies1, movies2,
         [m IN movies1 WHERE m IN movies2] AS common_movies,
         movies1 + [m IN movies2 WHERE NOT (m IN movies1)] AS all_movies
    RETURN u1 IS NOT NULL AS user1_exists,
           u2 IS NOT NULL AS user2_exists,
           size(common_movies) AS common_count,
           size(all_movies) AS total_count,
           CASE WHEN size(all_movies) = 0 THEN null
                ELSE round(100.0 * toFloat(size(common_movies)) / size(all_movies)) / 100.0
           END AS jaccard_similarity
    """
    result = graph.query(query, params={"u1": u1, "u2": u2})
    if not result:
        return "Could not calculate similarity for these users."

    row = result[0]
    if not row["user1_exists"]:
        return f"No user named '{u1}' was found."
    if not row["user2_exists"]:
        return f"No user named '{u2}' was found."
    if row["jaccard_similarity"] is None:
        return f"Neither {u1} nor {u2} has rated any movies, so similarity cannot be calculated."
    return (
        f"The movie-rating Jaccard similarity between {u1} and {u2} is "
        f"{row['jaccard_similarity']:.2f} "
        f"({row['common_count']} shared movies out of {row['total_count']} "
        "rated by either user). This compares which movies they rated, "
        "not how they rated them."
    )


def get_movie_genre_similarity(movie_title: str) -> str:
    """
    Return the five closest movies by genre Jaccard similarity.
    """
    movie_title = movie_title.strip().strip("'\"")
    if not movie_title:
        return "Please provide a movie title."

    movie_result = graph.query(
        "MATCH (m:Movie {title: $title}) RETURN m.title AS title LIMIT 1",
        params={"title": movie_title},
    )
    if not movie_result:
        return f"No movie titled '{movie_title}' was found."

    query = """
    MATCH (m1:Movie {title: $title})-[:IN_GENRE]->(g:Genre)
    WITH m1, collect(DISTINCT g) AS g1
    MATCH (m2:Movie)-[:IN_GENRE]->(g:Genre)
    WHERE m2 <> m1
    WITH m1, g1, m2, collect(DISTINCT g) AS g2
    WITH m2.title AS similar_movie,
         [x IN g1 WHERE x IN g2] AS inter,
         g1 + [x IN g2 WHERE NOT x IN g1] AS union
    WHERE size(union) > 0
    RETURN similar_movie,
           round(100.0 * toFloat(size(inter)) / size(union)) / 100.0 AS similarity
    ORDER BY similarity DESC, similar_movie ASC
    LIMIT 5
    """
    result = graph.query(query, params={"title": movie_title})
    if not result:
        return (
            f"No genre-based matches were found for '{movie_title}'. "
            "The movie may not have genre relationships in the database."
        )

    matches = "\n".join(
        f"- {row['similar_movie']} (genre similarity: {row['similarity']:.2f})"
        for row in result
    )
    return f"Movies most similar to '{movie_title}' by genre:\n{matches}"


user_similarity_tool = Tool(
    name="UserReviewSimilarity",
    func=get_user_review_similarity,
    description="""Calculates Jaccard similarity based on the sets of movies two users rated. It does not compare their rating values. Input must be exactly two user names separated by a comma.
    For instance, if the prompt is "Which movies have Darlene Garcia and Robert Brooks both rated?",
    the input should be "Darlene Garcia, Robert Brooks".""",
)

movie_similarity_tool = Tool(
    name="MovieGenreSimilarity",
    func=get_movie_genre_similarity,
    description="""Returns up to five movies ranked by Jaccard similarity of genre sets, including each score.
    Input must be the exact movie title. For instance, if the prompt is "Which movies are most similar to Hollow Man by genre?",
    the input should be "Hollow Man".""",
)