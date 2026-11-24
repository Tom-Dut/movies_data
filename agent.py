from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import movies_data.user_consts as consts
from tools.similarity_tool import user_similarity_tool, movie_similarity_tool
from tools.recommendation_tool import recommendation_tool
from tools.review_chain import review_tool
from tools.cypher_qa_chain import graph_qa_tool

agent_tools = [
    review_tool,
    graph_qa_tool,
    user_similarity_tool,
    movie_similarity_tool,
    recommendation_tool,
]

agent_chat_model = ChatOllama(
    model=consts.OLLAMA_MODEL,
    base_url=consts.OLLAMA_BASE_URL,
    temperature=0,
    client_kwargs={
        "headers": {
            "Authorization": "Bearer " + consts.API_KEY,
        }
    },
)

agent_llm_with_tools = agent_chat_model.bind_tools(agent_tools)

movie_agent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Tu es un assistant spécialisé dans les films. "
        "Utilise l'outil adapté pour répondre aux questions sur les avis, "
        "les données du graphe, les similarités et les recommandations. "
        "Réponds dans la langue de l'utilisateur et n'invente pas de résultats.",
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

movie_agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", []),
        "agent_scratchpad": lambda x: format_to_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | movie_agent_prompt
    | agent_llm_with_tools
    | ToolsAgentOutputParser()
)

movie_agent_executor = AgentExecutor(
    agent=movie_agent,
    tools=agent_tools,
    max_iterations=5,
    handle_parsing_errors=True,
    verbose=True,
)