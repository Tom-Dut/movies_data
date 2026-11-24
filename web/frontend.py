import json
import requests
import streamlit as st

# URL de votre API FastAPI exigée par le sujet
API_URL = "http://localhost:8000/movie-rag-agent"

st.set_page_config(page_title="Movie RAG Chatbot", page_icon="🎬")
st.title("🎬 Assistant Cinéma (Neo4j & LangChain)")
st.caption("Posez vos questions sur les films, acteurs, réalisateurs, avis ou demandez des recommandations.")

# Initialisation de l'historique des messages dans la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages de la conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie en bas de l'écran
if prompt := st.chat_input("Ex: Quels sont les films de Tom Hanks ?"):
    # 1. Afficher et sauvegarder la question de l'utilisateur
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Envoyer la question au serveur FastAPI et afficher la réponse
    with st.chat_message("assistant"):
        answer = ""
        status = st.status("Traitement de votre demande…", expanded=True)
        try:
            with requests.post(
                API_URL,
                json={
                    "text": prompt,
                    "history": st.session_state.messages[:-1][-10:],
                },
                stream=True,
            ) as response:
                if response.status_code != 200:
                    answer = (
                        f"Erreur du serveur FastAPI "
                        f"(Code {response.status_code}) : {response.text}"
                    )
                    status.update(label="Échec de la requête", state="error")
                else:
                    for raw_line in response.iter_lines():
                        if not raw_line or not raw_line.startswith(b"data: "):
                            continue

                        event = json.loads(raw_line[6:].decode("utf-8"))
                        if event["type"] in {"progress", "tool_start", "tool_complete"}:
                            status.write(event["message"])
                        elif event["type"] == "final":
                            answer = event["output"]
                            status.update(label="Réponse prête", state="complete")
                        elif event["type"] == "error":
                            answer = event["message"]
                            status.update(label="Échec du traitement", state="error")

                    if not answer:
                        answer = "Le serveur a terminé le flux sans fournir de réponse."
                        status.update(label="Réponse manquante", state="error")
        except requests.exceptions.ConnectionError:
            answer = "❌ Impossible de joindre le serveur FastAPI. Vérifiez qu'Uvicorn tourne bien sur http://localhost:8000."
            status.update(label="Serveur FastAPI inaccessible", state="error")
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, UnicodeDecodeError) as error:
            answer = f"Erreur lors de la lecture de la réponse du serveur : {error}"
            status.update(label="Erreur de communication", state="error")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})