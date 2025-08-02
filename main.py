import streamlit as st
from utils.ai_methods import get_vector_store, get_ai_result, search_vector_store

if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Indian Constitutional Law by MP Jain")

with st.spinner("Please wait while we load the documents...", show_time=True):
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(placeholder="Enter your query here...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    docs_data = search_vector_store(query, st.session_state.vector_store)
    result = get_ai_result(query, docs_data)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "ai", "content": result})

    with st.chat_message("ai"):
        st.markdown(result)
