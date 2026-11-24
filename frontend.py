import streamlit as st
import httpx
from pathlib import Path

st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🌿"
)

st.title("🌿 SkinSense AI")
st.caption("Your personal skincare knowledge assistant")

st.info(
    "Ask me about acne, skincare routines, "
    "moisturizers, and retinoids."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a skincare question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching skincare documents..."):
            try:
                response = httpx.post(
                    "http://127.0.0.1:8000/ask",
                    json={
                        "question": question,
                        "top_k": 4
                    },
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

                answer = result["answer"]

                sources = sorted({
                    Path(s["source"]).name
                    for s in result["sources"]
                })

                st.markdown(answer)

                if sources:
                    with st.expander("View sources"):
                        for source in sources:
                            st.write(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except httpx.HTTPError:
                st.error(
                    "Unable to get an answer. "
                    "Make sure the API is running."
                )

with st.sidebar:
    st.header("About SkinSense")
    st.write(
        "A RAG-powered skincare assistant "
        "using dermatology reference documents."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption("For educational purposes only. Not medical advice.")