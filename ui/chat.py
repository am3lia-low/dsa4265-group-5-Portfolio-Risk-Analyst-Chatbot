import streamlit as st

def render_status(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

def render_chat_history(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])