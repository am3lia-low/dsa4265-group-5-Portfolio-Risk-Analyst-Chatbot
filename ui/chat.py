import streamlit as st

def render_chat_history(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            # # Show expander under every assistant answer
            # if msg["role"] == "assistant":
            #     with st.expander("More details"):
            #         st.write("Hidden content here")