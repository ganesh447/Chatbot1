import streamlit as st
from chatbot3 import answer_query

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Styling for chat bubbles
st.markdown("""
<style>
.user-message {
    background-color: #28231D;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
    align-self: flex-end;
}
.bot-message {
    background-color: #313639;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
}
.follow-up {
    font-style: calibri;
    color: #ffffff;
}
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("KowmaBot 1.0")
st.write("Ask about FAQs ")

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.history:
        # User message
        st.markdown(f'<div class="user-message"><b>You:</b> {msg["query"]}</div>', unsafe_allow_html=True)
        # Bot answer
        st.markdown(f'<div class="KowmaBot"><b>Bot:</b> {msg["answer"]}</div>', unsafe_allow_html=True)
        # Follow-up
        if msg["follow_up"]:
            st.markdown(f'<div class="bot-message follow-up"><b>KowmaBot:</b> {msg["follow_up"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input form
with st.form(key="input_form", clear_on_submit=True):
    query = st.text_input("Your question:", placeholder="Type here...", key=f"input_{st.session_state.input_key}")
    submit = st.form_submit_button("Send")

# Handle submission
if submit and query.strip():
    with st.spinner("Checking..."):
        answer, follow_up = answer_query(query, st.session_state.history)
        st.session_state.history.append({"query": query, "answer": answer, "follow_up": follow_up})
        st.session_state.input_key += 1  # Reset input field
    st.rerun()  # Updated from st.experimental_rerun()

# Follow-up button
if st.session_state.history and st.session_state.history[-1].get("follow_up"):
    if st.button("Yes, tell me more!"):
        with st.spinner("Checking..."):
            answer, follow_up = answer_query("yes", st.session_state.history)
            st.session_state.history.append({"query": "Yes", "answer": answer, "follow_up": follow_up})
            st.session_state.input_key += 1
        st.rerun()  # Updated from st.experimental_rerun()

# Clear chat
if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.input_key = 0
    st.rerun()  # Updated from st.experimental_rerun()