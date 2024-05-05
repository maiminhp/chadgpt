from typing import Any, Generator
from datetime import datetime
import streamlit as st

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama

LAST_RESPONSE_META = None


def get_response(user_query: str, chat_history: list[ChatMessage]) -> Generator[str, None, None]:
    """response to be streamed"""
    global LAST_RESPONSE_META

    template = PromptTemplate(
        "You are a helpful assistant. Answer the following questions considering the history of the conversation:\n"
        "---------------------\n"
        "Chat History\n"
        "{chat_history}\n"
        "---------------------\n"
        "User question: {user_question}\n"
    )
    prompt = template.format(chat_history=chat_history, user_question=user_query)

    llm = Ollama(model="llama3")

    for r in llm.stream_complete(prompt):
        if r.additional_kwargs["done"]:
            LAST_RESPONSE_META = r.additional_kwargs
        yield r.delta


def metadata(last_response_meta: dict[str, Any]) -> str:
    """Meta info re response timestamp, token per second and total duration generating a response"""
    fmt_string = "%Y-%m-%dT%H:%M:%S.%f"
    ts = datetime.strptime(last_response_meta["created_at"][:26], fmt_string)
    output_str = ""
    output_str += f"created_at {ts.strftime('%Y-%m-%d %H:%M:%S')}; "
    output_str += f"{round(last_response_meta['eval_count'] / (last_response_meta['eval_duration'] / 1_000_000_000), 2)} token/s; "
    output_str += f"total duration: {round(last_response_meta['total_duration'] / 1_000_000_000, 2)}s"
    return output_str


def main():
    global LAST_RESPONSE_META
    st.set_page_config(page_title="ChadGPT", page_icon="🤖")
    st.title("ChadGPT - 💪😎🏋️‍♂️")
    st.subheader("The poor man's ChatGPT")

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ChatMessage(
                role=MessageRole.CHATBOT,
                content="Hello, I am a bot. Beep boop. How can I help you?",
            ),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if message.role is MessageRole.CHATBOT:
            with st.chat_message("AI"):
                st.write(message.content)
        elif message.role is MessageRole.USER:
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_query)
        )

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(
                get_response(user_query, st.session_state.chat_history)
            )

        if LAST_RESPONSE_META is not None:
            st.caption(metadata(LAST_RESPONSE_META))
            LAST_RESPONSE_META = None

        st.session_state.chat_history.append(
            ChatMessage(role=MessageRole.CHATBOT, content=response)
        )


if __name__ == "__main__":
    main()
