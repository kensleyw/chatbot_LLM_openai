from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from openai import AuthenticationError
import streamlit as st

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
st.header("IA Generativa - Chatbot")
st.write("Criando chatbot com LLM da OpenAI")
st.markdown("****")

with st.sidebar:
    openai_api_key=st.text_input("OpenAI API Key * (obrigatório)", type="password")
    
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="Olá. Sou seu assistente virtual. Como posso ajudar?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)
    
prompt = st.chat_input(placeholder="Digite sua mensagem")

if prompt:
    st.session_state.messages.append(ChatMessage(role="user", content="contexto: esporte orientação " + prompt))
    st.chat_message("user").write(prompt)
    
    if not openai_api_key:
        st.error("Por favor, insira a OpenAI API Key")
        st.stop()
     
    try:   
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
            response = llm(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))

    except AuthenticationError as auth_error:
        st.error(f"Ocorreu um erro de autenticação, confira se sua **OpenAI API Key** está correta.\n\n {auth_error}")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")