from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.chains import LLMChain
# define the memory object to add to the llm_chain object
from langchain.memory.buffer import ConversationBufferMemory
from gen_ai_hub.proxy.langchain.init_models import init_llm

# Load Library
from ai_core_sdk.ai_core_v2_client import AICoreV2Client

# Create Connection
ai_core_client = AICoreV2Client(
    base_url =  os.getenv("BASE_URL")+ "/v2", # The present SAP AI Core API version is 2
    auth_url=  os.getenv("AUTH_URL") + "/oauth/token", # Suffix to add
    client_id = os.getenv("CLIENT_ID"),
    client_secret = os.getenv("CLIENT_SECRET")
    # aicore_resource_group=os.getenv("AICORE_RESOURCE_GROUP")
)

os.environ['AICORE_RESOURCE_GROUP'] = os.getenv("AICORE_RESOURCE_GROUP")
os.environ['AICORE_AUTH_URL'] = os.getenv("AUTH_URL") + "/oauth/token"
os.environ['AICORE_CLIENT_ID'] = os.getenv("CLIENT_ID")
os.environ['AICORE_CLIENT_SECRET'] = os.getenv("CLIENT_SECRET")
os.environ['AICORE_BASE_URL'] = os.getenv("BASE_URL")+ "/v2"

# add the chat_history variable to our prompt template:
ice_cream_assistant_template = """
You are an ice cream assistant chatbot named "Scoopsie". Your expertise is
exclusively in providing information and advice about anything related to
ice creams. If a question is not about ice cream, respond with, "I specialize
only in ice cream related queries."
Chat History: {chat_history}
Question: {question}
Answer:"""

ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=ice_cream_assistant_template
)

# @cl.on_chat_start. This function prepares our model, memory object, and llm_chain for user interaction.
@cl.on_chat_start
def quey_llm():
    # llm=ChatGroq(
    # temperature=0.0,
    # groq_api_key=os.environ['GROQ_API_KEY'] ,
    # model_name="llama-3.1-70b-versatile"
# )
    llm=init_llm('gpt-4o', temperature=0.0, max_tokens=256)
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    llm_chain = LLMChain(llm=llm,
                         prompt=ice_cream_assistant_prompt_template,
                         memory=conversation_memory)

    cl.user_session.set("llm_chain", llm_chain)
    

@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")

    response = await llm_chain.acall(message.content,
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])

    await cl.Message(response["text"]).send()
