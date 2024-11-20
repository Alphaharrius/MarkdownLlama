from typing import List, Generator, LiteralString

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage

from pydantic import BaseModel


RETRIEVE_TEMPLATE = """
### System:
You are an respectful and honest assistant. You have to provide the relevant details to the user's question \
from the context provided to you with no less than 100 words. If you don't know the answer, just say you don't \
know. Don't try to make up an answer. In your answer if it contains latex equations please use the latex format. \
If you are providing equations DO NOT use the reference number, instead please provide the entire equation.

### Context:
{context}

### User:
{question}

### Response:
"""


QUESTION_REFORMAT = """
Based on the prior conversation with the user, reformat the last prompt by the user have explicit meanings.
Please subsitute all context-based abbreviations, acronyms, and pronouns with their full meanings.
Only reformat the prompt if it is really needed.

User's last prompt: {prompt}
"""


class MdRAG:

    class Chatbot:
        def __init__(self, rag: 'MdRAG'):
            self.rag = rag
            self.history: List[ChatMessage] = []

        def chat(self, message: LiteralString) -> LiteralString:
            self.history.append(ChatMessage(role='user', content=message))
            self.history = self.rag.chat(self.history)
            return self.history[-1].content

    def __init__(self, llama_host: str, chat_model: str, embed_model: str, 
                 data_chunk_size: int = 1000, data_chunk_overlap: int = 100):
        self.chat_client = Ollama(base_url=llama_host, model=chat_model)
        self.chat_model = chat_model
        self.retrieval_llm = OllamaLLM(model=chat_model)
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=data_chunk_size,
            chunk_overlap=data_chunk_overlap)
        self.vectorstore = None
        self.data_chain = None
        
    def build_data_chain(self, data_paths: List[LiteralString], encoding: LiteralString = 'utf-8'):
        loaders: Generator[TextLoader] = (TextLoader(file_path=path, encoding=encoding) for path in data_paths)
        documents: Generator[Document] = (loader.load() for loader in loaders)
        chunks: List = [chunk for doc in documents for chunk in self.text_splitter.split_documents(doc)]

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.__build_data_chain()

    def store_data_chain(self, folder_path: LiteralString):
        self.vectorstore.save_local(folder_path)

    def load_data_chain(self, folder_path: LiteralString):
        self.vectorstore = FAISS.load_local(
            folder_path=folder_path, embeddings=self.embeddings, allow_dangerous_deserialization=True)
        self.__build_data_chain()

    def __build_data_chain(self):
        retriever = self.vectorstore.as_retriever()

        prompt = PromptTemplate(template=RETRIEVE_TEMPLATE, input_variables=['context', 'question'])

        self.data_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.retrieval_llm
            | StrOutputParser()
        )
        
    def chat(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        class QuestionReformat(BaseModel):
            question: str

        structured = self.chat_client.as_structured_llm(QuestionReformat)
        query = messages.pop()
        prompt = QUESTION_REFORMAT.format(prompt=query.content)
        messages = [*messages, ChatMessage(role='user', content=prompt)]
        while True:
            try:
                json: LiteralString = structured.chat(messages=messages).message.content
                reformat = QuestionReformat.model_validate_json(json)
                break
            except: continue
        
        print(f"<q>\n{reformat.question}\n</q>\n")

        retrieved: LiteralString = self.data_chain.invoke(reformat.question)

        messages.append(ChatMessage(role='user', content=query.content))
        messages.append(ChatMessage(role='assistant', content=retrieved))

        return messages
    
    def chatbot(self):
        return self.Chatbot(self)
