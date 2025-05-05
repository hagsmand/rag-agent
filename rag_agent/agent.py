import datetime
import asyncio
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm
from task_manager import AgentWithTaskManager
from google.adk.runners import Runner
import os
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from dotenv import load_dotenv
from embedding import Embedding

load_dotenv()

# def list_collections() -> dict:
#     """Lists all Milvus collections.
#     Returns:
#         dict: status and result or error msg.
#     """
#     milvus = MilvusConnection()
#     milvus.create_collection('test_rag')
#     collections = milvus.list_collections()
#     return collections

# def search_information(query: str, collection_name: str) -> dict:
#     """Searches Milvus for information related to a given city.

#     Args:
#         query (str): The query to search Milvus for.
#         collection_name (str): The name of the Milvus collection to search.
#     Returns:
#         dict: status and result or error msg.
#     """
#     milvus = MilvusConnection()
#     search_result = milvus.search(collection_name, query)
#     print(f'Search result: {search_result}')
#     return search_result

class RAGAgent(AgentWithTaskManager):
    """An agent that handles generating SQL queries."""
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self._embedding = Embedding(model_id="sentence-transformers/all-mpnet-base-v2")
        self._agent = self._build_agent()
        self._user_id = "remote_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
    
    def _build_agent(self) -> Agent:
        """Builds the LLM agent for answer with data from Wikipedia with RAG."""
        async def init_toolset():
            return await MCPToolset.from_server(
                connection_params=StdioServerParameters(
                    command='npx',
                    args=["-y",    
                        "@pinecone-database/mcp",
                    ],   
                    env={
                        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY"),
                        "HF_TOKEN": os.environ.get("HF_TOKEN")  # Add HuggingFace token
                    }
                )
            )
        
        tools, exit_stack = asyncio.run(init_toolset())

        return Agent(
            name="rag_agent",
            model=LiteLlm(
                model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            ),
            description=(
                "Answer user's question by retrieving information from Wikipedia index on Pinecone DB"
            ),
            instruction=(
                """You are a helpful assistant. You can answer user's question by retrieving information using tool 'search-docs' and you can use 'list-indexes' to get all indexes in Pinecone DB.
                """
            ),
            tools=[tools[1], tools[0]],
        )

