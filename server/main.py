import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

VECTOR_STORE_PROVIDER = os.environ.get("VECTOR_STORE_PROVIDER", "milvus")
VECTOR_STORE_HOST = os.environ.get("VECTOR_STORE_HOST", "")
VECTOR_STORE_DIMS = os.environ.get("VECTOR_STORE_DIMS", "4096")
VECTOR_STORE_USER = os.environ.get("VECTOR_STORE_USER", "user")
VECTOR_STORE_PASSWORD = os.environ.get("VECTORSTORE_PASSWORD", "password")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "openai")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "openai")
LLM_MAX_TOKENS = os.environ.get("LLM_MAX_TOKENS", "2000")

EMBEDDER_PROVIDER = os.environ.get("EMBEDDER_PROVIDER", "openai")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "")
EMBEDDER_API_KEY = os.environ.get("EMBEDDER_API_KEY", "")
EMBEDDER_BASE_URL = os.environ.get("EMBEDDER_BASE_URL", "")

GRAPH_STORE_PROVIDER = os.environ.get("GRAPH_STORE_PROVIDER", "neo4j")
GRAPH_STORE_HOST = os.environ.get("GRAPH_STORE_HOST", "")
GRAPH_STORE_USER = os.environ.get("GRAPH_STORE_USER", "user")
GRAPH_STORE_PASSWORD = os.environ.get("VECTORDB_PASSWORD", "password")

SERVER_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": VECTOR_STORE_PROVIDER,
        "config": {
            "url": VECTOR_STORE_HOST,
            "embedding_model_dims": VECTOR_STORE_DIMS,
            #"user": VECTOR_STORE_USER,
            #"password": VECTOR_STORE_PASSWORD,
            "collection_name": {},
            "token": "8e4b8ca9cf2c67"
        },
    },
    "graph_store": {
        "provider": GRAPH_STORE_PROVIDER,
        "config": {
            "url": GRAPH_STORE_HOST, 
            "username": GRAPH_STORE_USER, 
            "password": GRAPH_STORE_PASSWORD
        }
    },
    "llm": {
        "provider": LLM_PROVIDER, 
        "config": {
            "api_key": LLM_API_KEY, 
            "temperature": 0.2, 
            "model": LLM_MODEL,
            "max_tokens": int(LLM_MAX_TOKENS)
        }
    },
    "embedder": {
        "provider": EMBEDDER_PROVIDER, 
        "config": {
            "api_key": EMBEDDER_API_KEY, 
            "model": EMBEDDER_MODEL
        }
    }
}

SERVER_CONFIG['llm']['config'][f'{LLM_PROVIDER}_base_url'] = LLM_BASE_URL
SERVER_CONFIG['embedder']['config'][f'{EMBEDDER_PROVIDER}_base_url'] = EMBEDDER_BASE_URL

app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
)

@app.post("/v1/register", summary="Register the project, user should call the API when add/create a project")
def register(project_id: str, project_name: Optional[str] = None): 
    ## TODO: generate collection name for vector store, generate space for graph store
    return None

@app.post("/v1/unregister", summary="Register the project, user should call the API when remove a project")
def unregister(project_id: str, project_name: Optional[str] = None): 
    ## TODO: remove project related info, WON'T clear memories the project related
    return None

class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")

class MemoryCreate(BaseModel):
    project_id: str
    member_id: str
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    project_id: str
    member_id: str
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

######## project related APIs
def getConfig(project_id):
    ## TODO: read collection_name from database
    SERVER_CONFIG["vector_store"]['config']['collection_name'] = project_id
    logging.info(SERVER_CONFIG)
    return SERVER_CONFIG

#@app.post("/v1/configure/update", summary="Use the API to update some user customized settings.")
def set_config(
    project_id: str,
    config: Dict[str, Any]):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}

######### memory and entity related APIs
memInstances = {}

def getMemInstance(project_id: str):
    if (project_id not in memInstances):
        config = getConfig(project_id)
        memInstances[project_id] = Memory.from_config(config)
        
    return memInstances[project_id]

@app.post("/v1/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = getMemInstance(memory_create.project_id).add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/memories", summary="Get memories")
def get_all_memories(
    project_id: str,
    member_id: str,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    MEMORY_INSTANCE = getMemInstance(project_id)
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/memories/{memory_id}", summary="Get a memory based on it's id")
def get_memory(
    project_id: str,
    member_id: str,
    memory_id: str):
    """Retrieve a specific memory by ID."""
    try:
        return getMemInstance(project_id).get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/memories/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return getMemInstance(search_req.project_id).search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/memories/{memory_id}", summary="Update a memory")
def update_memory(
    project_id: str,
    member_id: str,   
    memory_id: str, 
    updated_memory: Dict[str, Any]):
    """Update an existing memory."""
    try:
        return getMemInstance(project_id).update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/{memory_id}/history", summary="Get memory history according to it's id")
def memory_history(
    project_id: str,
    member_id: str,   
    memory_id: str):
    """Retrieve memory history."""
    try:
        return getMemInstance(project_id).history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/{memory_id}", summary="Delete a memory according to it's id")
def delete_memory(
    project_id: str,
    member_id: str,   
    memory_id: str):
    """Delete a specific memory by ID."""
    try:
        getMemInstance(project_id).delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories", summary="Delete all memories related")
def delete_all_memories(
    project_id: str,
    member_id: str,   
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        getMemInstance(project_id).delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/reset", summary="Reset all memories")
def reset_memory(
    project_id: str,
    member_id: str):
    """Completely reset stored memories."""
    try:
        getMemInstance(project_id).reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
