from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class TicketCreate(BaseModel):
    name: str
    contact: str
    query: str

class TicketResponse(BaseModel):
    id: int
    name: str
    contact: str
    query: str
    tags: Optional[Dict[str, Any]] = None
    embedding: Optional[bytes] = None
    cluster_id: Optional[int] = None
    cluster_summary: Optional[str] = None
    status: str = "pending"
    auto_reply: Optional[str] = None
    human_reply: Optional[str] = None
    created_at: datetime

class AgentReply(BaseModel):
    ticket_id: int
    reply_text: str

class KnowledgeEntry(BaseModel):
    query: str
    solution: str

class ClusterInfo(BaseModel):
    cluster_id: int
    summary: str
    ticket_count: int