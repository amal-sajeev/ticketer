# AI Customer Service Ticket Application
# FastAPI + MongoDB + Qdrant + Ollama + Sentence Transformers

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
import asyncio
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import httpx
import json
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Required for cloud Qdrant
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Enums
class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    INTERNET = "internet"
    ACCOUNT = "account"
    GENERAL = "general"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class AgentCategory(str, Enum):
    BILLING_SPECIALIST = "billing_specialist"
    TECH_SUPPORT = "tech_support"
    NETWORK_SPECIALIST = "network_specialist"
    GENERAL_SUPPORT = "general_support"
    SENIOR_SUPPORT = "senior_support"

class TicketStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"

class ResolutionType(str, Enum):
    AI = "ai"
    MANUAL = "manual"
    HYBRID = "hybrid"

class NoteType(str, Enum):
    INTERNAL = "internal"
    CUSTOMER_FACING = "customer_facing"

# Pydantic models
class TicketCreate(BaseModel):
    title: str
    description: str
    customer_email: str
    customer_name: str



class TicketResponse(BaseModel):
    id: str
    title: str
    description: str
    customer_email: str
    customer_name: str
    category: TicketCategory
    priority: Priority
    agent_category: AgentCategory
    status: TicketStatus
    sentiment: Sentiment
    created_at: datetime
    updated_at: datetime
    ai_resolution: Optional[str] = None
    manual_resolution: Optional[str] = None  # New field
    resolution_type: Optional[ResolutionType] = None  # New field
    confidence_score: Optional[float] = None
    assigned_agent: Optional[str] = None
    resolution_feedback: Optional[int] = Field(None, ge=1, le=5)  # New field
    notes: List[Dict] = Field(default_factory=list)  # New field


class KnowledgeEntry(BaseModel):
    title: str
    content: str
    category: TicketCategory

class ServiceMemoryEntry(BaseModel):
    query: str
    resolution: str
    category: TicketCategory
    agent_name: str

class AnalyticsResponse(BaseModel):
    total_tickets: int
    resolved_by_ai: int
    escalated_to_human: int
    avg_resolution_time: float
    sentiment_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    top_issues: List[Dict[str, Any]]

class ResolutionType(str, Enum):
    AI = "ai"
    MANUAL = "manual"
    HYBRID = "hybrid"

class NoteType(str, Enum):
    INTERNAL = "internal"
    CUSTOMER_FACING = "customer_facing"

class ManualResolution(BaseModel):
    resolution: str
    agent: str
    add_to_service_memory: bool = True

class ResolutionFeedback(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None

class TicketNote(BaseModel):
    content: str
    agent: str
    note_type: NoteType = NoteType.INTERNAL

class TicketAssignment(BaseModel):
    agent_name: str

# Global variables
app = FastAPI(title="AI Customer Service Ticket System")
db_client = None
db = None
qdrant_client = None
embedding_model = None
executor = ThreadPoolExecutor(max_workers=10)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_db()
    await startup_qdrant()
    await startup_embedding()
    yield
    # Shutdown
    if db_client:
        db_client.close()

app.router.lifespan_context = lifespan

async def startup_db():
    global db_client, db
    db_client = MongoClient(MONGODB_URL)
    db = db_client.customer_service
    
    # Create indexes using thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, db.tickets.create_index, "created_at")
    await loop.run_in_executor(executor, db.tickets.create_index, "status")
    await loop.run_in_executor(executor, db.tickets.create_index, "category")
    await loop.run_in_executor(executor, db.tickets.create_index, "priority")
    logger.info("Database initialized")

async def startup_qdrant():
    global qdrant_client
    
    # Initialize Qdrant client with API key for cloud hosting
    if QDRANT_API_KEY:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        logger.info("Connected to cloud Qdrant with API key")
    else:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        logger.info("Connected to local Qdrant")
    
    # Create collections if they don't exist
    collections = ["documentation", "service_memory"]
    for collection in collections:
        try:
            # Check if collection exists first
            existing_collections = qdrant_client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]
            
            if collection not in collection_names:
                qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {collection}")
            else:
                logger.info(f"Collection {collection} already exists")
        except Exception as e:
            logger.error(f"Error with collection {collection}: {e}")
            # Continue with other collections even if one fails

async def startup_embedding():
    global embedding_model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded")

# Dependency injection
async def get_db():
    return db

async def get_qdrant():
    return qdrant_client

async def get_embedding_model():
    return embedding_model

# AI Service Classes
class SentimentAnalyzer:
    @staticmethod
    async def analyze_sentiment(text: str) -> Sentiment:
        prompt = f"""
        Analyze the sentiment of this customer service message and classify it as one of:
        - positive: Customer is happy, satisfied, or expressing gratitude
        - neutral: Customer is asking questions or providing information without emotion
        - negative: Customer is disappointed, unsatisfied, or mildly upset
        - frustrated: Customer is angry, very upset, or expressing strong negative emotions
        
        Message: "{text}"
        
        Respond with only the sentiment classification (positive, neutral, negative, or frustrated).
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
                )
                result = response.json()
                sentiment_text = result["response"].strip().lower()
                
                if "frustrated" in sentiment_text:
                    return Sentiment.FRUSTRATED
                elif "negative" in sentiment_text:
                    return Sentiment.NEGATIVE
                elif "positive" in sentiment_text:
                    return Sentiment.POSITIVE
                else:
                    return Sentiment.NEUTRAL
                    
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return Sentiment.NEUTRAL

class TicketClassifier:
    @staticmethod
    async def classify_ticket(title: str, description: str) -> tuple[TicketCategory, Priority, AgentCategory]:
        prompt = f"""
        Classify this customer service ticket:
        
        Title: "{title}"
        Description: "{description}"
        
        Provide classification in this exact format:
        Category: [billing/technical/internet/account/general]
        Priority: [low/medium/high/urgent]
        Agent: [billing_specialist/tech_support/network_specialist/general_support/senior_support]
        
        Guidelines:
        - billing: Payment issues, refunds, charges, invoices
        - technical: Software problems, device issues, troubleshooting
        - internet: Connection problems, speed issues, outages
        - account: Login issues, profile changes, account settings
        - general: General inquiries, feedback, other issues
        
        Priority based on urgency and impact:
        - urgent: Service completely down, security issues, angry customers
        - high: Major functionality issues, multiple users affected
        - medium: Single user issues, minor functionality problems
        - low: Questions, feature requests, minor issues
        
        Agent assignment based on expertise needed.
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
                )
                result = response.json()
                classification = result["response"].strip()
                
                # Parse response
                category = TicketCategory.GENERAL
                priority = Priority.MEDIUM
                agent_category = AgentCategory.GENERAL_SUPPORT
                
                lines = classification.split('\n')
                for line in lines:
                    line = line.strip().lower()
                    if line.startswith('category:'):
                        cat_text = line.split(':')[1].strip()
                        if cat_text in [e.value for e in TicketCategory]:
                            category = TicketCategory(cat_text)
                    elif line.startswith('priority:'):
                        pri_text = line.split(':')[1].strip()
                        if pri_text in [e.value for e in Priority]:
                            priority = Priority(pri_text)
                    elif line.startswith('agent:'):
                        agent_text = line.split(':')[1].strip()
                        if agent_text in [e.value for e in AgentCategory]:
                            agent_category = AgentCategory(agent_text)
                
                return category, priority, agent_category
                
        except Exception as e:
            logger.error(f"Ticket classification error: {e}")
            return TicketCategory.GENERAL, Priority.MEDIUM, AgentCategory.GENERAL_SUPPORT

class KnowledgeSearcher:
    @staticmethod
    async def search_documentation(query: str, limit: int = 5) -> List[Dict]:
        try:
            query_vector = embedding_model.encode(query).tolist()
            
            results = qdrant_client.search(
                collection_name="documentation",
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.7,
                with_payload=True,
                with_vectors=False
            )
            
            return [{
                "content": point.payload.get("content", ""),
                "title": point.payload.get("title", ""),
                "category": point.payload.get("category", ""),
                "score": point.score
            } for point in results]
            
        except Exception as e:
            logger.error(f"Documentation search error: {e}")
            return []
    
    @staticmethod
    async def search_service_memory(query: str, limit: int = 5) -> List[Dict]:
        try:
            query_vector = embedding_model.encode(query).tolist()
            
            results = qdrant_client.search(
                collection_name="service_memory",
                query_vector=query_vector,
                limit=limit,
                score_threshold=0.75,
                with_payload=True,
                with_vectors=False
            )
            
            return [{
                "query": point.payload.get("query", ""),
                "resolution": point.payload.get("resolution", ""),
                "category": point.payload.get("category", ""),
                "agent_name": point.payload.get("agent_name", ""),
                "score": point.score
            } for point in results]
            
        except Exception as e:
            logger.error(f"Service memory search error: {e}")
            return []

class ResolutionEvaluator:
    @staticmethod
    async def evaluate_resolution(ticket_description: str, proposed_resolution: str) -> tuple[bool, float]:
        prompt = f"""
        Evaluate if this proposed resolution adequately addresses the customer's issue:
        
        Customer Issue: "{ticket_description}"
        Proposed Resolution: "{proposed_resolution}"
        
        Consider:
        1. Does the resolution directly address the customer's problem?
        2. Is the resolution clear and actionable?
        3. Would this likely satisfy the customer?
        4. Is the resolution appropriate for the issue complexity?
        
        Respond in this exact format:
        Decision: [ACCEPT/REJECT]
        Confidence: [0.0-1.0]
        Reasoning: [brief explanation]
        """
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
                )
                result = response.json()
                evaluation = result["response"].strip()
                
                # Parse response
                decision = False
                confidence = 0.5
                
                lines = evaluation.split('\n')
                for line in lines:
                    line = line.strip().lower()
                    if line.startswith('decision:'):
                        dec_text = line.split(':')[1].strip()
                        decision = dec_text == 'accept'
                    elif line.startswith('confidence:'):
                        conf_text = line.split(':')[1].strip()
                        try:
                            confidence = float(conf_text)
                        except ValueError:
                            confidence = 0.5
                
                return decision, confidence
                
        except Exception as e:
            logger.error(f"Resolution evaluation error: {e}")
            return False, 0.0

async def add_to_service_memory(ticket: dict, resolution: str, agent: str, qdrant, embedding_model):
    """Add successful manual resolution to service memory"""
    try:
        query = f"{ticket['title']} {ticket['description']}"
        embedding = embedding_model.encode(query).tolist()
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "query": query,
                "resolution": resolution,
                "category": ticket['category'],
                "agent_name": agent,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        qdrant.upsert(
            collection_name="service_memory", 
            points=[point],
            wait=True
        )
        logger.info(f"Added manual resolution to service memory for ticket {ticket['_id']}")
    except Exception as e:
        logger.error(f"Error adding to service memory: {e}")

# Core ticket processing
async def process_ticket(ticket_data: TicketCreate, db) -> TicketResponse:
    # Step 1: Analyze sentiment
    sentiment = await SentimentAnalyzer.analyze_sentiment(
        f"{ticket_data.title} {ticket_data.description}"
    )
    
    # Step 2: Classify ticket
    category, priority, agent_category = await TicketClassifier.classify_ticket(
        ticket_data.title, ticket_data.description
    )
    
    # Adjust priority based on sentiment
    if sentiment == Sentiment.FRUSTRATED and priority in [Priority.LOW, Priority.MEDIUM]:
        priority = Priority.HIGH
    
    # Step 3: Search knowledge bases
    search_query = f"{ticket_data.title} {ticket_data.description}"
    
    # Search documentation
    doc_results = await KnowledgeSearcher.search_documentation(search_query)
    
    # Search service memory
    memory_results = await KnowledgeSearcher.search_service_memory(search_query)
    
    # Step 4: Evaluate potential resolutions
    ai_resolution = None
    confidence_score = 0.0
    status = TicketStatus.PENDING
    
    # Try service memory first (previous successful resolutions)
    if memory_results:
        best_memory = memory_results[0]
        is_good, confidence = await ResolutionEvaluator.evaluate_resolution(
            ticket_data.description, best_memory["resolution"]
        )
        
        if is_good and confidence > 0.7:
            ai_resolution = best_memory["resolution"]
            confidence_score = confidence
            status = TicketStatus.RESOLVED
    
    # If no good memory result, try documentation
    if not ai_resolution and doc_results:
        best_doc = doc_results[0]
        is_good, confidence = await ResolutionEvaluator.evaluate_resolution(
            ticket_data.description, best_doc["content"]
        )
        
        if is_good and confidence > 0.6:
            ai_resolution = best_doc["content"]
            confidence_score = confidence
            status = TicketStatus.RESOLVED
    
    # If no AI resolution found, escalate to human
    if not ai_resolution:
        status = TicketStatus.ESCALATED
    
    # Create ticket document
    ticket_doc = {
        "_id": str(uuid.uuid4()),
        "title": ticket_data.title,
        "description": ticket_data.description,
        "customer_email": ticket_data.customer_email,
        "customer_name": ticket_data.customer_name,
        "category": category.value,
        "priority": priority.value,
        "agent_category": agent_category.value,
        "status": status.value,
        "sentiment": sentiment.value,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "ai_resolution": ai_resolution,
        "manual_resolution": None,  # New field
        "resolution_type": ResolutionType.AI.value if ai_resolution else None,  # New field
        "confidence_score": confidence_score,
        "assigned_agent": None,
        "resolution_feedback": None,  # New field
        "notes": []  # New field for storing notes
    }
    
    # Save to database
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, db.tickets.insert_one, ticket_doc)
    
    return TicketResponse(**ticket_doc, id=ticket_doc["_id"])

# API Endpoints
@app.post("/tickets", response_model=TicketResponse)
async def create_ticket(
    ticket: TicketCreate,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """Create a new customer service ticket"""
    try:
        result = await process_ticket(ticket, db)
        
        # Log analytics in background
        background_tasks.add_task(log_ticket_analytics, result)
        
        return result
    except Exception as e:
        logger.error(f"Error creating ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tickets", response_model=List[TicketResponse])
async def get_tickets(
    status: Optional[TicketStatus] = None,
    category: Optional[TicketCategory] = None,
    priority: Optional[Priority] = None,
    limit: int = 50,
    skip: int = 0,
    db = Depends(get_db)
):
    """Get tickets with optional filtering"""
    try:
        filter_query = {}
        if status:
            filter_query["status"] = status.value
        if category:
            filter_query["category"] = category.value
        if priority:
            filter_query["priority"] = priority.value
        
        def get_tickets_sync():
            cursor = db.tickets.find(filter_query).sort("created_at", -1).skip(skip).limit(limit)
            return list(cursor)
        
        loop = asyncio.get_event_loop()
        tickets = await loop.run_in_executor(executor, get_tickets_sync)
        
        return [TicketResponse(**ticket, id=ticket["_id"]) for ticket in tickets]
    except Exception as e:
        logger.error(f"Error getting tickets: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str, db = Depends(get_db)):
    """Get a specific ticket"""
    try:
        loop = asyncio.get_event_loop()
        ticket = await loop.run_in_executor(executor, db.tickets.find_one, {"_id": ticket_id})
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        return TicketResponse(**ticket, id=ticket["_id"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# New endpoint for manual resolution
@app.put("/tickets/{ticket_id}/resolve-manual", response_model=TicketResponse)
async def resolve_ticket_manually(
    ticket_id: str,
    resolution: ManualResolution,
    background_tasks: BackgroundTasks,
    db = Depends(get_db),
    qdrant = Depends(get_qdrant),
    embedding_model = Depends(get_embedding_model)
):
    """Resolve a ticket manually and optionally add to service memory"""
    try:
        update_data = {
            "manual_resolution": resolution.resolution,
            "resolution_type": ResolutionType.MANUAL.value,
            "status": TicketStatus.RESOLVED.value,
            "assigned_agent": resolution.agent,
            "updated_at": datetime.utcnow()
        }
        
        if resolution.add_to_service_memory:
            background_tasks.add_task(
                add_to_service_memory, 
                ticket_id, 
                resolution.resolution, 
                resolution.agent,
                qdrant,
                embedding_model
            )
        
        def update_ticket():
            return db.tickets.update_one(
                {"_id": ticket_id},
                {"$set": update_data}
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, update_ticket)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Return updated ticket
        ticket = await get_ticket(ticket_id, db)
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving ticket manually: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# New endpoint for adding notes
@app.post("/tickets/{ticket_id}/notes", response_model=TicketResponse)
async def add_ticket_note(
    ticket_id: str,
    note: TicketNote,
    db = Depends(get_db)
):
    """Add a note to a ticket"""
    try:
        new_note = {
            "id": str(uuid.uuid4()),
            "content": note.content,
            "agent": note.agent,
            "note_type": note.note_type.value,
            "created_at": datetime.utcnow()
        }
        
        def update_ticket():
            return db.tickets.update_one(
                {"_id": ticket_id},
                {
                    "$push": {"notes": new_note},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, update_ticket)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Return updated ticket
        ticket = await get_ticket(ticket_id, db)
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding note to ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# New endpoint for updating status
@app.put("/tickets/{ticket_id}/status", response_model=TicketResponse)
async def update_ticket_status(
    ticket_id: str,
    status: TicketStatus,
    db = Depends(get_db)
):
    """Update ticket status"""
    try:
        def update_ticket():
            return db.tickets.update_one(
                {"_id": ticket_id},
                {
                    "$set": {
                        "status": status.value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, update_ticket)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Return updated ticket
        ticket = await get_ticket(ticket_id, db)
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ticket status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# New endpoint for resolution feedback
@app.post("/tickets/{ticket_id}/feedback", response_model=TicketResponse)
async def add_resolution_feedback(
    ticket_id: str,
    feedback: ResolutionFeedback,
    db = Depends(get_db)
):
    """Add resolution feedback from customer"""
    try:
        def update_ticket():
            return db.tickets.update_one(
                {"_id": ticket_id},
                {
                    "$set": {
                        "resolution_feedback": feedback.rating,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, update_ticket)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Return updated ticket
        ticket = await get_ticket(ticket_id, db)
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding resolution feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Updated agent assignment endpoint
@app.put("/tickets/{ticket_id}/assign", response_model=TicketResponse)
async def assign_ticket(
    ticket_id: str,
    assignment: TicketAssignment,  # Changed to use Pydantic model
    db = Depends(get_db)
):
    """Assign a ticket to an agent"""
    try:
        def update_ticket():
            return db.tickets.update_one(
                {"_id": ticket_id},
                {
                    "$set": {
                        "assigned_agent": assignment.agent_name,
                        "status": TicketStatus.IN_PROGRESS.value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, update_ticket)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Return updated ticket
        ticket = await get_ticket(ticket_id, db)
        return ticket
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/knowledge/documentation")
async def add_documentation(
    entry: KnowledgeEntry,
    qdrant = Depends(get_qdrant),
    embedding_model = Depends(get_embedding_model)
):
    """Add documentation to knowledge base"""
    try:
        # Generate embedding
        embedding = embedding_model.encode(entry.content).tolist()
        
        # Store in Qdrant
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "title": entry.title,
                "content": entry.content,
                "category": entry.category.value,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        qdrant.upsert(
            collection_name="documentation", 
            points=[point],
            wait=True
        )
        
        return {"message": "Documentation added successfully"}
    except Exception as e:
        logger.error(f"Error adding documentation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/knowledge/service-memory")
async def add_service_memory(
    entry: ServiceMemoryEntry,
    qdrant = Depends(get_qdrant),
    embedding_model = Depends(get_embedding_model)
):
    """Add successful resolution to service memory"""
    try:
        # Generate embedding for the query
        embedding = embedding_model.encode(entry.query).tolist()
        
        # Store in Qdrant
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "query": entry.query,
                "resolution": entry.resolution,
                "category": entry.category.value,
                "agent_name": entry.agent_name,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        qdrant.upsert(
            collection_name="service_memory", 
            points=[point],
            wait=True
        )
        
        return {"message": "Service memory added successfully"}
    except Exception as e:
        logger.error(f"Error adding service memory: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = 30,
    db = Depends(get_db)
):
    """Get analytics and insights"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        def get_analytics_data():
            # Total tickets
            total_tickets = db.tickets.count_documents({"created_at": {"$gte": start_date}})
            
            # Resolved by AI
            resolved_by_ai = db.tickets.count_documents({
                "created_at": {"$gte": start_date},
                "status": TicketStatus.RESOLVED.value,
                "ai_resolution": {"$exists": True, "$ne": None}
            })
            
            # Escalated to human
            escalated_to_human = db.tickets.count_documents({
                "created_at": {"$gte": start_date},
                "status": TicketStatus.ESCALATED.value
            })
            
            # Sentiment distribution
            sentiment_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
            ]
            sentiment_results = list(db.tickets.aggregate(sentiment_pipeline))
            
            # Category distribution
            category_pipeline = [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {"_id": "$category", "count": {"$sum": 1}}}
            ]
            category_results = list(db.tickets.aggregate(category_pipeline))
            
            # New metrics
            manual_resolutions = db.tickets.count_documents({
                "created_at": {"$gte": start_date},
                "resolution_type": ResolutionType.MANUAL.value
            })
            
            hybrid_resolutions = db.tickets.count_documents({
                "created_at": {"$gte": start_date},
                "resolution_type": ResolutionType.HYBRID.value
            })
            
            # Calculate average feedback score
            feedback_pipeline = [
                {"$match": {
                    "created_at": {"$gte": start_date},
                    "resolution_feedback": {"$exists": True}
                }},
                {"$group": {
                    "_id": None,
                    "avg_feedback": {"$avg": "$resolution_feedback"},
                    "count": {"$sum": 1}
                }}
            ]
            feedback_result = list(db.tickets.aggregate(feedback_pipeline))
            avg_feedback = feedback_result[0]["avg_feedback"] if feedback_result else 0

            return {
                "total_tickets": total_tickets,
                "resolved_by_ai": resolved_by_ai,
                "escalated_to_human": escalated_to_human,
                "sentiment_results": sentiment_results,
                "category_results": category_results,
                "manual_resolutions": manual_resolutions,
                "hybrid_resolutions": hybrid_resolutions,
                "avg_feedback": avg_feedback
            }
        
        loop = asyncio.get_event_loop()
        analytics_data = await loop.run_in_executor(executor, get_analytics_data)
        
        # Process results
        sentiment_distribution = {item["_id"]: item["count"] for item in analytics_data["sentiment_results"]}
        category_distribution = {item["_id"]: item["count"] for item in analytics_data["category_results"]}
        
        # Average resolution time (simplified)
        avg_resolution_time = 2.5  # hours - would need more complex calculation
        
        # Top issues (simplified)
        top_issues = [
            {"issue": "Internet connectivity", "count": 25, "resolution_rate": 0.8},
            {"issue": "Billing inquiries", "count": 18, "resolution_rate": 0.95},
            {"issue": "Account access", "count": 12, "resolution_rate": 0.7}
        ]
        
        return AnalyticsResponse(
            total_tickets=analytics_data["total_tickets"],
            resolved_by_ai=analytics_data["resolved_by_ai"],
            escalated_to_human=analytics_data["escalated_to_human"],
            avg_resolution_time=avg_resolution_time,
            manual_resolutions=analytics_data["manual_resolutions"],
            hybrid_resolutions=analytics_data["hybrid_resolutions"],
            avg_feedback=analytics_data["avg_feedback"],
            sentiment_distribution=sentiment_distribution,
            category_distribution=category_distribution,
            top_issues=top_issues
        )
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Background tasks
async def log_ticket_analytics(ticket: TicketResponse):
    """Log ticket creation for analytics"""
    logger.info(f"Ticket {ticket.id} created: {ticket.category}, {ticket.priority}, {ticket.status}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)