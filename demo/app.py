
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json
import re
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pymongo
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from bson import ObjectId
import uvicorn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Union


# Add Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Ticketing System", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
MONGODB_URL = "mongodb://datamaster:B8znzNgx2559BzWF1EJw@localhost:27017/"
DATABASE_NAME = "ai_ticketing"

# Qdrant configuration
QDRANT_URL = os.environ["qdranturl"]
QDRANT_API_KEY = os.environ["qdrantkey"]  # Set if using cloud or auth
TICKET_COLLECTION = "tickets"
KNOWLEDGE_COLLECTION = "knowledge_base"

# Initialize MongoDB client
client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Initialize Qdrant client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create Qdrant collections if they don't exist
EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2 vector size

try:
    qdrant.get_collection(TICKET_COLLECTION)
except Exception:
    qdrant.create_collection(
        collection_name=TICKET_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_SIZE,
            distance=models.Distance.COSINE
        )
    )
    logger.info(f"Created Qdrant collection: {TICKET_COLLECTION}")

try:
    qdrant.get_collection(KNOWLEDGE_COLLECTION)
except Exception:
    qdrant.create_collection(
        collection_name=KNOWLEDGE_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_SIZE,
            distance=models.Distance.COSINE
        )
    )
    logger.info(f"Created Qdrant collection: {KNOWLEDGE_COLLECTION}")

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=8)  # Increased workers for better concurrency

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Enums
class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketCategory(str, Enum):
    NETWORK = "network"
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SERVICE = "service"
    GENERAL = "general"

class SentimentLevel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    URGENT = "urgent"

# Pydantic models
class CustomerInfo(BaseModel):
    customer_id: str
    name: str
    email: str
    phone: str
    location: Optional[str] = None
    service_area: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude coordinate (-90 to 90)")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude coordinate (-180 to 180)")

class TicketCreate(BaseModel):
    title: str
    description: str
    customer: CustomerInfo
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None

class TicketResponse(BaseModel):
    id: str
    title: str
    description: str
    customer: CustomerInfo
    category: TicketCategory
    priority: TicketPriority
    status: TicketStatus
    created_at: datetime
    updated_at: datetime
    resolution: Optional[str] = None
    agent_assigned: Optional[str] = None
    confidence_score: Optional[float] = None
    auto_resolved: bool = False
    sentiment: Optional[SentimentLevel] = None
    summary: Optional[str] = None
    related_tickets: Optional[List[str]] = None

class AutoResolutionResult(BaseModel):
    can_resolve: bool
    confidence: float
    suggested_response: str
    category: TicketCategory
    priority: TicketPriority
    sentiment: SentimentLevel

class KnowledgeBaseArticle(BaseModel):
    title: str
    keywords: List[str]
    solution: str
    category: TicketCategory
    priority: TicketPriority
    embedding: Optional[List[float]] = None

class SentimentAnalysisResult(BaseModel):
    sentiment: SentimentLevel
    confidence: float
    key_phrases: List[str]

class TicketCluster(BaseModel):
    cluster_id: int
    representative_ticket_id: str
    representative_title: str
    count: int
    category: TicketCategory
    priority: TicketPriority

# Initialize knowledge base collection
knowledge_collection = db.knowledge_base

# Initialize tickets collection
tickets_collection = db.tickets
tickets_collection.create_index("created_at")
tickets_collection.create_index("status")

# Preload knowledge base
def initialize_knowledge_base():
    knowledge_base = [
        KnowledgeBaseArticle(
            title="Internet Down",
            keywords=["no internet", "connection down", "can't connect", "wifi not working"],
            solution="Please try restarting your modem by unplugging it for 30 seconds, then plugging it back in. If the issue persists, we'll dispatch a technician. Your service area shows no widespread outages.",
            category=TicketCategory.NETWORK,
            priority=TicketPriority.HIGH
        ),
        KnowledgeBaseArticle(
            title="Slow Internet",
            keywords=["slow internet", "slow connection", "buffering", "slow speed"],
            solution="We can see your connection is active. Try restarting your router and running a speed test. If speeds are below your plan, we'll schedule a technician visit to check your line.",
            category=TicketCategory.NETWORK,
            priority=TicketPriority.MEDIUM
        ),
        KnowledgeBaseArticle(
            title="High Bill",
            keywords=["high bill", "expensive", "charges", "overcharge"],
            solution="I can see your recent usage patterns. The increase appears to be due to additional services activated last month. I'll email you a detailed breakdown and can set up a payment plan if needed.",
            category=TicketCategory.BILLING,
            priority=TicketPriority.MEDIUM
        ),
        KnowledgeBaseArticle(
            title="Payment Issue",
            keywords=["payment failed", "can't pay", "payment problem", "autopay"],
            solution="I can help you update your payment method. Please log into your account or I can process a payment over the phone. Your service will remain active while we resolve this.",
            category=TicketCategory.BILLING,
            priority=TicketPriority.HIGH
        ),
        KnowledgeBaseArticle(
            title="Email Setup",
            keywords=["email not working", "email setup", "can't send email", "email config"],
            solution="I'll send you the current email server settings. For most devices: IMAP server: mail.yourprovider.com, Port: 993, SSL enabled. Would you like me to walk you through the setup?",
            category=TicketCategory.TECHNICAL,
            priority=TicketPriority.LOW
        )
    ]
    
    # Generate embeddings and insert into Qdrant
    for article in knowledge_base:
        embedding = embedding_model.encode(article.title + " " + " ".join(article.keywords)).tolist()
        article_dict = article.dict()
        
        # Store in MongoDB without embedding
        result = knowledge_collection.update_one(
            {"title": article.title},
            {"$set": article_dict},
            upsert=True
        )
        
        # Get MongoDB ID
        doc = knowledge_collection.find_one({"title": article.title})
        doc_id = str(doc["_id"])
        
        # Store embedding in Qdrant
        qdrant.upsert(
            collection_name=KNOWLEDGE_COLLECTION,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "title": article.title,
                        "category": article.category.value,
                        "priority": article.priority.value
                    }
                )
            ]
        )
    logger.info("Knowledge base initialized with sample articles")

# Initialize on startup
asyncio.get_event_loop().run_in_executor(executor, initialize_knowledge_base)

class TicketService:
    def __init__(self, db):
        self.db = db
        self.tickets_collection = db.tickets
        self.knowledge_collection = db.knowledge_base
        self.executor = executor

    async def create_ticket(self, ticket_data: TicketCreate) -> TicketResponse:
        """Create a new ticket with enhanced AI features"""
        # Generate embedding for the ticket
        embedding = await self._generate_embedding(ticket_data.description)
        
        # Analyze sentiment
        sentiment_result = await self._analyze_sentiment(ticket_data.description)
        
        # Attempt auto-resolution
        auto_resolution = await self._attempt_auto_resolution(ticket_data, embedding)
        
        # Generate summary
        summary = await self._generate_summary(ticket_data.description)
        
        # Find similar tickets
        related_tickets = await self._find_related_tickets(embedding)
        
        # Create ticket document (without embedding)
        ticket_doc = {
            "title": ticket_data.title,
            "description": ticket_data.description,
            "customer": ticket_data.customer.dict(),
            "category": auto_resolution.category,
            "priority": auto_resolution.priority,
            "status": TicketStatus.RESOLVED if auto_resolution.can_resolve else TicketStatus.OPEN,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "resolution": auto_resolution.suggested_response if auto_resolution.can_resolve else None,
            "agent_assigned": None if auto_resolution.can_resolve else await self._assign_agent(auto_resolution.category),
            "confidence_score": auto_resolution.confidence,
            "auto_resolved": auto_resolution.can_resolve,
            "sentiment": sentiment_result.sentiment.value,
            "summary": summary,
            "related_tickets": []
        }
        
        # Insert ticket into MongoDB
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.tickets_collection.insert_one(ticket_doc)
        )
        ticket_id = str(result.inserted_id)
        ticket_doc["_id"] = result.inserted_id
        
        # Store embedding in Qdrant
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: qdrant.upsert(
                collection_name=TICKET_COLLECTION,
                points=[
                    models.PointStruct(
                        id=ticket_id,
                        vector=embedding,
                        payload={
                            "status": ticket_doc["status"],
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
        )
        
        # Find and store related tickets
        related_tickets = await self._find_related_tickets(embedding, exclude_id=ticket_id)
        self.tickets_collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {"related_tickets": related_tickets}}
        )
        ticket_doc["related_tickets"] = related_tickets
        
        # Log creation
        logger.info(f"Created ticket {result.inserted_id} - Auto-resolved: {auto_resolution.can_resolve} - Sentiment: {sentiment_result.sentiment}")
        
        # Schedule escalation check if not resolved
        if not auto_resolution.can_resolve and auto_resolution.priority in [TicketPriority.HIGH, TicketPriority.CRITICAL]:
            asyncio.create_task(self._check_for_escalation(str(result.inserted_id)))
        
        return self._format_ticket_response(ticket_doc)

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, 
            lambda: embedding_model.encode(text).tolist()
        )

    async def _attempt_auto_resolution(self, ticket_data: TicketCreate, embedding: List[float]) -> AutoResolutionResult:
        """Enhanced auto-resolution with semantic search and LLM fallback"""
        # 1. Semantic search in knowledge base
        kb_result = await self._semantic_search_knowledge_base(embedding)
        if kb_result and kb_result["confidence"] > 0.75:
            return AutoResolutionResult(
                can_resolve=True,
                confidence=kb_result["confidence"],
                suggested_response=kb_result["solution"],
                category=kb_result["category"],
                priority=kb_result["priority"],
                sentiment=SentimentLevel.NEUTRAL  # Will be updated later
            )
        
        # 2. Use LLM for advanced analysis
        return await self._analyze_with_llm(ticket_data)

    async def _semantic_search_knowledge_base(self, embedding: List[float]) -> Optional[Dict]:
        """Semantic search in knowledge base using Qdrant"""
        try:
            search_results = qdrant.search(
                collection_name=KNOWLEDGE_COLLECTION,
                query_vector=embedding,
                limit=1,
                with_payload=True,
                score_threshold=0.7  # Minimum similarity score
            )
            
            if search_results:
                best_match = search_results[0]
                # Get full document from MongoDB
                kb_doc = self.knowledge_collection.find_one(
                    {"_id": ObjectId(best_match.id)}
                )
                
                if kb_doc:
                    return {
                        "confidence": best_match.score,
                        "solution": kb_doc["solution"],
                        "category": kb_doc["category"],
                        "priority": kb_doc["priority"]
                    }
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        return None

    async def _find_related_tickets(self, embedding: List[float], exclude_id: str = None, limit: int = 3) -> List[str]:
        """Find semantically similar tickets using Qdrant"""
        try:
            # Search Qdrant for similar tickets
            search_results = qdrant.search(
                collection_name=TICKET_COLLECTION,
                query_vector=embedding,
                limit=limit + 1,  # +1 to account for possible self-match
                with_payload=False,
                score_threshold=0.6
            )
            
            # Filter out current ticket and low confidence matches
            related_ids = []
            for result in search_results:
                if result.id != exclude_id and result.score > 0.6:
                    related_ids.append(result.id)
                    if len(related_ids) >= limit:
                        break
            
            return related_ids
        except Exception as e:
            logger.error(f"Related tickets search failed: {e}")
            return []

    async def _analyze_with_llm(self, ticket_data: TicketCreate) -> AutoResolutionResult:
        """Enhanced LLM analysis with sentiment and context awareness"""
        prompt = f"""
        Analyze this customer service ticket and provide a JSON response:
        
        **Customer**: {ticket_data.customer.name}
        **Location**: {ticket_data.customer.location}
        **Issue**: {ticket_data.title}
        **Description**: {ticket_data.description}
        
        Determine:
        1. Can this be automatically resolved? (true/false)
        2. Confidence level (0.0-1.0)
        3. Suggested response to customer
        4. Category: {', '.join([c.value for c in TicketCategory])}
        5. Priority: {', '.join([p.value for p in TicketPriority])}
        6. Sentiment: {', '.join([s.value for s in SentimentLevel])}
        
        Consider:
        - Urgency indicators and emotional tone
        - Technical complexity
        - Need for human intervention
        - Potential business impact
        
        Response format:
        {{
            "can_resolve": boolean,
            "confidence": float,
            "suggested_response": "string",
            "category": "string",
            "priority": "string",
            "sentiment": "string"
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ollama.chat(
                    model='llama3.1', 
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.3}
                )
            )
            
            # Parse LLM response
            response_content = response['message']['content']
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                
                # Validate and clamp confidence
                confidence = analysis.get("confidence", 0.0)
                confidence = max(0.0, min(confidence, 1.0))
                
                return AutoResolutionResult(
                    can_resolve=analysis.get("can_resolve", False),
                    confidence=confidence,
                    suggested_response=analysis.get("suggested_response", "We'll review your issue and get back to you."),
                    category=analysis.get("category", TicketCategory.GENERAL.value),
                    priority=analysis.get("priority", TicketPriority.MEDIUM.value),
                    sentiment=analysis.get("sentiment", SentimentLevel.NEUTRAL.value)
                )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
        
        # Fallback response
        return AutoResolutionResult(
            can_resolve=False,
            confidence=0.0,
            suggested_response="Thank you for contacting us. We'll review your issue and get back to you shortly.",
            category=TicketCategory.GENERAL.value,
            priority=TicketPriority.MEDIUM.value,
            sentiment=SentimentLevel.NEUTRAL.value
        )

    async def _analyze_sentiment(self, text: str) -> SentimentAnalysisResult:
        """Analyze sentiment using LLM"""
        prompt = f"""
        Analyze the sentiment of this customer message and provide a JSON response:
        
        **Message**: {text}
        
        Determine:
        1. Sentiment level: {', '.join([s.value for s in SentimentLevel])}
        2. Confidence score (0.0-1.0)
        3. Key emotional phrases (list of top 3 phrases)
        
        Response format:
        {{
            "sentiment": "string",
            "confidence": float,
            "key_phrases": ["phrase1", "phrase2", "phrase3"]
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ollama.chat(
                    model='llama3.1', 
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.2}
                )
            )
            
            response_content = response['message']['content']
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                sentiment_data = json.loads(json_str)
                
                # Validate and clamp confidence
                confidence = sentiment_data.get("confidence", 0.5)
                confidence = max(0.0, min(confidence, 1.0))
                
                return SentimentAnalysisResult(
                    sentiment=sentiment_data.get("sentiment", SentimentLevel.NEUTRAL.value),
                    confidence=confidence,
                    key_phrases=sentiment_data.get("key_phrases", [])
                )
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
        
        return SentimentAnalysisResult(
            sentiment=SentimentLevel.NEUTRAL.value,
            confidence=0.5,
            key_phrases=[]
        )

    async def _generate_summary(self, description: str) -> str:
        """Generate a concise summary of the ticket description"""
        if len(description) < 150:  # No need to summarize short descriptions
            return description
        
        prompt = f"Summarize this customer support ticket description in 1-2 sentences:\n\n{description}"
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: ollama.chat(
                    model='llama3.1', 
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.1}
                )
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return description[:200] + "..." if len(description) > 200 else description

    async def _find_related_tickets(self, embedding: List[float], limit: int = 3) -> List[str]:
        """Find semantically similar tickets"""
        try:
            query = [
                {
                    "$vectorSearch": {
                        "index": "embedding_index",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": 100,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": {"$toString": "$_id"},
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: list(self.tickets_collection.aggregate(query))
            )
            
            return [res["_id"] for res in results if res["score"] > 0.6]
        except Exception as e:
            logger.error(f"Related tickets search failed: {e}")
            return []

    async def _assign_agent(self, category: str) -> str:
        """Assign appropriate agent based on category and workload"""
        # Basic implementation - in production, integrate with agent availability system
        agent_assignments = {
            "network": "Network Specialist",
            "billing": "Billing Specialist", 
            "technical": "Technical Support",
            "account": "Account Manager",
            "service": "Customer Service",
            "general": "Support Agent"
        }
        return agent_assignments.get(category, "Support Agent")

    async def _check_for_escalation(self, ticket_id: str):
        """Check if ticket needs escalation after delay"""
        await asyncio.sleep(3600)  # Check after 1 hour
        
        ticket = await self.get_ticket_by_id(ticket_id)
        if ticket and ticket.status == TicketStatus.OPEN and ticket.priority in [TicketPriority.HIGH, TicketPriority.CRITICAL]:
            await self.update_ticket_status(ticket_id, TicketStatus.ESCALATED)
            logger.info(f"Ticket {ticket_id} escalated due to inactivity")

    async def get_tickets(self, status: Optional[TicketStatus] = None, limit: int = 50) -> List[TicketResponse]:
        """Get tickets with optional status filter"""
        
        query = {}
        if status:
            query["status"] = status
        
        tickets = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: list(self.tickets_collection.find(query).limit(limit).sort("created_at", -1))
        )
        
        return [self._format_ticket_response(ticket_doc) for ticket_doc in tickets]

    async def get_ticket_by_id(self, ticket_id: str) -> Optional[TicketResponse]:
        """Get a specific ticket by ID"""
        
        try:
            ticket_doc = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.tickets_collection.find_one({"_id": ObjectId(ticket_id)})
            )
            if ticket_doc:
                return self._format_ticket_response(ticket_doc)
        except Exception as e:
            logger.error(f"Error fetching ticket {ticket_id}: {e}")
        
        return None

    async def update_ticket_status(self, ticket_id: str, status: TicketStatus) -> Optional[TicketResponse]:
        """Update ticket status"""
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.tickets_collection.update_one(
                    {"_id": ObjectId(ticket_id)},
                    {"$set": {"status": status, "updated_at": datetime.utcnow()}}
                )
            )
            
            if result.modified_count > 0:
                return await self.get_ticket_by_id(ticket_id)
        except Exception as e:
            logger.error(f"Error updating ticket {ticket_id}: {e}")
        
        return None

    async def get_analytics(self) -> Dict[str, Any]:
        """Get system analytics"""
        
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_tickets": {"$sum": 1},
                    "auto_resolved": {"$sum": {"$cond": ["$auto_resolved", 1, 0]}},
                    "avg_confidence": {"$avg": "$confidence_score"}
                }
            }
        ]
        
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: list(self.tickets_collection.aggregate(pipeline))
        )
        
        if result:
            stats = result[0]
            automation_rate = (stats["auto_resolved"] / stats["total_tickets"]) * 100 if stats["total_tickets"] > 0 else 0
            
            return {
                "total_tickets": stats["total_tickets"],
                "auto_resolved": stats["auto_resolved"],
                "automation_rate": round(automation_rate, 2),
                "average_confidence": round(stats["avg_confidence"] or 0, 2)
            }
        
        return {
            "total_tickets": 0,
            "auto_resolved": 0,
            "automation_rate": 0,
            "average_confidence": 0
        }

    async def cluster_tickets(self, n_clusters: int = 5) -> List[TicketCluster]:
        """Cluster open tickets to identify common issues"""
        try:
            # Get embeddings for open tickets
            open_tickets = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: list(self.tickets_collection.find(
                    {"status": TicketStatus.OPEN.value}, 
                    {"embedding": 1, "title": 1, "category": 1, "priority": 1}
                ))
            )
            
            if len(open_tickets) < 2:
                return []
                
            embeddings = [t["embedding"] for t in open_tickets]
            ticket_ids = [str(t["_id"]) for t in open_tickets]
            titles = [t["title"] for t in open_tickets]
            
            # Perform K-means clustering
            actual_clusters = min(n_clusters, len(open_tickets))
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
            kmeans.fit(embeddings)
            
            # Find closest tickets to cluster centers
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
            
            # Build cluster info
            clusters = []
            for cluster_id, ticket_idx in enumerate(closest):
                ticket_id = ticket_ids[ticket_idx]
                cluster_tickets = [ticket_ids[i] for i in range(len(open_tickets)) if kmeans.labels_[i] == cluster_id]
                
                # Get representative ticket details
                rep_ticket = open_tickets[ticket_idx]
                
                clusters.append(TicketCluster(
                    cluster_id=cluster_id,
                    representative_ticket_id=ticket_id,
                    representative_title=titles[ticket_idx],
                    count=len(cluster_tickets),
                    category=rep_ticket.get("category", TicketCategory.GENERAL.value),
                    priority=rep_ticket.get("priority", TicketPriority.MEDIUM.value)
                ))
            
            return clusters
        except Exception as e:
            logger.error(f"Ticket clustering failed: {e}")
            return []

    def _format_ticket_response(self, ticket_doc: Dict) -> TicketResponse:
        return TicketResponse(
            id=str(ticket_doc["_id"]),
            title=ticket_doc["title"],
            description=ticket_doc["description"],
            customer=CustomerInfo(**ticket_doc["customer"]),
            category=ticket_doc["category"],
            priority=ticket_doc["priority"],
            status=ticket_doc["status"],
            created_at=ticket_doc["created_at"],
            updated_at=ticket_doc["updated_at"],
            resolution=ticket_doc.get("resolution"),
            agent_assigned=ticket_doc.get("agent_assigned"),
            confidence_score=ticket_doc.get("confidence_score"),
            auto_resolved=ticket_doc.get("auto_resolved", False),
            sentiment=ticket_doc.get("sentiment"),
            summary=ticket_doc.get("summary"),
            related_tickets=ticket_doc.get("related_tickets", [])
        )

# Initialize service
ticket_service = TicketService(db)

# API Routes
@app.post("/tickets/", response_model=TicketResponse)
async def create_ticket(ticket: Union[TicketCreate,List[TicketCreate]]):
    """Create a new ticket with AI enhancements"""
    try:
        if type(ticket) == list:
            reslist =[]
            for i in ticket:
                reslist.append(await ticket_service.create_ticket(i))
            return(reslist)
        else:
            return await ticket_service.create_ticket(ticket)
    except Exception as e:
        logger.error(f"Error creating ticket: {e}")
        raise HTTPException(status_code=500, detail="Failed to create ticket")

@app.get("/tickets/", response_model=List[TicketResponse])
async def list_tickets(status: Optional[TicketStatus] = None, limit: int = 50):
    """Get tickets with optional status filter"""
    return await ticket_service.get_tickets(status, limit)

@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str):
    """Get a specific ticket"""
    ticket = await ticket_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

@app.put("/tickets/{ticket_id}/status", response_model=TicketResponse)
async def update_ticket_status(ticket_id: str, status: TicketStatus):
    """Update ticket status"""
    ticket = await ticket_service.update_ticket_status(ticket_id, status)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

@app.get("/analytics/")
async def get_analytics():
    """Get system analytics"""
    return await ticket_service.get_analytics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/demo/create-sample-tickets")
async def create_sample_tickets():
    """Create sample tickets for demo purposes"""
    
    sample_tickets = [
        TicketCreate(
            title="Internet connection is down",
            description="My internet has been down for 2 hours. Can't connect to WiFi at all. This is unacceptable!",
            customer=CustomerInfo(
                customer_id="CUST001",
                name="John Doe",
                email="john@example.com",
                phone="555-1234",
                location="New York, NY",
                service_area="NYC_ZONE_1"
            )
        ),
        TicketCreate(
            title="Bill is higher than usual",
            description="My bill this month is $50 higher than last month. I haven't changed my plan. Please explain why.",
            customer=CustomerInfo(
                customer_id="CUST002",
                name="Jane Smith",
                email="jane@example.com",
                phone="555-5678",
                location="Los Angeles, CA",
                service_area="LA_ZONE_2"
            )
        ),
        TicketCreate(
            title="Email not working on phone",
            description="Cannot send or receive emails on my iPhone. Other internet works fine. Need help configuring my email.",
            customer=CustomerInfo(
                customer_id="CUST003",
                name="Bob Johnson",
                email="bob@example.com",
                phone="555-9012",
                location="Chicago, IL",
                service_area="CHI_ZONE_1"
            )
        )
    ]
    
    created_tickets = []
    for ticket_data in sample_tickets:
        ticket = await ticket_service.create_ticket(ticket_data)
        created_tickets.append(ticket)
    
    return {"message": f"Created {len(created_tickets)} sample tickets", "tickets": created_tickets}

@app.get("/tickets/clusters", response_model=List[TicketCluster])
async def get_ticket_clusters(n_clusters: int = 5):
    """Cluster open tickets to identify common issues"""
    return await ticket_service.cluster_tickets(n_clusters)

@app.get("/tickets/similar/{ticket_id}", response_model=List[TicketResponse])
async def get_similar_tickets(ticket_id: str, limit: int = 5):
    """Get tickets similar to a given ticket"""
    ticket = await ticket_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    similar_tickets = []
    for related_id in ticket.related_tickets[:limit]:
        related_ticket = await ticket_service.get_ticket_by_id(related_id)
        if related_ticket:
            similar_tickets.append(related_ticket)
    
    return similar_tickets

@app.post("/knowledge/", status_code=201)
async def add_knowledge_article(article: KnowledgeBaseArticle):
    """Add knowledge base article with vector embedding"""
    try:
        # Generate embedding
        embedding = await ticket_service._generate_embedding(
            article.title + " " + " ".join(article.keywords)
        )
        
        # Insert into knowledge base
        article_dict = article.dict()
        article_dict["embedding"] = embedding
        
        await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: knowledge_collection.insert_one(article_dict)
        )
        
        return {"message": "Knowledge article added successfully"}
    except Exception as e:
        logger.error(f"Error adding knowledge article: {e}")
        raise HTTPException(status_code=500, detail="Failed to add knowledge article")

@app.get("/knowledge/", response_model=List[KnowledgeBaseArticle])
async def get_knowledge_articles(limit: int = 100):
    """List knowledge base articles"""
    articles = await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: list(knowledge_collection.find().limit(limit))
    )
    return [KnowledgeBaseArticle(**article) for article in articles]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)