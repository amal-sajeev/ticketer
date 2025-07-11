from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from datetime import datetime
import models
import ollama_client
import json
import numpy as np

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
conn = sqlite3.connect('tickets.db', check_same_thread=False)
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    contact TEXT NOT NULL,
    query TEXT NOT NULL,
    tags TEXT,
    embedding BLOB,
    cluster_id INTEGER,
    cluster_summary TEXT,
    status TEXT DEFAULT 'pending',
    auto_reply TEXT,
    human_reply TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    solution TEXT NOT NULL,
    embedding BLOB,
    use_count INTEGER DEFAULT 0
)
''')
conn.commit()

@app.post("/submit")
def submit_ticket(ticket: models.TicketCreate):
    cursor.execute('''
    INSERT INTO tickets (name, contact, query, status)
    VALUES (?, ?, ?, 'pending')
    ''', (ticket.name, ticket.contact, ticket.query))
    conn.commit()
    return {"message": "Ticket submitted successfully"}

@app.post("/process")
def process_tickets():
    # Get unprocessed tickets
    cursor.execute("SELECT * FROM tickets WHERE status = 'pending'")
    pending_tickets = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    # Process each ticket
    for row in pending_tickets:
        ticket = dict(zip(columns, row))
        ticket_id = ticket['id']
        query = ticket['query']
        
        # Generate tags and embedding
        tags = ollama_client.tag_query(query)
        embedding = ollama_client.get_embedding(query)
        
        # Check if we can auto-resolve
        auto_resolve = False
        if not tags.get('requires_human', True):
            # Find similar solutions
            cursor.execute("SELECT embedding FROM knowledge_base")
            kb_embeddings = [row[0] for row in cursor.fetchall()]
            similar_ids = ollama_client.find_similar_solutions(query, kb_embeddings)
            
            if similar_ids:
                # Get best solution
                cursor.execute("SELECT solution FROM knowledge_base WHERE id = ?", (similar_ids[0],))   # IDs start at 1
                solution = cursor.fetchone()[0]
                auto_resolve = True
        
        # Update ticket metadata
        cursor.execute('''
        UPDATE tickets 
        SET tags = ?, embedding = ?
        WHERE id = ?
        ''', (json.dumps(tags), embedding, ticket_id))
        
        # Generate response or escalate
        if auto_resolve:
            cursor.execute('''
            UPDATE tickets 
            SET auto_reply = ?, status = 'resolved' 
            WHERE id = ?
            ''', (solution, ticket_id))
        else:
            auto_reply = ollama_client.generate_auto_reply(query, tags)
            print('''
                INSERT INTO knowledge_base (query, solution, embedding)
                VALUES (?, ?, ?)
                ''', (query, auto_reply, embedding))
            if not auto_reply.get('escalate', True):
                cursor.execute('''
                UPDATE tickets 
                SET auto_reply = ?, status = 'resolved' 
                WHERE id = ?
                ''', (auto_reply, ticket_id))
                
                # Add to knowledge base if resolved
                cursor.execute('''
                INSERT INTO knowledge_base (query, solution, embedding)
                VALUES (?, ?, ?)
                ''', (query, auto_reply, embedding))
            else:
                cursor.execute('''
                UPDATE tickets 
                SET auto_reply = ?, status = 'needs_human' 
                WHERE id = ?
                ''', (auto_reply, ticket_id))
    
    conn.commit()
    
    # Cluster tickets needing human review
    cursor.execute('''
    SELECT id, query, tags, embedding 
    FROM tickets 
    WHERE status = 'needs_human'
    ''')
    needs_human = [
        {"id": row[0], "query": row[1], "tags": row[2], "embedding": row[3]}
        for row in cursor.fetchall()
    ]
    
    clusters = ollama_client.cluster_tickets(needs_human)
    
    # Create cluster summaries
    for cluster_idx, ticket_ids in enumerate(clusters, start=1):
        # Get queries for this cluster
        placeholders = ','.join(['?'] * len(ticket_ids))
        cursor.execute(
            f"SELECT query FROM tickets WHERE id IN ({placeholders})",
            ticket_ids
        )
        queries = [row[0] for row in cursor.fetchall()]
        
        # Generate cluster summary
        summary = ollama_client.summarize_cluster(queries)
        
        # Update tickets with cluster info
        cursor.execute(
            f"UPDATE tickets SET cluster_id = ?, cluster_summary = ? WHERE id IN ({placeholders})",
            (cluster_idx, summary, *ticket_ids)
        )
    
    conn.commit()
    return {"processed": len(pending_tickets), "clusters_created": len(clusters)}

@app.get("/tickets")
def get_tickets(status: str = None):
    query = "SELECT * FROM tickets"
    params = ()
    
    if status:
        query += " WHERE status = ?"
        params = (status,)
    
    query += " ORDER BY created_at DESC"
    
    cursor.execute(query, params)
    columns = [col[0] for col in cursor.description]
    tickets = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    # Parse tags JSON
    for ticket in tickets:
        ticket.pop('embedding', None)
    if ticket['tags']:
        try:
            ticket['tags'] = json.loads(ticket['tags'])
        except json.JSONDecodeError:
            # Handle invalid JSON more gracefully
            ticket['tags'] = {"error": "Invalid tag format"}
    
    return tickets

@app.post("/reply")
def agent_reply(reply: models.AgentReply):
    # Get ticket details
    cursor.execute("SELECT query FROM tickets WHERE id = ?", (reply.ticket_id,))
    query = cursor.fetchone()[0]
    
    # Update ticket
    cursor.execute('''
    UPDATE tickets 
    SET human_reply = ?, status = 'resolved' 
    WHERE id = ?
    ''', (reply.reply_text, reply.ticket_id))
    
    # Add to knowledge base
    embedding = ollama_client.get_embedding(query)
    cursor.execute('''
    INSERT INTO knowledge_base (query, solution, embedding)
    VALUES (?, ?, ?)
    ''', (query, reply.reply_text, embedding))
    
    conn.commit()
    return {"message": "Reply submitted and added to knowledge base"}

@app.get("/cluster/{cluster_id}")
def get_cluster_tickets(cluster_id: int):
    cursor.execute('''
    SELECT * 
    FROM tickets 
    WHERE cluster_id = ?
    ORDER BY created_at DESC
    ''', (cluster_id,))
    
    columns = [col[0] for col in cursor.description]
    tickets = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return tickets

@app.get("/knowledge")
def search_knowledge(query: str):
# ✅ Get embeddings AND ids
    cursor.execute("SELECT id, embedding FROM knowledge_base")
    kb_items = [{"id": row[0], "embedding": row[1]} for row in cursor.fetchall()]

    # ✅ Find similar embeddings by index
    similar_indexes = ollama_client.find_similar_solutions(
        query, [item["embedding"] for item in kb_items]
    )

    if similar_indexes:
        similar_kb_id = kb_items[similar_indexes[0]]["id"]
        cursor.execute("SELECT solution FROM knowledge_base WHERE id = ?", (similar_kb_id,))
        row = cursor.fetchone()
        if row:
            solution = row[0]
            auto_resolve = True
    
    # Get solutions
    solutions = []
    for kb_id in similar_ids:
        cursor.execute("SELECT query, solution FROM knowledge_base WHERE id = ?", (kb_id + 1,))
        row = cursor.fetchone()
        solutions.append({
            "id": kb_id + 1,
            "query": row[0],
            "solution": row[1]
        })
    
    return solutions