import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import uuid
import pydeck as pdk

# Page configuration
st.set_page_config(
    page_title="Telecom AI Customer Service Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = os.getenv("TICKET_API_URL", "http://localhost:8000")

# Custom CSS for modern styling
st.markdown("""
<style>
    /* ---- MAIN HEADER ---- */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        color: white;
        border-radius: 0 0 15px 15px;
    }

    /* ---- GLASS PANELS ---- */
    .metric-card,
    .ticket-card,
    .form-container,
    .analytics-container {
        background: rgba(255, 255, 255, 0.03); /* semi-transparent white */
        backdrop-filter: blur(15px);           /* glass blur */
        -webkit-backdrop-filter: blur(15px);   /* Safari support */
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2); /* subtle border */
        padding: 1.5rem;
        margin: 1rem 0;
        color: #f1f5f9;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1); /* optional depth */
    }

    .metric-card:hover,
    .ticket-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    }

    .metric-flex {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .metric-left {
        display: flex;
        align-items: center;
        gap: 10px;
        border-right: 1px solid rgba(255, 255, 255, 0.2);
        padding-right: 20px;
    }

    .metric-left .icon {
        font-size: 1.5rem;
    }

    .metric-left .title {
        font-size: 1rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        padding-left: 20px;
        color: #1E90FF;
    }

    .ticket-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
    }

    .status-pending { border-left: 4px solid #fbbf24; }
    .status-in-progress { border-left: 4px solid #3b82f6; }
    .status-resolved { border-left: 4px solid #10b981; }
    .status-escalated { border-left: 4px solid #ef4444; }

    .priority-low { color: #10b981; }
    .priority-medium { color: #f59e0b; }
    .priority-high { color: #ef4444; }
    .priority-urgent { color: #dc2626; font-weight: bold; }

    /* ---- MESSAGES ---- */
    .success-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }

    .error-message {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }

    /* ---- BADGES ---- */
    .info-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #f1f5f9;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
    }

    /* ---- NAV BUTTON ---- */
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 0.25rem;
    }

    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* ---- SIDEBAR BUTTONS ---- */
    .sidebar .stButton button {
        width: 100%;
        text-align: left;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.1);
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .sidebar .stButton button:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    .sidebar .stButton button:active {
        background: rgba(255, 255, 255, 0.3);
    }

    /* ---- FOOTER ---- */
    .footer {
        color: #f1f5f9;
    }

    /* ---- BODY ---- */
    body {
        background: linear-gradient(135deg, #0f172a, #1e293b);
    }
</style>

""", unsafe_allow_html=True)

# Initialize session state
if 'selected_ticket_id' not in st.session_state:
    st.session_state.selected_ticket_id = None
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# API Helper Functions
def create_ticket(ticket_data):
    """Create a new support ticket"""
    try:
        response = requests.post(f"{API_BASE_URL}/tickets", json=ticket_data, timeout=10)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {str(e)}")
        return None

def get_tickets(status=None, category=None, priority=None, limit:int = None):
    """Fetch tickets with optional filters"""
    params = {}
    if status: params["status"] = status
    if category: params["category"] = category
    if priority: params["priority"] = priority
    if limit: params["limit"] = limit
    
    try:
        response = requests.get(f"{API_BASE_URL}/tickets", params=params, timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException:
        # Return mock data for demo purposes
        return generate_mock_tickets()

def get_ticket(ticket_id):
    """Get specific ticket details"""
    try:
        response = requests.get(f"{API_BASE_URL}/tickets/{ticket_id}", timeout=10)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return generate_mock_ticket(ticket_id)

def get_analytics(days=30):
    """Get analytics data"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics?days={days}", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return generate_mock_analytics()

def update_ticket_status(ticket_id, status, agent_name=None):
    """Update ticket status"""
    try:
        data = {"status": status}
        if agent_name:
            data["assigned_agent"] = agent_name
        response = requests.patch(f"{API_BASE_URL}/tickets/{ticket_id}", json=data, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return True  # Mock success for demo

def add_ticket_note(ticket_id, content, note_type="internal", agent_name=""):
    """Add note to ticket"""
    try:
        data = {
            "content": content,
            "type": note_type,
            "agent_name": agent_name
        }
        response = requests.post(f"{API_BASE_URL}/tickets/{ticket_id}/notes", json=data, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return True  # Mock success for demo

# Mock data generators for demo purposes
def generate_mock_tickets():
    """Generate mock tickets for demo"""
    statuses = ["pending", "in_progress", "resolved", "escalated"]
    categories = ["billing", "technical", "internet", "account"]
    priorities = ["low", "medium", "high", "urgent"]
    
    tickets = []
    for i in range(10):
        ticket = {
            "id": str(uuid.uuid4()),
            "title": f"Sample Issue #{i+1}",
            "description": f"This is a sample description for ticket {i+1}. Customer is experiencing issues with their service.",
            "customer_name": f"Customer {i+1}",
            "customer_email": f"customer{i+1}@example.com",
            "status": statuses[i % len(statuses)],
            "category": categories[i % len(categories)],
            "priority": priorities[i % len(priorities)],
            "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
            "ai_resolution": "AI suggested resolution" if i % 3 == 0 else None,
            "manual_resolution": "Manual resolution provided" if i % 4 == 0 else None
        }
        tickets.append(ticket)
    
    return tickets

def generate_mock_ticket(ticket_id):
    """Generate a single mock ticket"""
    return {
        "id": ticket_id,
        "title": "Sample Network Connectivity Issue",
        "description": "Customer experiencing intermittent connectivity issues in their area. Speed tests show significant degradation during peak hours.",
        "customer_name": "John Doe",
        "customer_email": "john.doe@example.com",
        "status": "in_progress",
        "category": "technical",
        "priority": "high",
        "created_at": datetime.now().isoformat(),
        "ai_resolution": "AI analysis suggests network congestion in the area. Recommend upgrading local infrastructure.",
        "location": {
            "coordinates": [-122.4194, 37.7749]
        },
        "notes": [
            {
                "content": "Initial diagnostic completed",
                "type": "internal",
                "agent_name": "Agent Smith",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

def generate_mock_analytics():
    """Generate mock analytics data"""
    return {
        "total_tickets": 150,
        "resolved_by_ai": 90,
        "escalated_to_human": 25,
        "avg_resolution_time": 4.5,
        "category_distribution": {
            "Technical": 45,
            "Billing": 30,
            "Internet": 40,
            "Account": 25,
            "General": 10
        },
        "sentiment_distribution": {
            "Positive": 40,
            "Neutral": 70,
            "Negative": 30,
            "Critical": 10
        },
        "daily_tickets": [
            {"date": "2024-01-01", "count": 12},
            {"date": "2024-01-02", "count": 18},
            {"date": "2024-01-03", "count": 15},
            {"date": "2024-01-04", "count": 22},
            {"date": "2024-01-05", "count": 19}
        ]
    }

def get_documentation():
    """Fetch documentation from knowledge base"""
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge/documentation", timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {str(e)}")
        return []

def get_service_memory():
    """Fetch service memory entries"""
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge/service-memory", timeout=10)
        return response.json() if response.status_code == 200 else []
    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {str(e)}")
        return []


# Utility functions
def format_datetime(dt_string):
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        return dt.strftime('%b %d, %Y at %I:%M %p')
    except:
        return dt_string

def get_status_color(status):
    """Get color for ticket status"""
    colors = {
        "pending": "#fbbf24",
        "in_progress": "#3b82f6", 
        "resolved": "#10b981",
        "escalated": "#ef4444"
    }
    return colors.get(status, "#6b7280")

def get_priority_color(priority):
    """Get color for priority level"""
    colors = {
        "low": "#10b981",
        "medium": "#f59e0b",
        "high": "#ef4444", 
        "urgent": "#dc2626"
    }
    return colors.get(priority, "#6b7280")

# Page Components
def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Customer Service Dashboard</h1>
        <p>Intelligent Telecom Support Management System</p>
    </div>
    """, unsafe_allow_html=True)

def render_navigation():
    """Render navigation sidebar"""
    st.sidebar.title("📱 Navigation")
    
    nav_options = {
        "🏠 Dashboard": "dashboard",
        "📝 Create Ticket": "create_ticket", 
        "📋 View Tickets": "view_tickets",
        "🔍 Ticket Details": "ticket_details",
        "📚 Knowledge Base": "knowledge_base",
        "📊 Analytics": "analytics",
        "🗺️ Location analysis" : "location_analysis"
    }
    
    for label, page_key in nav_options.items():
        if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.rerun()

def render_dashboard():
    """Render main dashboard"""
    st.header("📊 Service Overview")
    
    # Quick stats
    analytics = get_analytics(7)  # Last 7 days
    
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("Total Tickets", analytics.get('total_tickets', 0), "🎫"),
        ("AI Resolved", analytics.get('resolved_by_ai', 0), "🤖"),
        ("Escalated", analytics.get('escalated_to_human', 0), "⚠️"),
        ("Avg Resolution", f"{analytics.get('avg_resolution_time', 0):.1f}h", "⏱️")
    ]
    
    for i, (title, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-flex">
                    <div class="metric-left">
                        <span class="icon" style="font-size: 2rem">{icon}</span>
                        <span class="title" style="font-size: 1.5rem">{title}</span>
                    </div>
                    <div class="metric-value">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    

    # ----------- TICKET CARDS ----------
    st.markdown("### 📋 Recent Tickets")

    tickets = get_tickets()[:10]

    # Dynamically set cards per row
    CARDS_PER_ROW = 3
    rows = [tickets[i:i + CARDS_PER_ROW] for i in range(0, len(tickets), CARDS_PER_ROW)]

    for row in rows:
        cols = st.columns(CARDS_PER_ROW)
        for col, ticket in zip(cols, row):
            with col:
                priority_color = get_priority_color(ticket['priority'])
                status_class = f"status-{ticket['status'].replace('_', '-')}"

                with st.container():
                    st.markdown(f"""
                        <div class="ticket-card {status_class}">
                            <h4>{ticket['title']}</h4>
                            <p><strong>Customer:</strong> {ticket['customer_name']}</p>
                            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;">
                                <span class="info-badge">📂 {ticket['category'].title()}</span>
                                <span class="info-badge" style="background: {priority_color}20; color: {priority_color};">
                                    🔥 {ticket['priority'].title()}
                                </span>
                                <span class="info-badge">📅 {format_datetime(ticket['created_at'])}</span>
                            </div>
                    """, unsafe_allow_html=True)

                    # Pure Streamlit working button
                    if st.button("🔍 View Details", key=f"view_{ticket['id']}", use_container_width=True):
                        st.session_state.selected_ticket_id = ticket['id']
                        st.session_state.page = 'ticket_details'
                        st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)

def render_create_ticket():
    """Render create ticket form"""
    st.header("📝 Create New Support Ticket")
    
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    with st.form("create_ticket_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_name = st.text_input("Customer Name *", placeholder="Enter customer name")
            customer_email = st.text_input("Email Address *", placeholder="customer@example.com")
            category = st.selectbox("Category", ["billing", "technical", "internet", "account", "general"])
        
        with col2:
            title = st.text_input("Issue Title *", placeholder="Brief description of the issue")
            priority = st.selectbox("Priority", ["low", "medium", "high", "urgent"])
            phone = st.text_input("Phone Number", placeholder="Optional")
        
        description = st.text_area("Detailed Description *", 
                                 placeholder="Please provide detailed information about the issue...",
                                 height=150)
        
        # Location (optional)
        st.markdown("##### 📍 Location Information (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", format="%.6f", value=0.0)
        with col2:
            longitude = st.number_input("Longitude", format="%.6f", value=0.0)
        
        submitted = st.form_submit_button("🚀 Create Ticket", use_container_width=True)
        
        if submitted:
            # Validation
            if not all([customer_name, customer_email, title, description]):
                st.error("❌ Please fill in all required fields marked with *")
            else:
                # Create ticket data
                ticket_data = {
                    "title": title,
                    "description": description,
                    "customer_name": customer_name,
                    "customer_email": customer_email,
                    "category": category,
                    "priority": priority
                }
                
                if phone:
                    ticket_data["phone"] = phone
                
                if latitude != 0.0 and longitude != 0.0:
                    ticket_data["location"] = {
                        "latitude": latitude,
                        "longitude": longitude
                    }
                
                # Submit ticket
                with st.spinner("Creating ticket..."):
                    result = create_ticket(ticket_data)
                
                if result:
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ Ticket created successfully!<br>
                        <strong>Ticket ID:</strong> {result.get('id', 'N/A')}<br>
                        <strong>Status:</strong> {result.get('status', 'pending').title()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show ticket details
                    with st.expander("View Created Ticket"):
                        st.json(result)
                else:
                    st.markdown("""
                    <div class="error-message">
                        ❌ Failed to create ticket. Please check API connection and try again.
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_view_tickets():
    """Render tickets list with filters"""
    st.header("📋 Support Tickets")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.selectbox("Status", ["All", "pending", "in_progress", "resolved", "escalated"])
    with col2:
        category_filter = st.selectbox("Category", ["All", "billing", "technical", "internet", "account", "general"])
    with col3:
        priority_filter = st.selectbox("Priority", ["All", "low", "medium", "high", "urgent"])
    with col4:
        search_query = st.text_input("🔍 Search", placeholder="Search tickets...")
    
    # Get filtered tickets
    tickets = get_tickets(
        status=status_filter if status_filter != "All" else None,
        category=category_filter if category_filter != "All" else None,
        priority=priority_filter if priority_filter != "All" else None
    )
    
    # Apply search filter
    if search_query:
        tickets = [t for t in tickets if search_query.lower() in t['title'].lower() or 
                  search_query.lower() in t['customer_name'].lower()]
    
    st.markdown(f"**Found {len(tickets)} tickets**")
    
    if not tickets:
        st.info("No tickets found matching your criteria.")
        return
    
    # Display tickets
    for ticket in tickets:
        status_class = f"status-{ticket['status'].replace('_', '-')}"
        priority_color = get_priority_color(ticket['priority'])
        
        st.markdown(f"""
        <div class="ticket-card {status_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex-grow: 1;">
                    <h4>{ticket['title']}</h4>
                    <p><strong>Customer:</strong> {ticket['customer_name']} ({ticket['customer_email']})</p>
                    <p style="color: #9da2ab; margin: 10px 0;">{ticket['description'][:100]}...</p>
                    <div style="display: flex; gap: 15px; margin-top: 10px;">
                        <span class="info-badge">📂 {ticket['category'].title()}</span>
                        <span class="info-badge" style="background: {priority_color}20; color: {priority_color};">
                            🔥 {ticket['priority'].title()}
                        </span>
                        <span class="info-badge">📅 {format_datetime(ticket['created_at'])}</span>
                        <span class="info-badge">🔄 {ticket['status'].replace('_', ' ').title()}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"View Details", key=f"view_details_{ticket['id']}"):
            st.session_state.selected_ticket_id = ticket['id']
            st.session_state.page = 'ticket_details'
            st.rerun()

def render_ticket_details():
    """Render detailed ticket view"""
    ticket_id = st.session_state.selected_ticket_id
    
    if not ticket_id:
        st.warning("⚠️ No ticket selected. Please select a ticket from the tickets list.")
        if st.button("← Back to Tickets"):
            st.session_state.page = 'view_tickets'
            st.rerun()
        return
    
    ticket = get_ticket(ticket_id)
    if not ticket:
        st.error("❌ Ticket not found")
        return
    
    # Header
    st.markdown(f"### 🎫 Ticket Details - {ticket['title']}")
    
    # Back button
    if st.button("← Back to Tickets"):
        st.session_state.page = 'view_tickets'
        st.rerun()
    
    # Ticket information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("---")
        
        # Basic info
        st.markdown("#### 📋 Ticket Information")
        status_color = get_status_color(ticket['status'])
        priority_color = get_priority_color(ticket['priority'])
        
        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin: 20px 0;">
            <div style="flex: 1;">
                <p><strong>Status:</strong> <span style="color: {status_color};">● {ticket['status'].replace('_', ' ').title()}</span></p>
                <p><strong>Priority:</strong> <span style="color: {priority_color};">🔥 {ticket['priority'].title()}</span></p>
                <p><strong>Category:</strong> 📂 {ticket['category'].title()}</p>
                <p><strong>Created:</strong> 📅 {format_datetime(ticket['created_at'])}</p>
            </div>
            <div style="flex: 1;">
                <p><strong>Customer:</strong> {ticket['customer_name']}</p>
                <p><strong>Email:</strong> {ticket['customer_email']}</p>
                <p><strong>Ticket ID:</strong> {ticket['id'][:8]}...</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Description
        st.markdown("#### 📝 Issue Description")
        st.markdown(f'<div class="form-container" style="border-left: 4px solid #667eea;">{ticket["description"]}</div>', unsafe_allow_html=True)
        
        # AI Resolution
        if ticket.get('ai_resolution'):
            st.markdown("#### 🤖 AI Resolution")
            st.markdown(f'<div class = "form-container" style="border-radius: 10px; border-left: 4px solid #10b981;">{ticket["ai_resolution"]}</div>', unsafe_allow_html=True)
        
        # Manual Resolution
        if ticket.get('manual_resolution'):
            st.markdown("#### 👤 Agent Resolution")
            st.markdown(f'<div class = "form-container" style= border-radius: 10px; border-left: 4px solid #3b82f6;">{ticket["manual_resolution"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Actions panel
        st.markdown("---")
        st.markdown("#### ⚡ Quick Actions")
        
        # Status update
        new_status = st.selectbox("Update Status", 
                                ["pending", "in_progress", "resolved", "escalated"],
                                index=["pending", "in_progress", "resolved", "escalated"].index(ticket['status']))
        
        if st.button("Update Status", use_container_width=True):
            if update_ticket_status(ticket_id, new_status):
                st.success(f"✅ Status updated to {new_status}")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to update status")
        
        # Agent assignment
        agent_name = st.text_input("Assign Agent", placeholder="Enter agent name")
        if st.button("Assign Agent", use_container_width=True):
            if agent_name:
                if update_ticket_status(ticket_id, ticket['status'], agent_name):
                    st.success(f"✅ Assigned to {agent_name}")
                else:
                    st.error("❌ Failed to assign agent")
            else:
                st.warning("⚠️ Please enter agent name")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Notes section
    st.markdown("---")
    st.markdown("#### 📝 Add Note")
    
    with st.form("add_note_form"):
        note_content = st.text_area("Note Content", height=100)
        col1, col2 = st.columns(2)
        with col1:
            note_type = st.selectbox("Type", ["internal", "customer_facing"])
        with col2:
            agent_name = st.text_input("Agent Name")
        
        if st.form_submit_button("Add Note", use_container_width=True):
            if note_content:
                if add_ticket_note(ticket_id, note_content, note_type, agent_name):
                    st.success("✅ Note added successfully")
                else:
                    st.error("❌ Failed to add note")
            else:
                st.warning("⚠️ Please enter note content")

def render_analytics():
    """Render analytics dashboard"""
    st.header("📊 Service Analytics")
    
    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Performance Overview")
    with col2:
        days = st.selectbox("Period", [7, 30, 90], format_func=lambda x: f"Last {x} days")
    
    analytics = get_analytics(days)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Tickets", analytics.get('total_tickets', 0), "🎫"),
        ("AI Resolved", analytics.get('resolved_by_ai', 0), "🤖"),
        ("Escalated", analytics.get('escalated_to_human', 0), "⚠️"),
        ("Avg Resolution", f"{analytics.get('avg_resolution_time', 0):.1f}h", "⏱️")
    ]
    
    for i, (title, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-flex">
                    <div class="metric-left">
                        <span class="icon" style="font-size: 2rem">{icon}</span>
                        <span class="title" style="font-size: 1.5rem">{title}</span>
                    </div>
                    <div class="metric-value">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        
        st.markdown("#### 📂 Category Distribution")
        if "category_distribution" in analytics:
            cat_data = analytics["category_distribution"]
            fig = px.bar(
                x=list(cat_data.keys()),
                y=list(cat_data.values()),
                color=list(cat_data.values()),
                color_continuous_scale="blues"
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Category",
                yaxis_title="Number of Tickets",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### 😊 Sentiment Analysis")
        if "sentiment_distribution" in analytics:
            sent_data = analytics["sentiment_distribution"]
            fig = px.pie(
                values=list(sent_data.values()),
                names=list(sent_data.keys()),
                color_discrete_map={
                    "Positive": "#10b981",
                    "Neutral": "#6b7280", 
                    "Negative": "#f59e0b",
                    "Critical": "#ef4444"
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    # Daily trend
    if "daily_tickets" in analytics:
        st.markdown("#### 📈 Daily Ticket Trend")
        daily_data = analytics["daily_tickets"]
        df = pd.DataFrame(daily_data)
        fig = px.line(
            df, 
            x='date', 
            y='count',
            title="Daily Ticket Volume",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Tickets",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Resolution Performance")
        
        total_tickets = analytics.get('total_tickets', 1)
        ai_resolved = analytics.get('resolved_by_ai', 0)
        escalated = analytics.get('escalated_to_human', 0)
        
        performance_data = {
            "AI Resolved": ai_resolved,
            "Human Resolved": total_tickets - ai_resolved - escalated,
            "Still Open": escalated
        }
        
        fig = px.bar(
            x=list(performance_data.keys()),
            y=list(performance_data.values()),
            color=list(performance_data.values()),
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Resolution Type",
            yaxis_title="Number of Tickets",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ⏱️ Resolution Time Analysis")
        
        # Mock resolution time data
        resolution_times = {
            "< 1 hour": 45,
            "1-4 hours": 60,
            "4-24 hours": 35,
            "> 24 hours": 10
        }
        
        fig = px.pie(
            values=list(resolution_times.values()),
            names=list(resolution_times.keys()),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
def render_knowledge_base():
    """Render knowledge base management with real API data"""
    st.header("📚 Knowledge Base Management")
    
    tab1, tab2, tab3 = st.tabs(["📖 Documentation", "💾 Service Memory", "🔍 Search Knowledge"])
    
    with tab1:
        st.markdown("#### Add New Documentation")
        st.markdown("---")
        
        with st.form("add_documentation"):
            title = st.text_input("Documentation Title")
            category = st.selectbox("Category", ["billing", "technical", "internet", "account", "general"])
            content = st.text_area("Content", height=200)
            tags = st.text_input("Tags (comma-separated)", placeholder="troubleshooting, network, setup")
            
            if st.form_submit_button("📝 Add Documentation", use_container_width=True):
                if title and content:
                    # POST request would go here
                    st.success("✅ Documentation added successfully!")
                else:
                    st.error("❌ Please fill in title and content")
        
        # Display real documentation
        st.markdown("#### 📋 Documentation Library")
        with st.spinner("Loading documentation..."):
            docs = get_documentation()
        
        if not docs:
            st.info("No documentation found in the knowledge base.")
            return
            
        for doc in docs:
            st.markdown(f"""
            <div class="ticket-card">
                <h4>{doc.get('title', 'Untitled')}</h4>
                <p><strong>Category:</strong> {doc.get('category', '').title()} 
                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    {doc.get('content', 'No content available')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Add Service Memory")
        st.markdown("---")
        
        with st.form("add_service_memory"):
            query = st.text_input("Customer Query")
            resolution = st.text_area("Resolution", height=150)
            category = st.selectbox("Category", ["billing", "technical", "internet", "account", "general"])
            agent_name = st.text_input("Agent Name")
            effectiveness = st.slider("Effectiveness Rating", 1, 5, 5)
            
            if st.form_submit_button("💾 Add to Memory", use_container_width=True):
                if query and resolution:
                    # POST request would go here
                    st.success("✅ Service memory added successfully!")
                else:
                    st.error("❌ Please fill in query and resolution")
        
        # Display real service memory
        st.markdown("#### 💾 Service Memory Entries")
        with st.spinner("Loading service memory..."):
            memories = get_service_memory()
        
        if not memories:
            st.info("No service memory entries found.")
            return
            
        for memory in memories:
            st.markdown(f"""
            <div class="ticket-card">
            <h4>Customer Query</h4>
            <p>{memory.get('query', 'No query available')}</p>

            <h4>Resolution</h4>
            <p>{memory.get('resolution', 'No resolution available')}</p>

            <div style="display: flex; gap: 10px; margin-top: 15px;">
            <span class="info-badge">👤 {memory.get('agent_name', 'Unknown agent')}</span>
            <span class="info-badge">📂 {memory.get('category', '').title()}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # This tab remains similar but would need API integration for search
        st.markdown("#### 🔍 Search Knowledge Base")
        st.markdown("---")
        
        search_query = st.text_input("Search Query", placeholder="Enter keywords to search...")
        search_category = st.selectbox("Filter by Category", ["All", "billing", "technical", "internet", "account", "general"])
        
        if st.button("🔍 Search", use_container_width=True):
            st.info("Search functionality would call API endpoints here")

#Location Analytics

def render_ticket_map(API_BASE_URL: str):
    """
    Renders an interactive map page showing ticket locations.
    Filters: priority, status, category, customer, title, date (if available).
    Colors: markers colored by priority.
    """

    st.title("📍 Customer Service Tickets Map")

    tickets = get_tickets()

    # Extract valid locations
    records = []
    for t in tickets:
        loc = t.get("location")
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            records.append({
                "lat": loc["latitude"],
                "lon": loc["longitude"],
                "title": t.get("title"),
                "status": t.get("status"),
                "priority": t.get("priority"),
                "category": t.get("category"),
                "customer": t.get("customer_name"),
                "created_at": t.get("created_at")
            })

    if not records:
        st.info("No tickets with valid location data found.")
        return

    df = pd.DataFrame(records)
    print(len(df))
    st.sidebar.markdown("___")
    st.sidebar.header("🔍 Filters")
    
    # Priority filter
    priorities = sorted(df["priority"].dropna().unique().tolist())
    selected_priorities = st.sidebar.multiselect("Priority", priorities, default=priorities)

    # Status filter
    statuses = sorted(df["status"].dropna().unique().tolist())
    selected_statuses = st.sidebar.multiselect("Status", statuses, default=statuses)

    # Category filter
    categories = sorted(df["category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)

    # Customer filter
    customers = sorted(df["customer"].dropna().unique().tolist())
    selected_customers = st.sidebar.multiselect("Customer", customers, default=customers)

    # Title filter
    title_substring = st.sidebar.text_input("Title contains (optional)").strip().lower()

    # Date filter
    if df["created_at"].notnull().any():
        df["created_at"] = pd.to_datetime(df["created_at"])
        min_date = df["created_at"].min().date()
        max_date = df["created_at"].max().date()
        date_range = st.sidebar.date_input("Created date range", [min_date, max_date])
    else:
        date_range = None

    # Apply all filters
    filtered_df = df[
        df["priority"].isin(selected_priorities) &
        df["status"].isin(selected_statuses) &
        df["category"].isin(selected_categories) &
        df["customer"].isin(selected_customers)
    ]

    if title_substring:
        filtered_df = filtered_df[filtered_df["title"].str.lower().str.contains(title_substring)]

    if date_range and len(date_range) == 2 and df["created_at"].notnull().any():
        start, end = date_range
        filtered_df = filtered_df[
            (df["created_at"].dt.date >= start) &
            (df["created_at"].dt.date <= end)
        ]

    st.subheader(f"Showing {len(filtered_df)} tickets")

    if filtered_df.empty:
        st.warning("No tickets match the selected filters.")
        return

    # -------------------------------
    # Map priority to color
    # -------------------------------
    priority_colors = {
        "low": [0, 200, 0, 180],        # Green
        "medium": [255, 165, 0, 180],   # Orange
        "high": [255, 0, 0, 180],       # Red
        "urgent": [128, 0, 128, 180]    # Purple
    }

    def get_color(priority):
        return priority_colors.get(priority.lower(), [0, 0, 255, 180])  # Fallback: Blue

    filtered_df["color"] = filtered_df["priority"].apply(get_color)

    # -------------------------------
    # Scatterplot with color
    # -------------------------------
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=5000,
        pickable=True
    )

    view_state = pdk.ViewState(
        longitude=filtered_df["lon"].mean(),
        latitude=filtered_df["lat"].mean(),
        zoom=4,
        pitch=0,
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{customer}</b><br/>"
                    "Title: {title}<br/>"
                    "Category: {category}<br/>"
                    "Priority: {priority}<br/>"
                    "Status: {status}",
            "style": {"color": "white"}
        }
    )

    st.pydeck_chart(r)

    with st.expander("📄 See Filtered Data"):
        st.dataframe(filtered_df)
        if st.sidebar.button("🔄 Refresh Tickets"):
            st.cache_data.clear()
            st.rerun()

# Main application
def main():
    render_header()
    render_navigation()
    
    # Route to appropriate page
    page = st.session_state.get('page', 'dashboard')
    
    if page == 'dashboard':
        render_dashboard()
    elif page == 'create_ticket':
        render_create_ticket()
    elif page == 'view_tickets':
        render_view_tickets()
    elif page == 'ticket_details':
        render_ticket_details()
    elif page == 'analytics':
        render_analytics()
    elif page == 'knowledge_base':
        render_knowledge_base()
    elif page == 'location_analysis':  # Changed from 'location_analytics'
        render_ticket_map(API_BASE_URL)  # Also added the required API_BASE_URL parameter
    
    # Footer
    st.markdown("---")
    st.markdown("""  <!-- ADDED footer class -->
    <div class="footer" style="text-align: center; padding: 20px;">
        <p>🚀 AI Customer Service Dashboard | Built with Streamlit</p>
        <p>API Status: <span style="color: #10b981;">● Connected</span> | Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()