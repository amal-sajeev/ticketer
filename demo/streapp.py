import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
import asyncio
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(
    page_title="AI Ticketing System Demo",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: white;
        margin: 0.5rem 0 0 0;
        text-align: center;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .ticket-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .auto-resolved {
        border-left-color: #28a745 !important;
    }
    
    .high-priority {
        border-left-color: #dc3545 !important;
    }
    
    .medium-priority {
        border-left-color: #ffc107 !important;
    }
    
    .low-priority {
        border-left-color: #6c757d !important;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'tickets' not in st.session_state:
    st.session_state.tickets = []
if 'analytics' not in st.session_state:
    st.session_state.analytics = {}
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Helper functions
def make_api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Please ensure the FastAPI server is running on localhost:8000")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def get_priority_color(priority: str) -> str:
    """Get color for priority level"""
    colors = {
        "critical": "#dc3545",
        "high": "#fd7e14", 
        "medium": "#ffc107",
        "low": "#6c757d"
    }
    return colors.get(priority.lower(), "#6c757d")

def get_status_color(status: str) -> str:
    """Get color for status"""
    colors = {
        "open": "#17a2b8",
        "in_progress": "#fd7e14",
        "resolved": "#28a745",
        "closed": "#6c757d",
        "escalated": "#dc3545"
    }
    return colors.get(status.lower(), "#6c757d")

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emojis = {
        "positive": "😊",
        "neutral": "😐",
        "negative": "😟",
        "urgent": "🚨"
    }
    return emojis.get(sentiment.lower(), "😐")

def load_tickets():
    """Load tickets from API"""
    tickets = make_api_request("/tickets/")
    if tickets:
        st.session_state.tickets = tickets
        return tickets
    return []

def load_analytics():
    """Load analytics from API"""
    analytics = make_api_request("/analytics/")
    if analytics:
        st.session_state.analytics = analytics
        return analytics
    return {}

def create_sample_tickets():
    """Create sample tickets for demo"""
    result = make_api_request("/demo/create-sample-tickets", method="POST")
    if result:
        st.success(f"✅ {result['message']}")
        load_tickets()
        return True
    return False

# Header
st.markdown("""
<div class="main-header">
    <h1>🎫 AI Ticketing System Demo</h1>
    <p>Intelligent Customer Support with Auto-Resolution & Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Demo Controls")
    
    # Demo mode toggle
    demo_mode = st.toggle("Demo Mode", value=st.session_state.demo_mode)
    st.session_state.demo_mode = demo_mode
    
    if demo_mode:
        st.info("🎭 Demo mode enabled - Auto-refresh and sample data")
        # Auto-refresh every 10 seconds in demo mode
        st_autorefresh(interval=10000, key="demo_refresh")
    
    st.divider()
    
    # Quick actions
    st.header("⚡ Quick Actions")
    
    if st.button("🎯 Create Sample Tickets", type="primary"):
        with st.spinner("Creating sample tickets..."):
            create_sample_tickets()
    
    if st.button("🔄 Refresh Data"):
        with st.spinner("Refreshing..."):
            load_tickets()
            load_analytics()
            st.success("Data refreshed!")
    
    st.divider()
    
    # Filters
    st.header("🔍 Filters")
    
    status_filter = st.selectbox(
        "Status Filter",
        ["All", "open", "in_progress", "resolved", "closed", "escalated"]
    )
    
    priority_filter = st.selectbox(
        "Priority Filter", 
        ["All", "critical", "high", "medium", "low"]
    )
    
    auto_resolved_filter = st.selectbox(
        "Auto-Resolved Filter",
        ["All", "Yes", "No"]
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Tickets", "📊 Analytics", "🎯 Create Ticket", "🔬 Advanced"])
    
    with tab1:
        st.header("🎫 Ticket Management")
        
        # Load tickets
        tickets = load_tickets()
        
        if tickets:
            # Filter tickets
            filtered_tickets = tickets
            
            if status_filter != "All":
                filtered_tickets = [t for t in filtered_tickets if t['status'] == status_filter]
            
            if priority_filter != "All":
                filtered_tickets = [t for t in filtered_tickets if t['priority'] == priority_filter]
            
            if auto_resolved_filter != "All":
                auto_resolved_bool = auto_resolved_filter == "Yes"
                filtered_tickets = [t for t in filtered_tickets if t['auto_resolved'] == auto_resolved_bool]
            
            st.write(f"**Showing {len(filtered_tickets)} of {len(tickets)} tickets**")
            
            # Display tickets
            for ticket in filtered_tickets:
                priority_class = f"{ticket['priority'].lower()}-priority"
                if ticket['auto_resolved']:
                    priority_class += " auto-resolved"
                
                # Use st.container instead of raw HTML
                with st.container():
                    # Style the container with CSS classes by setting background color, borders, etc.
                    st.write(f"### {ticket['title']}")
                    st.write(f"**Customer:** {ticket['customer']['name']} ({ticket['customer']['email']})")
                    st.write(f"**Description:** {ticket['description']}")
                    st.write(f"**Created:** {ticket['created_at']}")
                    # Add status and priority with st.markdown and emoji if needed
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 View Details", key=f"view_{ticket['id']}"):
                            ...
                    with col2:
                        if st.button(f"✅ Resolve", key=f"resolve_{ticket['id']}"):
                            ...

                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"📋 View Details", key=f"view_{ticket['id']}"):
                        st.session_state.selected_ticket = ticket['id']
                
                with col2:
                    if ticket['status'] == 'open':
                        if st.button(f"✅ Resolve", key=f"resolve_{ticket['id']}"):
                            result = make_api_request(f"/tickets/{ticket['id']}/status", method="PUT", data={"status": "resolved"})
                            if result:
                                st.success("Ticket resolved!")
                                load_tickets()
                                st.rerun()
                
                st.divider()
        else:
            st.info("No tickets found. Create some sample tickets to get started!")
    
    with tab2:
        st.header("📊 System Analytics")
        
        # Load analytics
        analytics = load_analytics()
        
        if analytics:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Tickets",
                    analytics.get('total_tickets', 0),
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Auto-Resolved",
                    analytics.get('auto_resolved', 0),
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Automation Rate",
                    f"{analytics.get('automation_rate', 0):.1f}%",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Avg Confidence",
                    f"{analytics.get('average_confidence', 0):.1f}%",
                    delta=None
                )
            
            # Charts
            if tickets:
                # Status distribution
                status_counts = {}
                priority_counts = {}
                category_counts = {}
                sentiment_counts = {}
                
                for ticket in tickets:
                    status = ticket['status']
                    priority = ticket['priority']
                    category = ticket['category']
                    sentiment = ticket.get('sentiment', 'neutral')
                    
                    status_counts[status] = status_counts.get(status, 0) + 1
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1
                    category_counts[category] = category_counts.get(category, 0) + 1
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Status pie chart
                    if status_counts:
                        fig = px.pie(
                            values=list(status_counts.values()),
                            names=list(status_counts.keys()),
                            title="Ticket Status Distribution"
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Priority bar chart
                    if priority_counts:
                        fig = px.bar(
                            x=list(priority_counts.keys()),
                            y=list(priority_counts.values()),
                            title="Ticket Priority Distribution",
                            color=list(priority_counts.keys()),
                            color_discrete_map={
                                'critical': '#dc3545',
                                'high': '#fd7e14',
                                'medium': '#ffc107',
                                'low': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Category distribution
                    if category_counts:
                        fig = px.bar(
                            x=list(category_counts.keys()),
                            y=list(category_counts.values()),
                            title="Ticket Category Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    # Sentiment distribution
                    if sentiment_counts:
                        fig = px.pie(
                            values=list(sentiment_counts.values()),
                            names=list(sentiment_counts.keys()),
                            title="Customer Sentiment Distribution"
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Auto-resolution effectiveness
                st.subheader("🤖 Auto-Resolution Effectiveness")
                
                auto_resolved_tickets = [t for t in tickets if t['auto_resolved']]
                manual_tickets = [t for t in tickets if not t['auto_resolved']]
                
                if auto_resolved_tickets:
                    avg_confidence = sum(t.get('confidence_score', 0) for t in auto_resolved_tickets) / len(auto_resolved_tickets)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Auto-Resolved Tickets", len(auto_resolved_tickets))
                    
                    with col2:
                        st.metric("Manual Tickets", len(manual_tickets))
                    
                    with col3:
                        st.metric("Avg AI Confidence", f"{avg_confidence:.1%}")
                    
                    # Confidence distribution
                    confidences = [t.get('confidence_score', 0) for t in auto_resolved_tickets]
                    if confidences:
                        fig = px.histogram(
                            x=confidences,
                            nbins=10,
                            title="Auto-Resolution Confidence Distribution",
                            labels={'x': 'Confidence Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analytics data available.")
    
    with tab3:
        st.header("🎯 Create New Ticket")
        
        with st.form("create_ticket"):
            st.subheader("Customer Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                customer_name = st.text_input("Customer Name", value="Demo Customer")
                customer_email = st.text_input("Email", value="demo@example.com")
                customer_phone = st.text_input("Phone", value="555-0123")
            
            with col2:
                customer_location = st.text_input("Location", value="New York, NY")
                service_area = st.text_input("Service Area", value="NYC_ZONE_1")
                customer_id = st.text_input("Customer ID", value="DEMO001")
            
            st.subheader("Ticket Details")
            
            ticket_title = st.text_input("Issue Title", placeholder="Brief description of the problem")
            ticket_description = st.text_area(
                "Detailed Description",
                placeholder="Please describe the issue in detail...",
                height=150
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                suggested_category = st.selectbox(
                    "Suggested Category (Optional)",
                    ["", "network", "billing", "technical", "account", "service", "general"]
                )
            
            with col2:
                suggested_priority = st.selectbox(
                    "Suggested Priority (Optional)",
                    ["", "critical", "high", "medium", "low"]
                )
            
            submitted = st.form_submit_button("🚀 Create Ticket", type="primary")
            
            if submitted:
                if ticket_title and ticket_description:
                    ticket_data = {
                        "title": ticket_title,
                        "description": ticket_description,
                        "customer": {
                            "customer_id": customer_id,
                            "name": customer_name,
                            "email": customer_email,
                            "phone": customer_phone,
                            "location": customer_location,
                            "service_area": service_area
                        }
                    }
                    
                    if suggested_category:
                        ticket_data["category"] = suggested_category
                    
                    if suggested_priority:
                        ticket_data["priority"] = suggested_priority
                    
                    with st.spinner("Creating ticket and running AI analysis..."):
                        result = make_api_request("/tickets/", method="POST", data=ticket_data)
                        
                        if result:
                            st.success("✅ Ticket created successfully!")
                            
                            # Show AI analysis results
                            st.subheader("🤖 AI Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Category", result['category'].title())
                            
                            with col2:
                                st.metric("Priority", result['priority'].title())
                            
                            with col3:
                                st.metric("Sentiment", f"{get_sentiment_emoji(result.get('sentiment', 'neutral'))} {result.get('sentiment', 'neutral').title()}")
                            
                            if result['auto_resolved']:
                                st.success(f"🎉 **Auto-Resolved!** (Confidence: {result.get('confidence_score', 0):.1%})")
                                st.info(f"**AI Response:** {result.get('resolution', 'No resolution provided')}")
                            else:
                                st.warning(f"⚠️ Requires human attention - assigned to {result.get('agent_assigned', 'Support Team')}")
                            
                            if result.get('summary'):
                                st.write(f"**Summary:** {result['summary']}")
                            
                            # Refresh ticket list
                            load_tickets()
                else:
                    st.error("Please fill in both title and description.")
    
    with tab4:
        st.header("🔬 Advanced Features")
        
        # Ticket clustering
        st.subheader("🎯 Ticket Clustering")
        st.write("Analyze patterns in open tickets to identify common issues")
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        if st.button("🔍 Analyze Ticket Patterns"):
            with st.spinner("Analyzing ticket patterns..."):
                clusters = make_api_request(f"/tickets/clusters?n_clusters={n_clusters}")
                
                if clusters:
                    st.success(f"Found {len(clusters)} ticket clusters")
                    
                    for cluster in clusters:
                        with st.expander(f"Cluster {cluster['cluster_id'] + 1}: {cluster['representative_title']} ({cluster['count']} tickets)"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Category", cluster['category'].title())
                            
                            with col2:
                                st.metric("Priority", cluster['priority'].title())
                            
                            with col3:
                                st.metric("Ticket Count", cluster['count'])
                            
                            st.write(f"**Representative Ticket:** {cluster['representative_title']}")
                            st.write(f"**Ticket ID:** {cluster['representative_ticket_id']}")
                else:
                    st.info("No clusters found. Create more tickets to enable clustering.")
        
        # Knowledge base management
        st.subheader("📚 Knowledge Base")
        
        # Display knowledge articles
        if st.button("📖 View Knowledge Base"):
            with st.spinner("Loading knowledge base..."):
                articles = make_api_request("/knowledge/")
                
                if articles:
                    st.success(f"Found {len(articles)} knowledge articles")
                    
                    for article in articles:
                        with st.expander(f"{article['title']} ({article['category']})"):
                            st.write(f"**Keywords:** {', '.join(article['keywords'])}")
                            st.write(f"**Category:** {article['category']}")
                            st.write(f"**Priority:** {article['priority']}")
                            st.write(f"**Solution:** {article['solution']}")
                else:
                    st.info("No knowledge articles found.")
        
        # Add new knowledge article
        st.subheader("➕ Add Knowledge Article")
        
        with st.form("add_knowledge"):
            kb_title = st.text_input("Article Title")
            kb_keywords = st.text_input("Keywords (comma-separated)")
            kb_solution = st.text_area("Solution", height=100)
            kb_category = st.selectbox("Category", ["network", "billing", "technical", "account", "service", "general"])
            kb_priority = st.selectbox("Priority", ["critical", "high", "medium", "low"])
            
            if st.form_submit_button("Add Article"):
                if kb_title and kb_keywords and kb_solution:
                    article_data = {
                        "title": kb_title,
                        "keywords": [k.strip() for k in kb_keywords.split(",")],
                        "solution": kb_solution,
                        "category": kb_category,
                        "priority": kb_priority
                    }
                    
                    result = make_api_request("/knowledge/", method="POST", data=article_data)
                    
                    if result:
                        st.success("Knowledge article added successfully!")
                else:
                    st.error("Please fill in all fields.")

with col2:
    st.header("🎯 System Status")
    
    # API Health Check
    health = make_api_request("/health")
    
    if health:
        st.success("✅ API Server Online")
        st.write(f"**Status:** {health['status']}")
        st.write(f"**Last Check:** {datetime.fromisoformat(health['timestamp'].replace('Z', '+00:00')).strftime('%H:%M:%S')}")
    else:
        st.error("❌ API Server Offline")
        st.warning("Please start the FastAPI server:\n```bash\npython app.py\n```")
    
    st.divider()
    
    # Quick stats
    st.subheader("📈 Quick Stats")
    
    analytics = st.session_state.get('analytics', {})
    
    if analytics:
        st.metric("Total Tickets", analytics.get('total_tickets', 0))
        st.metric("Auto-Resolved", analytics.get('auto_resolved', 0))
        st.metric("Automation Rate", f"{analytics.get('automation_rate', 0):.1f}%")
        st.metric("Avg Confidence", f"{analytics.get('average_confidence', 0):.1f}%")
    
    st.divider()
    
    # Demo tips
    st.subheader("💡 Demo Tips")
    
    with st.expander("🎭 Demo Features"):
        st.write("""
        **Key Features to Showcase:**
        
        1. **Auto-Resolution** - AI automatically resolves simple issues
        2. **Sentiment Analysis** - Detects customer emotions
        3. **Smart Categorization** - AI classifies tickets
        4. **Priority Assignment** - Automatic priority based on urgency
        5. **Ticket Clustering** - Identifies common issues
        6. **Knowledge Base** - Semantic search for solutions
        7. **Real-time Analytics** - Live dashboard updates
        """)
    
    with st.expander("🎯 Sample Scenarios"):
        st.write("""
        **Try these ticket scenarios:**
        
        1. **Network Issue:**
           - "My internet is down, can't connect to WiFi"
           - Should auto-resolve with restart instructions
        
        2. **Billing Question:**
           - "My bill is higher than usual this month"
           - Should escalate to billing specialist
        
        3. **Technical Support:**
           - "Can't set up email on my phone"
           - Should provide configuration help
        
        4. **Urgent Issue:**
           - "Complete service outage, affecting business!"
           - Should prioritize as critical
        """)
    
    st.divider()
    
    # System info
    st.subheader("⚙️ System Info")
    st.write(f"**Demo Mode:** {'🎭 Enabled' if st.session_state.demo_mode else '🎯 Disabled'}")
    st.write(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.session_state.demo_mode:
        st.info("🔄 Auto-refreshing every 10 seconds")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎫 AI Ticketing System Demo | Built with Streamlit & FastAPI</p>
    <p>Features: Auto-Resolution • Sentiment Analysis • Smart Categorization • Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)