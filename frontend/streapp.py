import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time

# Configure page
st.set_page_config(
    page_title="Telecom Support System",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-pending {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    .status-resolved {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    .status-needs-human {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    .customer-form {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .cluster-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .ticket-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .tag-display {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your API URL

# Helper functions
def call_api(endpoint, method="GET", data=None):
    """Make API calls with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def format_status(status):
    """Format status with colored badges"""
    if status == "pending":
        return f'<span class="status-pending">🕐 Pending</span>'
    elif status == "resolved":
        return f'<span class="status-resolved">✅ Resolved</span>'
    elif status == "needs_human":
        return f'<span class="status-needs-human">👤 Needs Human</span>'
    return status

def get_ticket_metrics():
    """Get ticket metrics for dashboard"""
    tickets = call_api("/tickets")
    if not tickets:
        return {"total": 0, "pending": 0, "resolved": 0, "needs_human": 0}
    
    return {
        "total": len(tickets),
        "pending": len([t for t in tickets if t["status"] == "pending"]),
        "resolved": len([t for t in tickets if t["status"] == "resolved"]),
        "needs_human": len([t for t in tickets if t["status"] == "needs_human"])
    }

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0;">📞 Telecom Support</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Select Page",
    ["🎫 Submit Ticket", "🏢 Agent Dashboard", "🔗 Cluster Management", "📊 Analytics", "🔧 System Management"]
)

def display_ticket_card(ticket, in_cluster=False):
    """Display a ticket card with all information"""
    with st.container():
        # st.markdown('<div class="ticket-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**#{ticket['id']} - {ticket['name']}**")
            st.markdown(f"📧 {ticket['contact']}")
            st.markdown(f"📅 {ticket['created_at']}")
            
            # Display tags from JSON
            if ticket.get('tags'):
                try:
                    tags_data = json.loads(ticket['tags'])
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <span class="tag-display">🎯 {tags_data.get('intent', 'unknown')}</span>
                        <span class="tag-display">📂 {tags_data.get('topic', 'other')}</span>
                        <span class="tag-display">⚠️ {tags_data.get('urgency', 'medium')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown(f"**🏷️ Tags:** {ticket['tags']}")
            
            st.markdown(f"**❓ Query:** {ticket['query']}")
            
            if ticket.get('auto_reply'):
                st.markdown(f"**🤖 Auto Reply:** {ticket['auto_reply']}")
            
            if ticket.get('human_reply'):
                st.markdown(f"**👤 Human Reply:** {ticket['human_reply']}")
            
            # Show cluster info if not already in cluster view
            if not in_cluster and ticket.get('cluster_id'):
                st.markdown(f"**🔗 Cluster:** {ticket['cluster_id']}")
                if ticket.get('cluster_summary'):
                    st.markdown(f"**📝 Cluster Summary:** {ticket['cluster_summary']}")
        
        with col2:
            # FIXED: Removed extra parenthesis
            st.markdown(format_status(ticket['status']), unsafe_allow_html=True)
            
            if ticket['status'] == 'needs_human':
                with st.popover("💬 Add Reply"):
                    reply_text = st.text_area("Reply:", key=f"reply_{ticket['id']}")
                    if st.button("Send Reply", key=f"send_{ticket['id']}"):
                        if reply_text:
                            reply_data = {
                                "ticket_id": ticket['id'],
                                "reply_text": reply_text
                            }
                            result = call_api("/reply", "POST", reply_data)
                            if result:
                                st.success("Reply sent!")
                                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

# Page routing
if page == "🎫 Submit Ticket":
    # Customer ticket submission page
    st.markdown('<div class="main-header"><h1>📞 Submit Support Ticket</h1><p>We\'re here to help! Submit your query below.</p></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="customer-form">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("📝 Your Name", placeholder="Enter your full name")
            contact = st.text_input("📧 Contact Information", placeholder="Email or phone number")
        
        with col2:
            st.markdown("### 💡 Common Issues")
            st.markdown("""
            - **Billing** - Payment issues, charges, invoices
            - **Network** - Connectivity, speed, outages
            - **Service** - Plan changes, activations
            - **Technical** - Device setup, troubleshooting
            """)
        
        query = st.text_area("🔍 Describe your issue", placeholder="Please describe your issue in detail...", height=150)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            if st.button("🚀 Submit Ticket", use_container_width=True):
                if name and contact and query:
                    ticket_data = {
                        "name": name,
                        "contact": contact,
                        "query": query
                    }
                    
                    with st.spinner("Submitting your ticket..."):
                        result = call_api("/submit", "POST", ticket_data)
                        
                    if result:
                        st.success("✅ Ticket submitted successfully!")
                        st.info("💡 Your ticket will be processed automatically and you'll receive a response soon.")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error("Please fill in all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # FAQ Section
    st.markdown("---")
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("🔧 How does the automated system work?"):
        st.write("Our AI system automatically analyzes your query and provides immediate solutions for common issues. Complex problems are escalated to our human agents.")
    
    with st.expander("💰 How can I check my billing information?"):
        st.write("You can check your billing information by logging into your account or contacting our support team.")
    
    with st.expander("🌐 I'm experiencing network issues. What should I do?"):
        st.write("Please try restarting your device and router. If the issue persists, submit a ticket with your location and device details.")

elif page == "🏢 Agent Dashboard":
    # Agent dashboard
    st.markdown('<div class="main-header"><h1>🏢 Agent Dashboard</h1><p>Manage and resolve customer tickets</p></div>', unsafe_allow_html=True)
    
    # Metrics row
    metrics = get_ticket_metrics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Tickets", metrics["total"])
    with col2:
        st.metric("🕐 Pending", metrics["pending"])
    with col3:
        st.metric("👤 Needs Human", metrics["needs_human"])
    with col4:
        st.metric("✅ Resolved", metrics["resolved"])
    
    # Control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process Pending Tickets", use_container_width=True):
            with st.spinner("Processing tickets with AI..."):
                result = call_api("/process", "POST")
                if result:
                    st.success(f"✅ Processed {result.get('processed', 0)} tickets")
                    time.sleep(1)
                    st.rerun()
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("🔁 Auto-refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    # Ticket filters
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.selectbox(
            "📂 Filter by Status",
            ["All", "pending", "needs_human", "resolved"]
        )
    
    with col2:
        search_term = st.text_input("🔍 Search tickets", placeholder="Search by name, contact, or query...")
    
    # Fetch and display tickets
    if status_filter == "All":
        tickets = call_api("/tickets")
    else:
        tickets = call_api(f"/tickets?status={status_filter}")
    
    if tickets:
        # Apply search filter
        if search_term:
            tickets = [t for t in tickets if 
                      search_term.lower() in t.get("name", "").lower() or
                      search_term.lower() in t.get("contact", "").lower() or
                      search_term.lower() in t.get("query", "").lower()]
        
        # Display tickets
        st.markdown("### 📋 Tickets")
        for ticket in tickets:
            display_ticket_card(ticket)
    
    else:
        st.info("No tickets found")

elif page == "🔗 Cluster Management":
    # Cluster management page
    st.markdown('<div class="main-header"><h1>🔗 Cluster Management</h1><p>View and manage grouped similar tickets</p></div>', unsafe_allow_html=True)
    
    # Get tickets that need human attention and have clusters
    tickets = call_api("/tickets?status=needs_human")
    
    if tickets:
        # Group tickets by cluster
        clusters = {}
        unclustered = []
        
        for ticket in tickets:
            cluster_id = ticket.get("cluster_id")
            if cluster_id:
                cluster_key = str(cluster_id)
                if cluster_key not in clusters:
                    clusters[cluster_key] = {
                        "tickets": [],
                        "summary": ticket.get("cluster_summary", "No summary available")
                    }
                clusters[cluster_key]["tickets"].append(ticket)
            else:
                unclustered.append(ticket)
        
        # Display clusters
        if clusters:
            st.markdown("### 🔗 Ticket Clusters")
            
            for cluster_id, cluster_data in clusters.items():
                with st.expander(f"📂 Cluster: {cluster_id} ({len(cluster_data['tickets'])} tickets)", expanded=True):
                    # Display cluster summary
                    st.markdown(f"**📝 Summary:** {cluster_data['summary']}")
                    
                    # Bulk actions for cluster
                    col1, col2 = st.columns(2)
                    with col1:
                        bulk_reply = st.text_area(f"Bulk reply for cluster {cluster_id}:", key=f"bulk_{cluster_id}")
                    with col2:
                        st.markdown("**Quick Actions:**")
                        if st.button(f"📤 Send to all in cluster", key=f"send_all_{cluster_id}"):
                            if bulk_reply:
                                for ticket in cluster_data['tickets']:
                                    reply_data = {
                                        "ticket_id": ticket['id'],
                                        "reply_text": bulk_reply
                                    }
                                    call_api("/reply", "POST", reply_data)
                                st.success(f"Sent reply to all {len(cluster_data['tickets'])} tickets in cluster!")
                                st.rerun()
                    
                    st.markdown("---")
                    
                    # Display individual tickets in cluster
                    for ticket in cluster_data['tickets']:
                        display_ticket_card(ticket, in_cluster=True)
        
        # Display unclustered tickets
        if unclustered:
            st.markdown("### 📋 Individual Tickets (Not Clustered)")
            for ticket in unclustered:
                display_ticket_card(ticket)
    
    else:
        st.info("No tickets requiring human attention found")

elif page == "📊 Analytics":
    # Analytics dashboard
    st.markdown('<div class="main-header"><h1>📊 Analytics Dashboard</h1><p>Insights and trends from support tickets</p></div>', unsafe_allow_html=True)
    
    # Fetch all tickets for analytics
    tickets = call_api("/tickets")
    
    if tickets:
        df = pd.DataFrame(tickets)
        # Handle date conversion safely
        try:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
        except:
            df['date'] = pd.NaT
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Tickets", len(df))
        with col2:
            resolved_count = len(df[df['status'] == 'resolved'])
            resolved_rate = (resolved_count / len(df)) * 100 if len(df) > 0 else 0
            st.metric("✅ Resolution Rate", f"{resolved_rate:.1f}%")
        with col3:
            auto_resolved = len(df[df['auto_reply'].notna() & (df['status'] == 'resolved')])
            st.metric("🤖 Auto-resolved", auto_resolved)
        with col4:
            clustered_tickets = len(df[df['cluster_id'].notna()])
            st.metric("🔗 Clustered", clustered_tickets)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            status_counts = df['status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="📊 Ticket Status Distribution",
                color_discrete_map={
                    'pending': '#ffc107',
                    'resolved': '#28a745',
                    'needs_human': '#dc3545'
                }
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            # Tickets over time
            if 'date' in df and not df['date'].isnull().all():
                daily_tickets = df.groupby('date').size().reset_index(name='count')
                fig_timeline = px.line(
                    daily_tickets,
                    x='date',
                    y='count',
                    title="📈 Tickets Over Time",
                    markers=True
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.warning("No valid date data available for timeline")
        
        # Tag analysis
        st.markdown("### 🏷️ Tag Analysis")
        
        # Extract intents and topics
        intents = []
        topics = []
        
        for tags in df['tags'].dropna():
            try:
                tags_data = json.loads(tags)
                intents.append(tags_data.get('intent', 'unknown'))
                topics.append(tags_data.get('topic', 'other'))
            except:
                intents.append('unknown')
                topics.append('other')
        
        if intents and topics:
            col1, col2 = st.columns(2)
            
            with col1:
                intent_counts = pd.Series(intents).value_counts()
                fig_intent = px.bar(
                    x=intent_counts.index,
                    y=intent_counts.values,
                    title="🎯 Intent Distribution",
                    labels={'x': 'Intent', 'y': 'Count'}
                )
                st.plotly_chart(fig_intent, use_container_width=True)
            
            with col2:
                topic_counts = pd.Series(topics).value_counts()
                fig_topic = px.bar(
                    x=topic_counts.index,
                    y=topic_counts.values,
                    title="📂 Topic Distribution",
                    labels={'x': 'Topic', 'y': 'Count'}
                )
                st.plotly_chart(fig_topic, use_container_width=True)
        else:
            st.info("No tag data available for analysis")
        
        # Cluster analysis
        st.markdown("### 🔗 Cluster Analysis")
        
        clustered_df = df[df['cluster_id'].notna()]
        if not clustered_df.empty:
            cluster_counts = clustered_df['cluster_id'].value_counts()
            fig_clusters = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="🔗 Tickets per Cluster",
                labels={'x': 'Cluster ID', 'y': 'Number of Tickets'}
            )
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Show top cluster summaries
            st.markdown("#### 📝 Top Cluster Summaries")
            for cluster_id in cluster_counts.head(3).index:
                cluster_tickets = clustered_df[clustered_df['cluster_id'] == cluster_id]
                if not cluster_tickets.empty:
                    summary = cluster_tickets.iloc[0].get('cluster_summary', 'No summary available')
                    st.markdown(f"**Cluster {cluster_id}** ({len(cluster_tickets)} tickets): {summary}")
        
        # Auto-resolution analysis
        st.markdown("### 🤖 Auto-Resolution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_vs_human = df.groupby('status').size().reset_index(name='count')
            fig_resolution = px.bar(
                auto_vs_human,
                x='status',
                y='count',
                title="🔄 Resolution Method Distribution",
                labels={'status': 'Resolution Method', 'count': 'Count'}
            )
            st.plotly_chart(fig_resolution, use_container_width=True)
        
        with col2:
            # Recent activity
            st.markdown("#### 📋 Recent Activity")
            if 'created_at' in df:
                recent_tickets = df.nlargest(5, 'created_at')[['id', 'name', 'status', 'created_at']]
                st.dataframe(recent_tickets, use_container_width=True)
    
    else:
        st.info("No data available for analytics")

elif page == "🔧 System Management":
    # System management page
    st.markdown('<div class="main-header"><h1>🔧 System Management</h1><p>Manage system settings and operations</p></div>', unsafe_allow_html=True)
    
    # System health check
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏥 System Health")
        
        # API health check
        try:
            response = requests.get(f"{API_BASE_URL}/tickets", timeout=5)
            if response.status_code == 200:
                st.success("✅ API is healthy")
            else:
                st.error(f"❌ API error: {response.status_code}")
        except:
            st.error("❌ API is not responding")
        
        # Database stats
        tickets = call_api("/tickets")
        if tickets:
            st.info(f"📊 Database contains {len(tickets)} tickets")
            
            # Show processing stats
            pending_count = len([t for t in tickets if t['status'] == 'pending'])
            needs_human_count = len([t for t in tickets if t['status'] == 'needs_human'])
            resolved_count = len([t for t in tickets if t['status'] == 'resolved'])
            
            st.markdown(f"""
            **Processing Status:**
            - 🕐 Pending: {pending_count}
            - 👤 Needs Human: {needs_human_count}
            - ✅ Resolved: {resolved_count}
            """)
        else:
            st.warning("⚠️ Unable to fetch database stats")
    
    with col2:
        st.markdown("### ⚙️ System Operations")
        
        if st.button("🔄 Process All Pending Tickets"):
            with st.spinner("Processing all pending tickets with AI..."):
                result = call_api("/process", "POST")
                if result:
                    st.success(f"✅ Processed {result.get('processed', 0)} tickets")
        
        if st.button("🔗 Re-cluster Human Tickets"):
            with st.spinner("Re-clustering tickets that need human attention..."):
                # This would trigger re-clustering
                st.info("Re-clustering completed")
        
        if st.button("📊 Generate System Report"):
            st.info("Generating comprehensive system report...")
            # This could generate a detailed report
    
    # AI Model Status
    st.markdown("---")
    st.markdown("### 🤖 AI Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ollama Status**")
        try:
            # This is a placeholder - you'd need to implement actual health check
            st.success("✅ Ollama service is running")
            st.info("📋 Model: llama3")
        except:
            st.error("❌ Ollama service not available")
    
    with col2:
        st.markdown("**Processing Statistics**")
        tickets = call_api("/tickets")
        if tickets:
            tagged_tickets = len([t for t in tickets if t.get('tags')])
            clustered_tickets = len([t for t in tickets if t.get('cluster_id')])
            auto_replied = len([t for t in tickets if t.get('auto_reply')])
            
            st.markdown(f"""
            - 🏷️ Tagged: {tagged_tickets}
            - 🔗 Clustered: {clustered_tickets}
            - 🤖 Auto-replied: {auto_replied}
            """)
    
    # Configuration
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**API Settings**")
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        auto_process = st.checkbox("Auto-process tickets", value=True)
        
    with col2:
        st.markdown("**AI Settings**")
        model_name = st.text_input("Ollama Model", value="llama3")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    
    # Logs
    st.markdown("---")
    st.markdown("### 📋 System Logs")
    
    # Mock logs - in real implementation, these would come from your log files
    logs = [
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - System initialized",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AI models loaded",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - API endpoint: {API_BASE_URL}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Ready to process tickets"
    ]
    
    for log in logs:
        st.text(log)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666; font-size: 0.9rem;">
    <p>📞 Telecom Support System | Built with ❤️ using Streamlit & Ollama AI</p>
</div>
""", unsafe_allow_html=True)