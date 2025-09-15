import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import time
import sqlite3
from typing import Dict, List, Optional, Any

# Import the integrated router and evaluation system
from router_and_eval_component import (
    SystemConfig,
    load_config,
    LangGraphWorkflowOrchestrator,
    DeepEvalManager,
    LANGCHAIN_AVAILABLE,
    LANGGRAPH_AVAILABLE,
    DEEPEVAL_AVAILABLE,
)

# Import existing components for direct access when needed
from onboarding_component import (
    OnboardingWorkflow,
    OnboardingConfig,
    create_sample_onboarding_docs,
)
from re_upskilling_component import (
    LeadershipDevelopmentWorkflow,
    LeadershipConfig,
    create_sample_leadership_docs,
)
from resume_reader_component import (
    ReaderWorkflow,
    ReaderConfig,
)

# Database compatibility
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ========================================================================================
# PAGE CONFIGURATION
# ========================================================================================

st.set_page_config(
    page_title="SAP HCM System - Early Talent Development",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.sap.com/support",
        "Report a bug": None,
        "About": "# SAP HCM System\nEarly Talent Development Platform\n\nVersion 2.0 - Integrated Router & Evaluation",
    },
)

# ========================================================================================
# CUSTOM STYLING
# ========================================================================================

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .evaluation-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .success-message {
        padding: 1rem;
        background: #d4edda;
        color: #155724;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        padding: 1rem;
        background: #fff3cd;
        color: #856404;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ========================================================================================
# SESSION STATE INITIALIZATION
# ========================================================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None

if "config" not in st.session_state:
    st.session_state.config = None

if "database_connected" not in st.session_state:
    st.session_state.database_connected = False

# ========================================================================================
# MOCK USER DATABASE
# ========================================================================================

USERS_DB = {
    "hr_admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "HR",
        "full_name": "Sarah Johnson",
        "department": "Human Resources",
    },
    "hr_manager": {
        "password": hashlib.sha256("manager123".encode()).hexdigest(),
        "role": "HR",
        "full_name": "Michael Chen",
        "department": "Talent Development",
    },
    "john_doe": {
        "password": hashlib.sha256("talent123".encode()).hexdigest(),
        "role": "EARLY_TALENT",
        "full_name": "John Doe",
        "department": "Engineering",
        "start_date": "2024-01-15",
        "manager": "Sarah Johnson",
    },
    "jane_smith": {
        "password": hashlib.sha256("talent456".encode()).hexdigest(),
        "role": "EARLY_TALENT",
        "full_name": "Jane Smith",
        "department": "Product Management",
        "start_date": "2024-02-01",
        "manager": "Michael Chen",
    },
}

# ========================================================================================
# DATABASE MANAGER FOR SCALABLE OPERATIONS
# ========================================================================================


class ScalableDatabaseManager:
    """Scalable database manager for resume analysis and talent data."""

    def __init__(self, db_path: str = "test_resumes.db"):
        self.db_path = Path(db_path)
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Ensure database exists and has required tables."""
        if not self.db_path.exists():
            st.warning(f"Database {self.db_path} not found. Creating new database.")
            # Initialize empty database with required structure
            self._create_empty_database()

    def _create_empty_database(self):
        """Create empty database with required structure."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create basic structure (simplified for demo)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_analyses (
                    analysis_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    analysis_timestamp DATETIME NOT NULL,
                    file_hash TEXT,
                    overall_score REAL,
                    early_talent_suitability REAL,
                    leadership_potential REAL,
                    confidence_score REAL,
                    career_progression_analysis TEXT,
                    processing_metadata TEXT
                )
            """
            )
            conn.commit()

    def get_all_resume_analyses(self) -> List[Dict[str, Any]]:
        """Get all resume analyses from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT analysis_id, full_name, analysis_timestamp, 
                           overall_score, early_talent_suitability, leadership_potential,
                           confidence_score
                    FROM resume_analyses
                    ORDER BY analysis_timestamp DESC
                """
                )

                results = cursor.fetchall()
                analyses = []

                for row in results:
                    analyses.append(
                        {
                            "analysis_id": row[0],
                            "full_name": row[1],
                            "analysis_timestamp": row[2],
                            "overall_score": row[3] if row[3] else 0,
                            "early_talent_suitability": row[4] if row[4] else 0,
                            "leadership_potential": row[5] if row[5] else 0,
                            "confidence_score": row[6] if row[6] else 0,
                        }
                    )

                return analyses

        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return []

    def get_candidate_summary(self, full_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for specific candidate."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT analysis_id, full_name, analysis_timestamp,
                           overall_score, early_talent_suitability, leadership_potential,
                           confidence_score, career_progression_analysis
                    FROM resume_analyses
                    WHERE full_name = ?
                    ORDER BY analysis_timestamp DESC
                    LIMIT 1
                """,
                    (full_name,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "analysis_id": row[0],
                        "full_name": row[1],
                        "analysis_timestamp": row[2],
                        "overall_score": row[3] if row[3] else 0,
                        "early_talent_suitability": row[4] if row[4] else 0,
                        "leadership_potential": row[5] if row[5] else 0,
                        "confidence_score": row[6] if row[6] else 0,
                        "career_progression_analysis": row[7]
                        or "No analysis available",
                    }
                return None

        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return None

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total count
                cursor.execute("SELECT COUNT(*) FROM resume_analyses")
                total_count = cursor.fetchone()[0]

                # Get average scores
                cursor.execute(
                    """
                    SELECT AVG(overall_score), AVG(early_talent_suitability), 
                           AVG(leadership_potential), AVG(confidence_score)
                    FROM resume_analyses 
                    WHERE overall_score IS NOT NULL
                """
                )
                avg_row = cursor.fetchone()

                return {
                    "total_analyses": total_count,
                    "avg_overall_score": avg_row[0] if avg_row[0] else 0,
                    "avg_early_talent_score": avg_row[1] if avg_row[1] else 0,
                    "avg_leadership_score": avg_row[2] if avg_row[2] else 0,
                    "avg_confidence_score": avg_row[3] if avg_row[3] else 0,
                }

        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return {
                "total_analyses": 0,
                "avg_overall_score": 0,
                "avg_early_talent_score": 0,
                "avg_leadership_score": 0,
                "avg_confidence_score": 0,
            }


# ========================================================================================
# SYSTEM INITIALIZATION
# ========================================================================================


@st.cache_resource
def initialize_system():
    """Initialize the integrated system with router and evaluation."""
    try:
        # Load configuration
        config = load_config()
        st.session_state.config = config

        # Initialize LangGraph orchestrator
        orchestrator = None
        if LANGGRAPH_AVAILABLE:
            orchestrator = LangGraphWorkflowOrchestrator(config)
            status = orchestrator.get_workflow_status()

            if not status["workflow_compiled"]:
                st.error("Failed to compile LangGraph workflow")
                return None, None
        else:
            st.warning("LangGraph not available - using fallback mode")

        # Initialize database manager
        db_manager = ScalableDatabaseManager()
        st.session_state.database_connected = True

        return orchestrator, db_manager

    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return None, None


# ========================================================================================
# AUTHENTICATION FUNCTIONS
# ========================================================================================


def authenticate_user(username: str, password: str) -> tuple:
    """Authenticate user and return role."""
    if username in USERS_DB:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if USERS_DB[username]["password"] == password_hash:
            return True, USERS_DB[username]["role"], USERS_DB[username]["full_name"]
    return False, None, None


def login_page():
    """Display login page."""
    st.markdown(
        '<h1 class="main-header">SAP Early Talent Development Platform</h1>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Welcome Back!")
        st.markdown("**Version 2.0** - Enhanced with AI Router & Quality Evaluation")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            col_login, col_demo = st.columns(2)

            with col_login:
                submit = st.form_submit_button(
                    "Login", type="primary", use_container_width=True
                )

            with col_demo:
                demo = st.form_submit_button("Demo Access", use_container_width=True)

            if submit:
                if username and password:
                    authenticated, role, full_name = authenticate_user(
                        username, password
                    )
                    if authenticated:
                        st.session_state.logged_in = True
                        st.session_state.user_role = role
                        st.session_state.username = username
                        st.session_state.full_name = full_name
                        st.success(f"Welcome back, {full_name}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")

            if demo:
                st.session_state.logged_in = True
                st.session_state.user_role = "HR"
                st.session_state.username = "demo_user"
                st.session_state.full_name = "Demo User"
                st.info("Logged in with demo access (HR role)")
                time.sleep(1)
                st.rerun()

        # Show system status
        with st.expander("System Status"):
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                st.write(f"🤖 LangGraph: {'✅' if LANGGRAPH_AVAILABLE else '❌'}")
                st.write(f"🧠 LangChain: {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
            with col_status2:
                st.write(f"📊 DeepEval: {'✅' if DEEPEVAL_AVAILABLE else '❌'}")
                st.write(
                    f"💾 Database: {'✅' if Path('test_resumes.db').exists() else '❌'}"
                )

        # Show demo credentials
        with st.expander("Demo Credentials"):
            st.markdown(
                """
            **HR Users:**
            - Username: `hr_admin` | Password: `admin123`
            - Username: `hr_manager` | Password: `manager123`
            
            **Early Talent Users:**
            - Username: `john_doe` | Password: `talent123`
            - Username: `jane_smith` | Password: `talent456`
            """
            )


# ========================================================================================
# ENHANCED CHAT INTERFACE WITH ROUTER AND EVALUATION
# ========================================================================================


def enhanced_chat_interface(orchestrator: LangGraphWorkflowOrchestrator):
    """Enhanced chat interface using the router and evaluation system."""

    st.header("🤖 AI Assistant with Smart Routing")
    st.markdown(
        "Ask me anything! I'll automatically route your query to the right specialist and evaluate the response quality."
    )

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><b>You:</b><br>{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Display AI response with evaluation metrics
                response_content = message["content"]

                st.markdown(
                    f'<div class="chat-message bot-message"><b>AI Assistant:</b><br>{response_content}</div>',
                    unsafe_allow_html=True,
                )

                # Show evaluation metrics if available
                if "evaluation" in message:
                    evaluation = message["evaluation"]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        quality_score = evaluation.get("overall_quality", 0)
                        st.metric("Quality", f"{quality_score:.2f}")
                    with col2:
                        faithfulness = evaluation.get("faithfulness")
                        if faithfulness is not None:
                            st.metric("Faithfulness", f"{faithfulness:.2f}")
                        else:
                            st.metric("Faithfulness", "N/A")
                    with col3:
                        relevancy = evaluation.get("relevancy")
                        if relevancy is not None:
                            st.metric("Relevancy", f"{relevancy:.2f}")
                        else:
                            st.metric("Relevancy", "N/A")
                    with col4:
                        precision = evaluation.get("precision")
                        if precision is not None:
                            st.metric(
                                "Precision", f"{1-precision:.2f}"
                            )  # Invert hallucination
                        else:
                            st.metric("Precision", "N/A")

                # Show metadata if available
                if "metadata" in message:
                    metadata = message["metadata"]
                    with st.expander("Response Details"):
                        st.json(
                            {
                                "Classification": message.get(
                                    "classification", "Unknown"
                                ),
                                "Workflow": message.get("workflow_type", "Unknown"),
                                "Processing Time": f"{metadata.get('processing_time', 0):.2f}s",
                                "Confidence": message.get("confidence", 0),
                            }
                        )

    # Chat input
    with st.form("enhanced_chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question:",
            placeholder="e.g., What are the company's core values? or Help me improve my leadership skills?",
        )

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit = st.form_submit_button("Send", type="primary")
        with col2:
            clear_chat = st.form_submit_button("Clear Chat")

        if clear_chat:
            st.session_state.messages = []
            st.rerun()

        if submit and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Process through orchestrator
            if orchestrator:
                with st.spinner("🤖 AI is processing your request..."):
                    try:
                        # Get response from orchestrator
                        result = orchestrator.process_query(user_input)

                        # Extract response components
                        answer = result.get(
                            "answer", "I apologize, but I encountered an error."
                        )
                        classification = result.get("query_classification", "Unknown")
                        workflow_type = result.get("workflow_type", "Unknown")
                        confidence = result.get("confidence", 0.0)
                        evaluation = result.get("evaluation", {})
                        metadata = result.get("metadata", {})

                        # Add AI response with full metadata
                        ai_message = {
                            "role": "assistant",
                            "content": answer,
                            "classification": classification,
                            "workflow_type": workflow_type,
                            "confidence": confidence,
                            "evaluation": evaluation,
                            "metadata": metadata,
                        }

                        st.session_state.messages.append(ai_message)

                        # Show processing summary
                        st.success(
                            f"✅ Query classified as: **{classification}** | Processed by: **{workflow_type}** | Quality: **{evaluation.get('overall_quality', 0):.2f}**"
                        )

                    except Exception as e:
                        error_message = f"I encountered an error: {str(e)}"
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_message,
                                "classification": "ERROR",
                                "workflow_type": "error_handler",
                                "confidence": 0.0,
                                "evaluation": {"overall_quality": 0.0},
                                "metadata": {"error": str(e)},
                            }
                        )
                        st.error(f"Error: {e}")
            else:
                # Fallback when orchestrator not available
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "The AI system is currently initializing. Please try again in a moment.",
                        "classification": "SYSTEM_UNAVAILABLE",
                        "workflow_type": "fallback",
                        "confidence": 0.0,
                        "evaluation": {"overall_quality": 0.0},
                        "metadata": {},
                    }
                )

            st.rerun()

    # Quick questions with routing examples
    st.markdown("### Quick Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🏢 Company Values", help="This will be routed to Onboarding workflow"
        ):
            question = "What are the company's core values?"
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

    with col2:
        if st.button(
            "🚀 Leadership Skills", help="This will be routed to Re/Upskilling workflow"
        ):
            question = "How can I improve my leadership skills?"
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

    with col3:
        if st.button(
            "💡 General Question", help="This will be routed to General workflow"
        ):
            question = "Explain machine learning algorithms"
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()


# ========================================================================================
# EARLY TALENT INTERFACE
# ========================================================================================


def early_talent_interface(orchestrator, db_manager):
    """Interface for early talent users."""
    st.markdown(
        '<h1 class="main-header">Early Talent Development Portal</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar with user info
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.full_name}!")
        st.markdown(f"**Role:** Early Talent")

        if st.session_state.username in USERS_DB:
            user_data = USERS_DB[st.session_state.username]
            st.markdown(f"**Department:** {user_data.get('department', 'N/A')}")
            st.markdown(f"**Manager:** {user_data.get('manager', 'N/A')}")
            st.markdown(f"**Start Date:** {user_data.get('start_date', 'N/A')}")

        # Show personal resume data if available
        candidate_data = db_manager.get_candidate_summary(st.session_state.full_name)
        if candidate_data:
            st.markdown("---")
            st.markdown("### Your Profile")
            st.metric("Overall Score", f"{candidate_data['overall_score']:.1f}/100")
            st.metric(
                "Early Talent Fit",
                f"{candidate_data['early_talent_suitability']:.1f}/100",
            )
            st.metric(
                "Leadership Potential",
                f"{candidate_data['leadership_potential']:.1f}/100",
            )

        st.markdown("---")
        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🤖 AI Assistant", "📊 My Progress", "📋 Resources"])

    with tab1:
        enhanced_chat_interface(orchestrator)

    with tab2:
        st.header("Performance Dashboard")

        # Show personal data if available
        if candidate_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{candidate_data['overall_score']:.1f}/100")
            with col2:
                st.metric(
                    "Early Talent Suitability",
                    f"{candidate_data['early_talent_suitability']:.1f}/100",
                )
            with col3:
                st.metric(
                    "Leadership Potential",
                    f"{candidate_data['leadership_potential']:.1f}/100",
                )

            # Career progression analysis
            if candidate_data["career_progression_analysis"]:
                with st.expander("Career Analysis"):
                    st.markdown(candidate_data["career_progression_analysis"])
        else:
            st.info("Complete your profile assessment to see personalized metrics!")

        # Mock progress chart
        dates = pd.date_range(start="2024-01", periods=12, freq="ME")
        progress_data = pd.DataFrame(
            {
                "Month": dates,
                "Score": [50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78],
            }
        )

        fig = px.line(
            progress_data,
            x="Month",
            y="Score",
            title="Development Progress Over Time",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Resources & Quick Links")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📚 Learning Resources")
            resources = [
                "Employee Handbook",
                "IT Security Guidelines",
                "Communication Best Practices",
                "Project Management Basics",
                "SAP Product Overview",
            ]
            for resource in resources:
                if st.button(f"📄 {resource}", key=f"resource_{resource}"):
                    st.info(f"Opening {resource}...")

        with col2:
            st.subheader("🎯 Quick Links")
            links = [
                ("SAP Learning Hub", "https://learning.sap.com"),
                ("Internal Wiki", "https://wiki.sap.com"),
                ("HR Portal", "https://hr.sap.com"),
                ("IT Support", "https://it.sap.com"),
                ("Feedback Form", "https://feedback.sap.com"),
            ]
            for name, url in links:
                st.markdown(f"🔗 [{name}]({url})")


# ========================================================================================
# HR INTERFACE WITH DATABASE INTEGRATION
# ========================================================================================


def hr_interface(orchestrator, db_manager):
    """Interface for HR users with database integration."""
    st.markdown(
        '<h1 class="main-header">HR Management Dashboard</h1>', unsafe_allow_html=True
    )

    # Sidebar with database stats
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.full_name}!")
        st.markdown(f"**Role:** HR Administrator")

        # Get database statistics
        db_stats = db_manager.get_database_stats()
        st.markdown("---")
        st.markdown("### Database Stats")
        st.metric("Total Analyses", db_stats["total_analyses"])
        st.metric("Avg Overall Score", f"{db_stats['avg_overall_score']:.1f}/100")
        st.metric(
            "Avg Early Talent Fit", f"{db_stats['avg_early_talent_score']:.1f}/100"
        )

        st.markdown("---")
        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "🤖 AI Assistant", "👥 Resume Database", "⚙️ System"]
    )

    with tab1:
        st.header("HR Analytics Dashboard")

        # Key metrics from database
        db_stats = db_manager.get_database_stats()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", db_stats["total_analyses"])
        with col2:
            st.metric("Avg Overall Score", f"{db_stats['avg_overall_score']:.1f}/100")
        with col3:
            st.metric(
                "Avg Early Talent Fit", f"{db_stats['avg_early_talent_score']:.1f}/100"
            )
        with col4:
            st.metric(
                "Avg Leadership Potential",
                f"{db_stats['avg_leadership_score']:.1f}/100",
            )

        # Recent analyses
        st.subheader("Recent Resume Analyses")
        analyses = db_manager.get_all_resume_analyses()

        if analyses:
            # Convert to DataFrame for display
            df = pd.DataFrame(analyses)
            df["analysis_timestamp"] = pd.to_datetime(df["analysis_timestamp"])

            # Display recent analyses
            recent_df = df.head(10)
            st.dataframe(
                recent_df[
                    [
                        "full_name",
                        "analysis_timestamp",
                        "overall_score",
                        "early_talent_suitability",
                        "leadership_potential",
                    ]
                ],
                use_container_width=True,
            )

            # Score distribution chart
            if len(df) > 1:
                fig = px.histogram(
                    df,
                    x="overall_score",
                    title="Distribution of Overall Scores",
                    nbins=10,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resume analyses found in database.")

    with tab2:
        enhanced_chat_interface(orchestrator)

    with tab3:
        st.header("Resume Analysis Database")

        # Get all analyses
        analyses = db_manager.get_all_resume_analyses()

        if analyses:
            # Convert to DataFrame
            df = pd.DataFrame(analyses)
            df["analysis_timestamp"] = pd.to_datetime(df["analysis_timestamp"])

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_score = st.slider("Minimum Overall Score", 0, 100, 0)
            with col2:
                search_name = st.text_input("Search by Name")

            # Apply filters
            filtered_df = df.copy()
            if min_score > 0:
                filtered_df = filtered_df[filtered_df["overall_score"] >= min_score]
            if search_name:
                filtered_df = filtered_df[
                    filtered_df["full_name"].str.contains(
                        search_name, case=False, na=False
                    )
                ]

            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)

            # Export functionality
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"resume_analyses_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

            # Detailed view
            if not filtered_df.empty:
                st.subheader("Detailed View")
                selected_candidate = st.selectbox(
                    "Select candidate for detailed view",
                    options=filtered_df["full_name"].tolist(),
                )

                if selected_candidate:
                    candidate_data = db_manager.get_candidate_summary(
                        selected_candidate
                    )
                    if candidate_data:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Overall Score",
                                f"{candidate_data['overall_score']:.1f}/100",
                            )
                        with col2:
                            st.metric(
                                "Early Talent Suitability",
                                f"{candidate_data['early_talent_suitability']:.1f}/100",
                            )
                        with col3:
                            st.metric(
                                "Leadership Potential",
                                f"{candidate_data['leadership_potential']:.1f}/100",
                            )

                        # Career analysis
                        if candidate_data["career_progression_analysis"]:
                            st.subheader("Career Progression Analysis")
                            st.markdown(candidate_data["career_progression_analysis"])
        else:
            st.info("No resume analyses found. Upload some resumes to see data here.")

    with tab4:
        st.header("System Configuration & Status")

        # System status
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("System Components")
            status_items = [
                (
                    "LangGraph Router",
                    LANGGRAPH_AVAILABLE,
                    "✅" if LANGGRAPH_AVAILABLE else "❌",
                ),
                (
                    "LangChain",
                    LANGCHAIN_AVAILABLE,
                    "✅" if LANGCHAIN_AVAILABLE else "❌",
                ),
                ("DeepEval", DEEPEVAL_AVAILABLE, "✅" if DEEPEVAL_AVAILABLE else "❌"),
                (
                    "Database",
                    st.session_state.database_connected,
                    "✅" if st.session_state.database_connected else "❌",
                ),
            ]

            for name, status, icon in status_items:
                col_name, col_status = st.columns([3, 1])
                with col_name:
                    st.write(name)
                with col_status:
                    st.write(f"{icon} {'Active' if status else 'Inactive'}")

        with col2:
            st.subheader("Configuration")
            if st.session_state.config:
                config = st.session_state.config
                st.json(
                    {
                        "Model": config.OpenAIModel,
                        "Temperature": config.ModelTemperature,
                        "Max Tokens": config.MaxTokens,
                        "Evaluation Enabled": config.EnableEvaluation,
                    }
                )

        # Database management
        st.subheader("Database Management")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Refresh Database Stats"):
                st.rerun()

        with col2:
            if st.button("Initialize Sample Data"):
                create_sample_onboarding_docs()
                create_sample_leadership_docs()
                st.success("Sample data initialized!")

        with col3:
            if st.button("Export Database", help="Export full database"):
                st.info("Database export functionality would be implemented here.")


# ========================================================================================
# MAIN APPLICATION
# ========================================================================================


def main():
    """Main application entry point."""

    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        # Initialize system if not already done
        if st.session_state.orchestrator is None:
            with st.spinner("🚀 Initializing SAP HCM System..."):
                orchestrator, db_manager = initialize_system()
                st.session_state.orchestrator = orchestrator
                st.session_state.db_manager = db_manager

                if orchestrator is None:
                    st.error("Failed to initialize system. Please check configuration.")
                    return
        else:
            orchestrator = st.session_state.orchestrator
            db_manager = st.session_state.db_manager

        # Show appropriate interface based on role
        if st.session_state.user_role == "HR":
            hr_interface(orchestrator, db_manager)
        elif st.session_state.user_role == "EARLY_TALENT":
            early_talent_interface(orchestrator, db_manager)
        else:
            st.error("Invalid user role")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.session_state.username = None
                st.rerun()


if __name__ == "__main__":
    main()
