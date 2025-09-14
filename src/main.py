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
from typing import Dict, List, Optional, Any

# Import existing components
from onboarding_component import (
    OnboardingWorkflow,
    OnboardingConfig,
    create_sample_onboarding_docs,
)
from re_upskilling_component import (
    LeadershipDevelopmentWorkflow,
    LeadershipConfig,
    create_sample_leadership_docs,
    run_sample_assessment,
)
from reader_component import (
    ReaderWorkflow,
    ReaderConfig,
)

# Import main system components
from tabnanny import verbose
from typing import Dict, List, Optional, TypedDict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from chromadb import QueryResult

# Optional imports with fallback handling
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.schema import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain not available - using mock implementations")
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver

    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("LangGraph not available - using simplified workflow")
    LANGGRAPH_AVAILABLE = False

try:
    from deepeval import evaluate
    from deepeval.models import GPTModel
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except ImportError:
    print("DeepEval not available - skipping evaluation")
    DEEPEVAL_AVAILABLE = False

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
        "About": "# SAP HCM System\nEarly Talent Development Platform\n\nVersion 1.0",
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
    
    .login-container {
        max-width: 400px;
        margin: auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .report-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
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

if "early_talents" not in st.session_state:
    st.session_state.early_talents = []

if "workflows_initialized" not in st.session_state:
    st.session_state.workflows_initialized = False

if "config_loaded" not in st.session_state:
    st.session_state.config_loaded = False

# ========================================================================================
# CONFIGURATION MANAGEMENT
# ========================================================================================


@dataclass
class SystemConfig:
    """System configuration for the HCM system."""

    OpenAIAPIKey: str
    OpenAIModel: str = "gpt-4o-mini"
    EmbeddingModel: str = "text-embedding-ada-002"
    ModelTemperature: float = 0.7
    MaxTokens: int = 4096
    EnableEvaluation: bool = True
    DebugModeOn: bool = False


def load_config(config_path: str = "config.json") -> SystemConfig:
    """Load system configuration from JSON file."""
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        api_key = config.get("OpenAIAPIKey", "")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return SystemConfig(**config)
    except FileNotFoundError:
        # Create default config
        default_config = {
            "OpenAIAPIKey": "",
            "OpenAIModel": "gpt-4o-mini",
            "EmbeddingModel": "text-embedding-ada-002",
            "ModelTemperature": 0.7,
            "MaxTokens": 4096,
            "EnableEvaluation": True,
            "DebugModeOn": False,
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return SystemConfig(**default_config)


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
        st.markdown("Please login to continue")

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
                # Demo access as HR
                st.session_state.logged_in = True
                st.session_state.user_role = "HR"
                st.session_state.username = "demo_user"
                st.session_state.full_name = "Demo User"
                st.info("Logged in with demo access (HR role)")
                time.sleep(1)
                st.rerun()

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
# WORKFLOW INITIALIZATION
# ========================================================================================


@st.cache_resource
def initialize_workflows():
    """Initialize all workflow components."""
    config = load_config()

    # Initialize LLM
    if LANGCHAIN_AVAILABLE and config.OpenAIAPIKey:
        llm = ChatOpenAI(
            model=config.OpenAIModel,
            temperature=config.ModelTemperature,
            max_tokens=config.MaxTokens,
            api_key=config.OpenAIAPIKey,
        )
        embeddings = OpenAIEmbeddings(api_key=config.OpenAIAPIKey)
    else:
        # Use mock implementations
        from onboarding_component import MockLLM

        llm = MockLLM()

        class MockEmbeddings:
            def __init__(self):
                self.model_name = "mock-embeddings"

        embeddings = MockEmbeddings()

    # Initialize workflows
    workflows = {}

    try:
        # Onboarding workflow
        onboarding_config = OnboardingConfig()
        workflows["onboarding"] = OnboardingWorkflow(llm, embeddings, onboarding_config)

        # Leadership development workflow
        leadership_config = LeadershipConfig()
        workflows["leadership"] = LeadershipDevelopmentWorkflow(
            llm, embeddings, leadership_config
        )

        # Resume reader workflow
        reader_config = ReaderConfig()
        workflows["reader"] = ReaderWorkflow(llm, reader_config)

    except Exception as e:
        st.error(f"Error initializing workflows: {e}")
        return None

    return workflows


# ========================================================================================
# REPORT GENERATOR
# ========================================================================================


class EmployerReportGenerator:
    """Generate comprehensive employer reports for early talents."""

    def __init__(self, workflows):
        self.workflows = workflows

    def generate_report(self, employee_id: str, employee_data: dict) -> dict:
        """Generate comprehensive report for an employee."""
        report = {
            "employee_id": employee_id,
            "full_name": employee_data.get("full_name", "Unknown"),
            "department": employee_data.get("department", "Unknown"),
            "start_date": employee_data.get("start_date", "Unknown"),
            "report_date": datetime.now().isoformat(),
            "sections": {},
        }

        # Section 1: Basic Information
        report["sections"]["basic_info"] = {
            "title": "Employee Information",
            "content": {
                "Name": employee_data.get("full_name"),
                "Department": employee_data.get("department"),
                "Start Date": employee_data.get("start_date"),
                "Manager": employee_data.get("manager"),
                "Role": "Early Talent",
                "Experience Level": self._calculate_experience(
                    employee_data.get("start_date")
                ),
            },
        }

        # Section 2: Onboarding Progress
        report["sections"]["onboarding"] = {
            "title": "Onboarding Progress",
            "content": self._get_onboarding_progress(employee_id),
        }

        # Section 3: Skills Assessment
        report["sections"]["skills"] = {
            "title": "Skills Development",
            "content": self._get_skills_assessment(employee_id),
        }

        # Section 4: Leadership Development
        report["sections"]["leadership"] = {
            "title": "Leadership Development",
            "content": self._get_leadership_progress(employee_id),
        }

        # Section 5: Performance Metrics
        report["sections"]["performance"] = {
            "title": "Performance Metrics",
            "content": self._get_performance_metrics(employee_id),
        }

        # Section 6: Recommendations
        report["sections"]["recommendations"] = {
            "title": "Development Recommendations",
            "content": self._get_recommendations(employee_id),
        }

        return report

    def _calculate_experience(self, start_date: str) -> str:
        """Calculate experience level based on start date."""
        if not start_date:
            return "Unknown"

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            months = (datetime.now() - start).days / 30

            if months < 6:
                return f"{int(months)} months (New Hire)"
            elif months < 12:
                return f"{int(months)} months (Junior)"
            elif months < 24:
                return f"{int(months)} months (Developing)"
            elif months < 36:
                return f"{int(months)} months (Experienced)"
            else:
                return f"{int(months/12)} years (Senior)"
        except:
            return "Unknown"

    def _get_onboarding_progress(self, employee_id: str) -> dict:
        """Get onboarding progress for employee."""
        # Simulated data - in production, this would query actual data
        return {
            "Overall Progress": "75%",
            "Completed Modules": [
                "Company Culture",
                "IT Setup",
                "HR Policies",
                "Team Introduction",
            ],
            "In Progress": ["Product Training", "Role-Specific Training"],
            "Pending": ["Advanced Skills Training"],
            "Estimated Completion": "2 weeks",
        }

    def _get_skills_assessment(self, employee_id: str) -> dict:
        """Get skills assessment for employee."""
        return {
            "Technical Skills": {
                "Python": "Advanced",
                "Data Analysis": "Intermediate",
                "Machine Learning": "Beginner",
                "Cloud Computing": "Intermediate",
            },
            "Soft Skills": {
                "Communication": "Advanced",
                "Teamwork": "Advanced",
                "Problem Solving": "Intermediate",
                "Leadership": "Developing",
            },
            "Improvement Areas": [
                "Project Management",
                "Strategic Thinking",
                "Public Speaking",
            ],
        }

    def _get_leadership_progress(self, employee_id: str) -> dict:
        """Get leadership development progress."""
        if self.workflows and "leadership" in self.workflows:
            try:
                progress = self.workflows["leadership"].get_user_progress(employee_id)
                return progress
            except:
                pass

        # Fallback data
        return {
            "Leadership Score": "65/100",
            "Core Competencies": {
                "Communication": "70%",
                "Decision Making": "60%",
                "Team Building": "65%",
                "Strategic Thinking": "55%",
                "Emotional Intelligence": "75%",
            },
            "Recent Achievements": [
                "Completed Team Lead Training",
                "Led first project successfully",
            ],
            "Next Milestone": "Advanced Leadership Workshop",
        }

    def _get_performance_metrics(self, employee_id: str) -> dict:
        """Get performance metrics."""
        return {
            "Overall Rating": "Exceeds Expectations",
            "Key Metrics": {
                "Task Completion Rate": "95%",
                "Quality Score": "88%",
                "Collaboration Rating": "4.5/5",
                "Innovation Index": "7/10",
            },
            "Strengths": ["Attention to detail", "Team collaboration", "Quick learner"],
            "Growth Areas": ["Time management", "Stakeholder communication"],
        }

    def _get_recommendations(self, employee_id: str) -> dict:
        """Get development recommendations."""
        return {
            "Immediate Actions": [
                "Enroll in Advanced Python course",
                "Shadow senior team member for 2 weeks",
                "Join cross-functional project team",
            ],
            "3-Month Goals": [
                "Complete leadership certification",
                "Lead a small team project",
                "Present at department meeting",
            ],
            "6-Month Goals": [
                "Take ownership of a key initiative",
                "Mentor a new team member",
                "Contribute to strategic planning",
            ],
            "Career Path": "Software Engineer → Senior Engineer → Team Lead → Engineering Manager",
        }

    def format_report_as_markdown(self, report: dict) -> str:
        """Format report as markdown for download."""
        md = f"# Employee Development Report\n\n"
        md += f"**Generated on:** {report['report_date']}\n\n"
        md += "---\n\n"

        for section_key, section in report["sections"].items():
            md += f"## {section['title']}\n\n"
            md += self._format_section_content(section["content"])
            md += "\n---\n\n"

        return md

    def _format_section_content(self, content: Any, indent: int = 0) -> str:
        """Recursively format section content."""
        md = ""
        indent_str = "  " * indent

        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, (dict, list)):
                    md += f"{indent_str}**{key}:**\n"
                    md += self._format_section_content(value, indent + 1)
                else:
                    md += f"{indent_str}- **{key}:** {value}\n"
        elif isinstance(content, list):
            for item in content:
                md += f"{indent_str}- {item}\n"
        else:
            md += f"{indent_str}{content}\n"

        return md


# ========================================================================================
# EARLY TALENT INTERFACE
# ========================================================================================


def early_talent_interface(workflows):
    """Interface for early talent users."""
    st.markdown(
        '<h1 class="main-header">Early Talent Development Portal</h1>',
        unsafe_allow_html=True,
    )

    # User info in sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.full_name}!")
        st.markdown(f"**Role:** Early Talent")

        if st.session_state.username in USERS_DB:
            user_data = USERS_DB[st.session_state.username]
            st.markdown(f"**Department:** {user_data.get('department', 'N/A')}")
            st.markdown(f"**Manager:** {user_data.get('manager', 'N/A')}")
            st.markdown(f"**Start Date:** {user_data.get('start_date', 'N/A')}")

        st.markdown("---")

        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🤖 Onboarding Buddy", "📚 Learning Path", "📊 My Progress", "📋 Resources"]
    )

    with tab1:
        st.header("AI Onboarding Assistant")
        st.markdown(
            "Ask me anything about company policies, procedures, or your onboarding journey!"
        )

        # Chat interface
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message"><b>You:</b><br>{message["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message bot-message"><b>AI Assistant:</b><br>{message["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your question:",
                placeholder="e.g., What are the company's core values?",
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.form_submit_button("Send", type="primary")

            if submit and user_input:
                # Add user message
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )

                # Get AI response
                if workflows and "onboarding" in workflows:
                    with st.spinner("AI is thinking..."):
                        response = workflows["onboarding"].handle_query(user_input)
                        ai_response = response.get(
                            "answer",
                            "I apologize, but I encountered an error processing your question.",
                        )
                else:
                    ai_response = "The onboarding system is currently initializing. Please try again in a moment."

                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )
                st.rerun()

        # Quick questions
        st.markdown("### Quick Questions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Company Values"):
                question = "What are the company's core values?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

        with col2:
            if st.button("Benefits Info"):
                question = "What benefits are available to employees?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

        with col3:
            if st.button("IT Security"):
                question = "What are the IT security policies?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

    with tab2:
        st.header("Your Learning Path")

        # Learning progress overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Progress", "65%", "↑ 5%")
        with col2:
            st.metric("Completed Modules", "8", "↑ 2")
        with col3:
            st.metric("Hours Learned", "24", "↑ 4")
        with col4:
            st.metric("Certifications", "2", "↑ 1")

        # Current focus areas
        st.subheader("Current Focus Areas")
        focus_areas = [
            "Communication Skills",
            "Technical Excellence",
            "Team Collaboration",
        ]
        for area in focus_areas:
            with st.expander(area):
                st.progress(0.7)
                st.markdown("**Current Level:** Intermediate")
                st.markdown("**Target Level:** Advanced")
                st.markdown("**Recommended Actions:**")
                st.markdown("- Complete online course on advanced communication")
                st.markdown("- Practice in team meetings")
                st.markdown("- Get feedback from mentor")

    with tab3:
        st.header("Performance Dashboard")

        # Performance metrics
        col1, col2 = st.columns(2)

        with col1:
            # Skills radar chart
            categories = [
                "Technical",
                "Communication",
                "Leadership",
                "Problem Solving",
                "Teamwork",
            ]
            values = [75, 85, 60, 80, 90]

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=values, theta=categories, fill="toself", name="Current Skills"
                )
            )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Skills Assessment",
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Progress over time
            dates = pd.date_range(start="2024-01", periods=12, freq="M")
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

        # Recent achievements
        st.subheader("Recent Achievements")
        achievements = [
            {
                "date": "2024-11-01",
                "achievement": "Completed Python Advanced Course",
                "badge": "🏆",
            },
            {
                "date": "2024-10-15",
                "achievement": "Led first team meeting",
                "badge": "🌟",
            },
            {
                "date": "2024-10-01",
                "achievement": "Received positive feedback from mentor",
                "badge": "👏",
            },
        ]

        for ach in achievements:
            st.success(f"{ach['badge']} **{ach['date']}:** {ach['achievement']}")

    with tab4:
        st.header("Resources & Documents")

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
# HR INTERFACE
# ========================================================================================


def hr_interface(workflows):
    """Interface for HR users."""
    st.markdown(
        '<h1 class="main-header">HR Management Dashboard</h1>', unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.full_name}!")
        st.markdown(f"**Role:** HR Administrator")
        st.markdown("---")

        # Quick stats
        st.metric("Total Early Talents", len(st.session_state.early_talents))
        st.metric("Active Onboarding", "12")
        st.metric("Completion Rate", "78%")

        st.markdown("---")

        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Dashboard",
            "➕ Register Talent",
            "📈 Reports",
            "👥 Talent Overview",
            "⚙️ Settings",
        ]
    )

    with tab1:
        st.header("HR Analytics Dashboard")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", "156", "↑ 12")
        with col2:
            st.metric("Early Talents", "42", "↑ 5")
        with col3:
            st.metric("Avg. Onboarding Time", "3.2 weeks", "↓ 0.3")
        with col4:
            st.metric("Retention Rate", "94%", "↑ 2%")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Department distribution
            dept_data = pd.DataFrame(
                {
                    "Department": [
                        "Engineering",
                        "Product",
                        "Sales",
                        "Marketing",
                        "HR",
                    ],
                    "Count": [45, 30, 25, 20, 15],
                }
            )
            fig = px.pie(
                dept_data,
                values="Count",
                names="Department",
                title="Early Talent Distribution by Department",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Hiring trend
            months = pd.date_range(start="2024-01", periods=12, freq="M")
            hiring_data = pd.DataFrame(
                {"Month": months, "Hires": [5, 7, 6, 8, 10, 9, 11, 12, 10, 8, 9, 11]}
            )
            fig = px.bar(
                hiring_data, x="Month", y="Hires", title="Monthly Hiring Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent activities
        st.subheader("Recent Activities")
        activities = [
            {
                "time": "2 hours ago",
                "action": "John Doe completed onboarding module",
                "type": "success",
            },
            {
                "time": "4 hours ago",
                "action": "New talent Jane Smith registered",
                "type": "info",
            },
            {
                "time": "1 day ago",
                "action": "Performance review scheduled for 5 employees",
                "type": "warning",
            },
        ]

        for activity in activities:
            if activity["type"] == "success":
                st.success(f"⏰ {activity['time']}: {activity['action']}")
            elif activity["type"] == "info":
                st.info(f"⏰ {activity['time']}: {activity['action']}")
            else:
                st.warning(f"⏰ {activity['time']}: {activity['action']}")

    with tab2:
        st.header("Register New Early Talent")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Upload Resume")

            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=["pdf", "docx", "txt"],
                help="Supported formats: PDF, DOCX, TXT",
            )

            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = Path(f"temp_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if st.button("Process Resume", type="primary"):
                    if workflows and "reader" in workflows:
                        with st.spinner("Analyzing resume..."):
                            result = workflows["reader"].process_resume(str(temp_path))

                        if result.get("confidence", 0) > 0:
                            st.success("Resume processed successfully!")

                            # Display extracted information
                            st.subheader("Extracted Information")

                            extracted = result.get("extracted_data", {})
                            personal_info = extracted.get("personal_info", {})

                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.markdown("**Personal Information:**")
                                st.write(
                                    f"Name: {personal_info.get('full_name', 'N/A')}"
                                )
                                st.write(f"Email: {personal_info.get('email', 'N/A')}")
                                st.write(f"Phone: {personal_info.get('phone', 'N/A')}")

                            with col_info2:
                                st.markdown("**Profile Summary:**")
                                st.write(f"Skills: {extracted.get('skills_count', 0)}")
                                st.write(
                                    f"Experience: {extracted.get('experience_count', 0)} positions"
                                )
                                st.write(
                                    f"Education: {extracted.get('education_count', 0)} degrees"
                                )

                            # Scores
                            st.subheader("Assessment Scores")
                            col_score1, col_score2, col_score3 = st.columns(3)
                            with col_score1:
                                st.metric(
                                    "Overall Score",
                                    f"{result.get('overall_score', 0):.1f}/100",
                                )
                            with col_score2:
                                st.metric(
                                    "Early Talent Fit",
                                    f"{result.get('early_talent_suitability', 0):.1f}/100",
                                )
                            with col_score3:
                                st.metric(
                                    "Leadership Potential",
                                    f"{result.get('leadership_potential', 0):.1f}/100",
                                )

                            # Add to talent pool
                            if st.button("Add to Talent Pool", type="primary"):
                                talent_data = {
                                    "name": personal_info.get("full_name", "Unknown"),
                                    "email": personal_info.get("email", "N/A"),
                                    "analysis_id": result.get("analysis_id"),
                                    "scores": {
                                        "overall": result.get("overall_score"),
                                        "early_talent": result.get(
                                            "early_talent_suitability"
                                        ),
                                        "leadership": result.get(
                                            "leadership_potential"
                                        ),
                                    },
                                    "registered_date": datetime.now().isoformat(),
                                }
                                st.session_state.early_talents.append(talent_data)
                                st.success(
                                    f"✅ {talent_data['name']} added to talent pool!"
                                )
                        else:
                            st.error("Failed to process resume. Please try again.")
                    else:
                        st.error(
                            "Resume reader not available. Please check system configuration."
                        )

                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

        with col2:
            st.subheader("Manual Registration")

            with st.form("manual_registration"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                department = st.selectbox(
                    "Department",
                    ["Engineering", "Product", "Sales", "Marketing", "HR", "Finance"],
                )
                start_date = st.date_input("Start Date")
                manager = st.text_input("Manager Name")

                if st.form_submit_button("Register", type="primary"):
                    if name and email:
                        talent_data = {
                            "name": name,
                            "email": email,
                            "department": department,
                            "start_date": start_date.isoformat(),
                            "manager": manager,
                            "registered_date": datetime.now().isoformat(),
                        }
                        st.session_state.early_talents.append(talent_data)
                        st.success(f"✅ {name} registered successfully!")
                    else:
                        st.error("Please fill in all required fields")

    with tab3:
        st.header("Generate Employer Reports")

        # Report generator
        report_gen = EmployerReportGenerator(workflows)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Select Employee")

            # Get list of employees
            employee_options = ["John Doe", "Jane Smith"] + [
                talent["name"] for talent in st.session_state.early_talents
            ]

            selected_employee = st.selectbox("Choose an employee", employee_options)

            if st.button("Generate Report", type="primary"):
                with st.spinner(f"Generating report for {selected_employee}..."):
                    # Generate report
                    employee_data = {
                        "full_name": selected_employee,
                        "department": "Engineering",
                        "start_date": "2024-01-15",
                        "manager": "Sarah Johnson",
                    }

                    report = report_gen.generate_report(
                        selected_employee, employee_data
                    )

                    # Display report
                    st.success("Report generated successfully!")

                    # Show report sections
                    for section_key, section in report["sections"].items():
                        with st.expander(section["title"]):
                            if isinstance(section["content"], dict):
                                for key, value in section["content"].items():
                                    if isinstance(value, dict):
                                        st.markdown(f"**{key}:**")
                                        for sub_key, sub_value in value.items():
                                            st.write(f"  • {sub_key}: {sub_value}")
                                    elif isinstance(value, list):
                                        st.markdown(f"**{key}:**")
                                        for item in value:
                                            st.write(f"  • {item}")
                                    else:
                                        st.write(f"**{key}:** {value}")

                    # Download button
                    report_md = report_gen.format_report_as_markdown(report)
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=report_md,
                        file_name=f"report_{selected_employee.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                    )

        with col2:
            st.subheader("Report Settings")

            include_sections = st.multiselect(
                "Include Sections",
                [
                    "Basic Info",
                    "Onboarding",
                    "Skills",
                    "Leadership",
                    "Performance",
                    "Recommendations",
                ],
                default=[
                    "Basic Info",
                    "Onboarding",
                    "Skills",
                    "Leadership",
                    "Performance",
                    "Recommendations",
                ],
            )

            report_format = st.radio("Report Format", ["Markdown", "PDF", "HTML"])

            st.markdown("---")

            st.subheader("Bulk Reports")
            if st.button("Generate All Reports"):
                st.info(
                    f"This will generate reports for {len(employee_options)} employees"
                )

    with tab4:
        st.header("Early Talent Overview")

        # Display talent pool
        if st.session_state.early_talents:
            df = pd.DataFrame(st.session_state.early_talents)
            st.dataframe(df, use_container_width=True)

            # Filters
            st.subheader("Filter Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                dept_filter = st.multiselect(
                    "Department",
                    options=["All", "Engineering", "Product", "Sales", "Marketing"],
                )

            with col2:
                score_filter = st.slider("Min Overall Score", 0, 100, 50)

            with col3:
                date_filter = st.date_input("Registered After")

            # Actions
            st.subheader("Bulk Actions")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Send Welcome Email"):
                    st.success("Welcome emails sent to all selected talents")

            with col2:
                if st.button("Assign Training"):
                    st.info("Training modules assigned")

            with col3:
                if st.button("Export to CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV", csv, "early_talents.csv", "text/csv"
                    )
        else:
            st.info(
                "No early talents registered yet. Use the 'Register Talent' tab to add new employees."
            )

    with tab5:
        st.header("System Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Configuration")

            # Load current config
            config = load_config()

            with st.form("config_form"):
                api_key = st.text_input(
                    "OpenAI API Key",
                    value="*" * 20 if config.OpenAIAPIKey else "",
                    type="password",
                )
                model = st.selectbox(
                    "Model", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"], index=0
                )
                temperature = st.slider(
                    "Temperature", 0.0, 1.0, config.ModelTemperature
                )
                max_tokens = st.number_input("Max Tokens", 100, 8000, config.MaxTokens)

                if st.form_submit_button("Save Configuration"):
                    # Update config
                    new_config = {
                        "OpenAIAPIKey": (
                            api_key if api_key != "*" * 20 else config.OpenAIAPIKey
                        ),
                        "OpenAIModel": model,
                        "ModelTemperature": temperature,
                        "MaxTokens": max_tokens,
                        "EmbeddingModel": config.EmbeddingModel,
                        "EnableEvaluation": config.EnableEvaluation,
                        "DebugModeOn": config.DebugModeOn,
                    }

                    with open("config.json", "w") as f:
                        json.dump(new_config, f, indent=2)

                    st.success(
                        "Configuration saved! Please restart the application for changes to take effect."
                    )

        with col2:
            st.subheader("System Status")

            status_items = [
                (
                    "LangChain",
                    LANGCHAIN_AVAILABLE,
                    "✅" if LANGCHAIN_AVAILABLE else "❌",
                ),
                (
                    "LangGraph",
                    LANGGRAPH_AVAILABLE,
                    "✅" if LANGGRAPH_AVAILABLE else "❌",
                ),
                ("DeepEval", DEEPEVAL_AVAILABLE, "✅" if DEEPEVAL_AVAILABLE else "❌"),
                (
                    "Workflows",
                    st.session_state.workflows_initialized,
                    "✅" if st.session_state.workflows_initialized else "❌",
                ),
            ]

            for name, status, icon in status_items:
                col_name, col_status = st.columns([3, 1])
                with col_name:
                    st.write(name)
                with col_status:
                    st.write(f"{icon} {'Active' if status else 'Inactive'}")

            st.markdown("---")

            st.subheader("Database")
            if st.button("Initialize Sample Data"):
                create_sample_onboarding_docs()
                create_sample_leadership_docs()
                st.success("Sample data initialized!")

            if st.button("Clear All Data", type="secondary"):
                st.warning("This will clear all data. Are you sure?")


# ========================================================================================
# MAIN APPLICATION
# ========================================================================================


def main():
    """Main application entry point."""

    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        # Initialize workflows if not already done
        if not st.session_state.workflows_initialized:
            with st.spinner("Initializing system components..."):
                workflows = initialize_workflows()
                st.session_state.workflows = workflows
                st.session_state.workflows_initialized = True
        else:
            workflows = st.session_state.workflows

        # Show appropriate interface based on role
        if st.session_state.user_role == "HR":
            hr_interface(workflows)
        elif st.session_state.user_role == "EARLY_TALENT":
            early_talent_interface(workflows)
        else:
            st.error("Invalid user role")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.session_state.username = None
                st.rerun()


if __name__ == "__main__":
    main()
