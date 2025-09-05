import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

from prototype import MasterAgent

# Page configuration
st.set_page_config(
    page_title="AI Assistant Pro",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "# Modern AI Chatbot\nBuilt with Streamlit",
    },
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #9c27b0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "theme": "Light",
        "model": "GPT-4",
        "temperature": 0.7,
        "max_tokens": 150,
    }

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration Panel")

    # User profile section
    st.subheader("User Profile")
    user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if user_name:
        st.session_state.user_name = user_name

    user_avatar = st.file_uploader("Upload Avatar", type=["png", "jpg", "jpeg"])

    # Model settings
    st.subheader("AI Model Settings")
    model = st.selectbox(
        "Select Model", ["GPT-4", "GPT-3.5", "Claude", "Llama-2"], index=0
    )

    temperature = st.slider(
        "Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.7, step=0.1
    )

    max_tokens = st.number_input(
        "Max Response Length", min_value=50, max_value=500, value=150, step=25
    )

    # Theme settings
    st.subheader("Appearance")
    theme = st.radio("Theme", ["Light", "Dark", "Auto"])

    # Language settings
    language = st.selectbox(
        "Language", ["English", "Spanish", "French", "German", "Chinese"]
    )

    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        enable_memory = st.checkbox("Enable Conversation Memory", value=True)
        enable_suggestions = st.checkbox("Show Response Suggestions", value=True)
        response_speed = st.select_slider(
            "Response Speed", options=["Slow", "Medium", "Fast"], value="Medium"
        )

    # Save settings button
    if st.button("Save Settings", type="primary"):
        st.session_state.settings.update(
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "theme": theme,
                "language": language,
            }
        )
        st.success("Settings saved!")

    # Clear chat button
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Main page header
st.markdown('<h1 class="main-header">AI Assistant Pro</h1>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Chat", "Analytics", "Data Explorer", "Settings", "About"]
)

with tab1:
    st.header("Interactive Chat")

    # Chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display chat history
        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                # If it's the first message, use st.markdown directly (not inside HTML)
                if message["role"] == "user":
                    st.markdown(
                        """
                        <div class="user-message" style="
                            background-color: #f0f4ff;
                            color: #222;
                            border-left: 4px solid #2196f3;
                            border-radius: 10px;
                            padding: 10px;
                            margin: 5px 0;
                        ">
                            <strong style="color:#1976d2;">You:</strong><br>{}</div>
                        """.format(
                            message["content"]
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="bot-message" style="
                            background-color: #f7f3fa;
                            color: #222;
                            border-left: 4px solid #9c27b0;
                            border-radius: 10px;
                            padding: 10px;
                            margin: 5px 0;
                        ">
                            <strong style="color:#7c3aed;">AI:</strong><br>{}</div>
                        """.format(
                            message["content"]
                        ),
                        unsafe_allow_html=True,
                    )

        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])

            with col_input:
                user_input = st.text_input(
                    "Message",
                    placeholder="Ask me anything...",
                    label_visibility="collapsed",
                )

            with col_send:
                send_button = st.form_submit_button("Send", type="primary")

        # Additional input options (outside the form)
        col_voice, col_file, col_image = st.columns(3)

        with col_voice:
            voice_input = st.button("Voice Input")
            if voice_input:
                st.info("Voice input would be activated here")

        with col_file:
            uploaded_file = st.file_uploader(
                "Upload File",
                type=["txt", "pdf", "docx", "csv"],
                label_visibility="collapsed",
            )

        with col_image:
            uploaded_image = st.file_uploader(
                "Upload Image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )

        # Process user input
        if send_button and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Simulate AI response with spinner
            with st.spinner("AI is thinking..."):
                # Create an instance of MasterAgent
                agent = MasterAgent()
                ai_raw = agent.process_query(user_input)

            # Extract only the 'response' section and format as markdown
            if isinstance(ai_raw, dict) and "response" in ai_raw:
                ai_response = "\n\n" + ai_raw["response"]
            else:
                ai_response = str(ai_raw)

            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )
            st.rerun()

    with col2:
        st.subheader("Quick Actions")

        # Quick action buttons
        if st.button("Summarize Conversation"):
            if st.session_state.messages:
                st.info("Conversation summary would appear here")
            else:
                st.warning("No conversation to summarize")

        if st.button("Export Chat"):
            if st.session_state.messages:
                chat_text = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in st.session_state.messages
                    ]
                )
                st.download_button(
                    "Download Chat",
                    chat_text,
                    file_name="chat_history.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No chat to export")

        st.subheader("Suggested Topics")
        suggestions = [
            "Explain quantum computing",
            "Write a Python script",
            "Plan a vacation",
            "Healthy recipes",
            "Career advice",
        ]

        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion}"):
                st.session_state.messages.append(
                    {"role": "user", "content": suggestion}
                )
                st.rerun()

with tab2:
    st.header("Chat Analytics")

    # Generate sample data for analytics
    if st.session_state.messages:
        message_count = len(st.session_state.messages)
        user_messages = len(
            [m for m in st.session_state.messages if m["role"] == "user"]
        )
        ai_messages = len(
            [m for m in st.session_state.messages if m["role"] == "assistant"]
        )

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", message_count)

        with col2:
            st.metric("Your Messages", user_messages)

        with col3:
            st.metric("AI Responses", ai_messages)

        with col4:
            avg_length = np.mean([len(m["content"]) for m in st.session_state.messages])
            st.metric("Avg Message Length", f"{avg_length:.0f}")

        # Chat activity over time (simulated)
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7), end=datetime.now(), freq="D"
        )
        activity_data = pd.DataFrame(
            {"Date": dates, "Messages": np.random.poisson(5, len(dates))}
        )

        st.subheader("Chat Activity Over Time")
        st.line_chart(activity_data.set_index("Date")["Messages"])

        # Message length distribution
        message_lengths = [len(m["content"]) for m in st.session_state.messages]

        fig = px.histogram(
            x=message_lengths, nbins=10, title="Message Length Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Start chatting to see analytics!")

        # Show sample analytics anyway
        st.subheader("Sample Analytics Dashboard")

        # Sample metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Users", "1,234", "12%")
        with col2:
            st.metric("Total Conversations", "5,678", "8%")
        with col3:
            st.metric("Avg Response Time", "0.8s", "-0.2s")
        with col4:
            st.metric("Satisfaction Score", "4.8/5", "0.1")

        # Sample charts
        sample_data = pd.DataFrame(
            {
                "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                "Usage": [120, 150, 180, 200, 170, 90, 80],
            }
        )

        st.bar_chart(sample_data.set_index("Day"))

with tab3:
    st.header("Data Explorer")

    # Sample dataset
    @st.cache_data
    def load_sample_data():
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=100),
                "Users": np.random.randint(50, 500, 100),
                "Messages": np.random.randint(100, 1000, 100),
                "Response_Time": np.random.normal(0.8, 0.3, 100),
                "Satisfaction": np.random.uniform(3.5, 5.0, 100),
                "Category": np.random.choice(
                    ["Tech", "Business", "Personal", "Education"], 100
                ),
            }
        )
        return data

    data = load_sample_data()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Dataset Overview")
        st.dataframe(data, use_container_width=True)

        # Interactive filtering
        st.subheader("Filter Data")
        category_filter = st.multiselect(
            "Select Categories",
            options=data["Category"].unique(),
            default=data["Category"].unique(),
        )

        date_range = st.date_input(
            "Date Range",
            value=(data["Date"].min(), data["Date"].max()),
            min_value=data["Date"].min(),
            max_value=data["Date"].max(),
        )

        # Apply filters
        filtered_data = data[
            (data["Category"].isin(category_filter))
            & (data["Date"] >= pd.to_datetime(date_range[0]))
            & (data["Date"] <= pd.to_datetime(date_range[1]))
        ]

        # Visualization options
        st.subheader("Visualizations")
        viz_type = st.selectbox(
            "Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart"]
        )

        x_axis = st.selectbox("X-axis", data.columns)
        y_axis = st.selectbox("Y-axis", data.select_dtypes(include=[np.number]).columns)

        if viz_type == "Line Chart":
            fig = px.line(filtered_data, x=x_axis, y=y_axis, color="Category")
        elif viz_type == "Bar Chart":
            fig = px.bar(
                filtered_data.groupby("Category")[y_axis].mean().reset_index(),
                x="Category",
                y=y_axis,
            )
        elif viz_type == "Scatter Plot":
            fig = px.scatter(
                filtered_data, x=x_axis, y=y_axis, color="Category", size="Users"
            )
        else:  # Area Chart
            fig = px.area(filtered_data, x=x_axis, y=y_axis)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Data Summary")
        st.write("**Dataset Info:**")
        st.write(f"- Rows: {len(filtered_data)}")
        st.write(f"- Columns: {len(filtered_data.columns)}")
        st.write(
            f"- Date Range: {filtered_data['Date'].min().date()} to {filtered_data['Date'].max().date()}"
        )

        st.subheader("Statistics")
        st.write(filtered_data.describe())

        # Download options
        st.subheader("Export Options")

        # CSV download
        csv = filtered_data.to_csv(index=False)
        st.download_button("Download CSV", csv, "data.csv", "text/csv")

        # JSON download
        json_data = filtered_data.to_json(orient="records")
        st.download_button("Download JSON", json_data, "data.json", "application/json")

with tab4:
    st.header("Advanced Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Configuration")

        api_key = st.text_input("API Key", type="password")
        api_endpoint = st.text_input("API Endpoint", value="https://api.openai.com/v1")

        st.subheader("Performance Settings")

        max_concurrent = st.slider("Max Concurrent Requests", 1, 10, 3)
        request_timeout = st.slider("Request Timeout (seconds)", 5, 60, 30)
        retry_attempts = st.slider("Retry Attempts", 1, 5, 3)

        st.subheader("Privacy Settings")

        store_conversations = st.checkbox("Store Conversations", value=True)
        anonymize_data = st.checkbox("Anonymize User Data", value=False)
        data_retention = st.selectbox(
            "Data Retention Period",
            ["1 month", "3 months", "6 months", "1 year", "Forever"],
        )

    with col2:
        st.subheader("Notification Settings")

        email_notifications = st.checkbox("Email Notifications", value=True)
        push_notifications = st.checkbox("Push Notifications", value=False)

        notification_types = st.multiselect(
            "Notification Types",
            ["New Messages", "System Updates", "Maintenance", "Security Alerts"],
            default=["New Messages", "Security Alerts"],
        )

        st.subheader("Backup Settings")

        auto_backup = st.checkbox("Auto Backup", value=True)
        backup_frequency = st.selectbox(
            "Backup Frequency", ["Daily", "Weekly", "Monthly"]
        )

        if st.button("Create Backup Now"):
            with st.spinner("Creating backup..."):
                time.sleep(2)
                st.success("Backup created successfully!")

        st.subheader("Reset Options")

        col_reset1, col_reset2 = st.columns(2)

        with col_reset1:
            if st.button("Reset Settings", type="secondary"):
                st.warning("This will reset all settings to default values.")

        with col_reset2:
            if st.button("Factory Reset", type="secondary"):
                st.error("This will delete all data and settings.")

with tab5:
    st.header("About AI Assistant Pro")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ### Welcome to AI Assistant Pro
        
        This is a comprehensive demonstration of Streamlit's capabilities through a modern AI chatbot interface. 
        This application showcases:
        
        **Core Features:**
        - Interactive chat interface with message history
        - Real-time analytics and data visualization
        - Advanced settings and configuration options
        - File upload and processing capabilities
        - Multi-tab navigation and responsive design
        
        **Streamlit Components Demonstrated:**
        - Text inputs, buttons, and form controls
        - Data display with charts and metrics
        - File uploaders and download buttons
        - Tabs, columns, and layout components
        - Session state management
        - Custom CSS styling
        - Progress indicators and status messages
        
        **Technology Stack:**
        - **Frontend:** Streamlit
        - **Visualization:** Plotly, built-in Streamlit charts
        - **Data Processing:** Pandas, NumPy
        - **Styling:** Custom CSS
        """
        )

        st.subheader("Recent Updates")

        updates = [
            {
                "version": "v2.1.0",
                "date": "2024-08-30",
                "changes": "Added voice input support",
            },
            {"version": "v2.0.0", "date": "2024-08-15", "changes": "Major UI redesign"},
            {
                "version": "v1.9.5",
                "date": "2024-08-01",
                "changes": "Performance improvements",
            },
            {
                "version": "v1.9.0",
                "date": "2024-07-15",
                "changes": "New analytics dashboard",
            },
        ]

        for update in updates:
            with st.expander(f"Version {update['version']} - {update['date']}"):
                st.write(update["changes"])

    with col2:
        st.subheader("Quick Stats")

        st.metric("Version", "2.1.0")
        st.metric("Total Features", "25+")
        st.metric("Components Used", "15+")
        st.metric("Lines of Code", "400+")

        st.subheader("Resources")

        st.markdown(
            """
        **Useful Links:**
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [Plotly Documentation](https://plotly.com/python/)
        - [Pandas Documentation](https://pandas.pydata.org/docs/)
        """
        )

        st.subheader("Contact")

        st.text_input("Your Email")
        st.text_area("Feedback", placeholder="Tell us what you think...")

        if st.button("Send Feedback", type="primary"):
            st.success("Thank you for your feedback!")

        st.subheader("System Status")

        status_items = [
            ("API Status", "Operational", "success"),
            ("Database", "Operational", "success"),
            ("File Storage", "Operational", "success"),
            ("Analytics", "Maintenance", "warning"),
        ]

        for item, status, status_type in status_items:
            col_item, col_status = st.columns([2, 1])
            with col_item:
                st.write(item)
            with col_status:
                if status_type == "success":
                    st.success(status)
                elif status_type == "warning":
                    st.warning(status)
                else:
                    st.error(status)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>AI Assistant Pro - Powered by Streamlit | Built with LZX team for demonstration purposes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Progress bar for loading (demonstration)
if st.sidebar.button("Simulate Loading"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Loading... {i+1}%")
        time.sleep(0.01)

    status_text.text("Loading complete!")
    st.balloons()  # Celebration animation
