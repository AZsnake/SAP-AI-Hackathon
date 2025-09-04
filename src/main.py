import os
import json
from typing import Dict, List, Optional, TypedDict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Optional imports with fallback handling
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    from crewai import Agent as CrewAIAgent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    print("CrewAI not available - using mock agents")
    CREWAI_AVAILABLE = False

try:
    from colorama import Fore, Style
except ImportError:
    # Fallback for colorama when not available
    class Fore:
        RED = GREEN = BLUE = ""
    class Style:
        RESET_ALL = ""


# ========================================================================================
# CONFIGURATION MANAGEMENT
# ========================================================================================

@dataclass
class SystemConfig:
    """
    System configuration for the multi-agent router system.
    Contains API keys, model settings, and operational parameters.
    """
    OpenAIAPIKey: str
    OpenAIModel: str
    EmbeddingModel: str
    ModelTemperature: float
    MaxTokens: int
    EnableEvaluation: bool
    DebugModeOn: bool


class QueryType(Enum):
    """
    Enumeration of supported query routing categories.
    Used by RouterAgent to classify incoming user requests.
    """
    ONBOARDING = "ONBOARDING"  # Company onboarding and HR-related queries
    OTHER = "OTHER"            # General queries not related to onboarding


def load_config(config_path: str = "config.json") -> SystemConfig:
    """
    Load system configuration from JSON file and set up environment.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        SystemConfig object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required config keys are missing
    """
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        
        # Extract configuration values with validation
        api_key = config["OpenAIAPIKey"]
        
        # Set environment variable for OpenAI API
        os.environ["OPENAI_API_KEY"] = api_key
        
        if api_key:
            print("Valid API key found, system ready.")
        else:
            print("Warning: No OpenAI API key found. Configure API key in config file.")
        
        return SystemConfig(
            OpenAIAPIKey=api_key,
            OpenAIModel=config["OpenAIModel"],
            EmbeddingModel=config["EmbeddingModel"],
            ModelTemperature=config["ModelTemperature"],
            MaxTokens=config["MaxTokens"],
            EnableEvaluation=config["EnableEvaluation"],
            DebugModeOn=config["DebugModeOn"]
        )
        
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return SystemConfig("", "gpt-4o-mini", "text-embedding-ada-002", 0.7, 4096, False, True)
    except KeyError as e:
        print(f"Missing config key: {e}. Check your config file.")
        raise


# ========================================================================================
# STATE MANAGEMENT
# ========================================================================================

class MasterAgentState(TypedDict):
    """
    Central state management for the multi-agent routing system.
    Tracks query processing through different workflow stages.
    """
    # Input and routing information
    user_query: str                                    # Original user input
    query_type: Optional[str]                         # Classified query category
    
    # Workflow-specific state (extensible for different agent types)
    topic: Optional[str]                              # Extracted topic for research workflows
    outline: Optional[Dict[str, Any]]                 # Content structure for generation tasks
    research_sections: Optional[List[Dict[str, Any]]] # Research workflow sections
    final_article: Optional[str]                      # Generated content output
    
    # Support workflow state
    support_category: Optional[str]                   # Support ticket classification
    solution_steps: Optional[List[str]]               # Troubleshooting steps
    escalation_needed: Optional[bool]                 # Whether human intervention needed
    
    # Shared processing state
    retrieved_documents: Optional[List]               # Retrieved context documents
    evaluation_results: Optional[Dict[str, Any]]      # Quality assessment results
    final_response: str                               # System's final response
    
    # System monitoring
    active_agents: List[str]                          # Currently active agent names
    processing_time: Optional[float]                  # Total processing duration
    confidence_score: Optional[float]                 # Response confidence rating


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

class MockLLM:
    """Mock LLM implementation for testing when LangChain unavailable."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def invoke(self, prompt: str) -> str:
        """Return mock response based on prompt content."""
        if "ONBOARDING" in prompt.upper():
            return "ONBOARDING"
        return "OTHER"


def initialize_llm(config: SystemConfig):
    """
    Initialize Language Learning Model with configuration parameters.
    
    Args:
        config: System configuration object
        
    Returns:
        Configured LLM instance (ChatOpenAI or MockLLM)
    """
    if LANGCHAIN_AVAILABLE and config.OpenAIAPIKey:
        return ChatOpenAI(
            model=config.OpenAIModel,
            temperature=config.ModelTemperature,
            max_tokens=config.MaxTokens,
            api_key=config.OpenAIAPIKey,
        )
    else:
        return MockLLM()


def log_activity(agent_name: str, action: str, details: Dict[str, Any] = None):
    """
    Log agent activities for debugging and system monitoring.
    
    Args:
        agent_name: Name of the agent performing the action
        action: Description of the action being performed
        details: Optional dictionary of additional details
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Fore.RED}[LOG]{Style.RESET_ALL} {Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {agent_name}: {action}")
    
    if details:
        print(f"Details: {json.dumps(details, indent=2, default=str)}")


# ========================================================================================
# ROUTER AGENT IMPLEMENTATION
# ========================================================================================

class Agent:
    """Simple agent class for instruction-based behavior."""
    def __init__(self, name: str, instruction: str):
        self.name = name
        self.instruction = instruction


class RouterAgent:
    """
    Primary routing agent that classifies user queries and determines workflow assignment.
    
    Uses instruction-based classification to route queries to appropriate specialized agents.
    Currently supports ONBOARDING (HR/company info) and OTHER (general queries) categories.
    """
    
    def __init__(self, llm, config: SystemConfig):
        """
        Initialize RouterAgent with LLM and system configuration.
        
        Args:
            llm: Language model for query classification
            config: System configuration object
        """
        self.llm = llm
        self.config = config
        self.agent = Agent(
            name="Router Agent",
            instruction="""
            You are a query classification agent. Categorize user requests into specific workflows.

            CLASSIFICATION RULES:
            - ONBOARDING: Queries about company onboarding, HR policies, employee handbook, 
              company rules, new hire information, organizational procedures
            - OTHER: All other queries including technical support, general questions, 
              content generation not related to onboarding

            OUTPUT: Respond with exactly 'ONBOARDING' or 'OTHER' (no additional text)

            EXAMPLES:
            "Create onboarding materials for new employees" -> ONBOARDING
            "What are our company vacation policies?" -> ONBOARDING  
            "Write an article about AI trends" -> OTHER
            "Help with password reset" -> OTHER
            """
        )

    def classify_query(self, state: MasterAgentState) -> MasterAgentState:
        """
        Classify user query and update system state with routing decision.
        
        Args:
            state: Current system state containing user query
            
        Returns:
            Updated state with query_type populated and agent tracking
        """
        if self.config.DebugModeOn:
            log_activity("RouterAgent", "Starting query classification", 
                        {"query": state["user_query"][:100]})  # Truncate for logging
        
        try:
            # Construct classification prompt
            prompt = f"{self.agent.instruction}\n\nQuery: {state['user_query']}"
            
            # Get LLM classification response
            response = self.llm.invoke(prompt)
            
            # Extract and normalize response text
            if hasattr(response, "content"):
                query_type = response.content.strip().upper()
            else:
                query_type = str(response).strip().upper()
            
            # Validate classification result
            if query_type not in ("ONBOARDING", "OTHER"):
                if self.config.DebugModeOn:
                    log_activity("RouterAgent", "Invalid classification, defaulting to OTHER", 
                               {"raw_response": query_type})
                query_type = "OTHER"
            
            # Update state with classification results
            state["query_type"] = query_type
            state["active_agents"] = ["RouterAgent"]
            
            if self.config.DebugModeOn:
                log_activity("RouterAgent", "Classification completed", 
                           {"query_type": query_type})
            
        except Exception as e:
            # Handle classification errors gracefully
            log_activity("RouterAgent", "Classification error occurred", 
                        {"error": str(e), "error_type": type(e).__name__})
            state["query_type"] = "OTHER"  # Safe fallback
            state["active_agents"] = ["RouterAgent"]
        
        return state

    def route_query(self, query: str) -> tuple[str, MasterAgentState]:
        """
        Complete query routing process from input to classification.
        
        Args:
            query: User input query string
            
        Returns:
            Tuple of (classified_type, final_state)
        """
        # Initialize state for processing
        initial_state: MasterAgentState = {
            "user_query": query,
            "query_type": None,
            "topic": None,
            "outline": None,
            "research_sections": None,
            "final_article": None,
            "support_category": None,
            "solution_steps": None,
            "escalation_needed": None,
            "retrieved_documents": None,
            "evaluation_results": None,
            "final_response": "",
            "active_agents": [],
            "processing_time": None,
            "confidence_score": None
        }
        
        # Process classification
        start_time = datetime.now()
        final_state = self.classify_query(initial_state)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update processing metrics
        final_state["processing_time"] = processing_time
        
        return final_state["query_type"], final_state


# Import onboarding workflow component
try:
    from onboarding_component import OnboardingWorkflow, OnboardingConfig, create_sample_onboarding_docs
    ONBOARDING_AVAILABLE = True
except ImportError:
    print("Onboarding component not available")
    ONBOARDING_AVAILABLE = False


# ========================================================================================
# WORKFLOW ORCHESTRATOR
# ========================================================================================

class WorkflowOrchestrator:
    """
    Orchestrates different workflows based on query classification.
    Routes queries to appropriate specialized agents and workflows.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize orchestrator with all available workflows.
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.llm = initialize_llm(config)
        
        # Initialize router agent
        self.router = RouterAgent(self.llm, config)
        
        # Initialize onboarding workflow if available
        self.onboarding_workflow = None
        if ONBOARDING_AVAILABLE and LANGCHAIN_AVAILABLE and config.OpenAIAPIKey:
            try:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(api_key=config.OpenAIAPIKey)
                onboarding_config = OnboardingConfig()
                self.onboarding_workflow = OnboardingWorkflow(self.llm, embeddings, onboarding_config)
                print("Onboarding workflow initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize onboarding workflow: {e}")
        else:
            print("Onboarding workflow unavailable (missing dependencies or API key)")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query through complete routing and workflow execution.
        
        Args:
            query: User input query
            
        Returns:
            Complete response with workflow results and metadata
        """
        start_time = datetime.now()
        
        if self.config.DebugModeOn:
            log_activity("Orchestrator", "Starting query processing", {"query": query[:100]})
        
        # Step 1: Route query to determine workflow
        query_type, routing_state = self.router.route_query(query)
        
        # Step 2: Execute appropriate workflow
        workflow_response = self._execute_workflow(query_type, query, routing_state)
        
        # Step 3: Compile final response
        total_time = (datetime.now() - start_time).total_seconds()
        
        final_response = {
            "query": query,
            "classification": query_type,
            "workflow_response": workflow_response,
            "routing_state": routing_state,
            "total_processing_time": total_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.config.DebugModeOn:
            log_activity("Orchestrator", "Query processing completed", 
                        {"total_time": total_time, "classification": query_type})
        
        return final_response
    
    def _execute_workflow(self, query_type: str, query: str, routing_state: MasterAgentState) -> Dict[str, Any]:
        """
        Execute the appropriate workflow based on query classification.
        
        Args:
            query_type: Classified query type (ONBOARDING or OTHER)
            query: Original user query
            routing_state: Current routing state
            
        Returns:
            Workflow execution results
        """
        if query_type == "ONBOARDING":
            return self._handle_onboarding_query(query, routing_state)
        else:
            return self._handle_general_query(query, routing_state)
    
    def _handle_onboarding_query(self, query: str, routing_state: MasterAgentState) -> Dict[str, Any]:
        """
        Handle onboarding-related queries using RAG workflow.
        
        Args:
            query: Onboarding-related query
            routing_state: Current routing state
            
        Returns:
            Onboarding workflow response
        """
        if self.onboarding_workflow:
            try:
                response = self.onboarding_workflow.handle_query(query)
                routing_state["active_agents"].append("OnboardingWorkflow")
                return response
            except Exception as e:
                log_activity("Orchestrator", "Onboarding workflow error", {"error": str(e)})
                return {
                    "answer": "I apologize, but I'm experiencing technical difficulties with the onboarding system. Please contact HR directly for assistance.",
                    "confidence": 0.0,
                    "sources": [],
                    "workflow": "onboarding_error",
                    "error": str(e)
                }
        else:
            return {
                "answer": "Onboarding workflow is not available. Please contact HR directly for company policy and onboarding questions.",
                "confidence": 0.0,
                "sources": [],
                "workflow": "onboarding_unavailable"
            }
    
    def _handle_general_query(self, query: str, routing_state: MasterAgentState) -> Dict[str, Any]:
        """
        Handle general queries using basic LLM response.
        
        Args:
            query: General query not related to onboarding
            routing_state: Current routing state
            
        Returns:
            General response
        """
        try:
            # For general queries, provide a helpful response
            general_prompt = f"""
            You are a helpful AI assistant. Please provide a comprehensive and helpful response to this query:
            
            Query: {query}
            
            Provide a well-structured, informative response that addresses the user's question directly.
            """
            
            response = self.llm.invoke(general_prompt)
            
            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
            
            routing_state["active_agents"].append("GeneralWorkflow")
            
            return {
                "answer": answer,
                "confidence": 0.8,
                "sources": ["AI Assistant"],
                "workflow": "general",
                "retrieved_documents_count": 0
            }
            
        except Exception as e:
            log_activity("Orchestrator", "General workflow error", {"error": str(e)})
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing your question.",
                "confidence": 0.0,
                "sources": [],
                "workflow": "general_error",
                "error": str(e)
            }


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    """
    Main execution function demonstrating complete multi-agent workflow system.
    
    Initializes all components, creates sample data, and processes test queries
    through the complete routing and workflow execution pipeline.
    """
    print("=" * 70)
    print("MULTI-AGENT ROUTER SYSTEM WITH ONBOARDING WORKFLOW")
    print("=" * 70)
    
    # Step 1: Load system configuration
    try:
        config = load_config()
        print(f"Configuration loaded: Model={config.OpenAIModel}, Debug={config.DebugModeOn}")
    except Exception as e:
        print(f"Configuration error: {e}")
        return
    
    # Step 2: Create sample onboarding documents if needed
    if ONBOARDING_AVAILABLE:
        try:
            create_sample_onboarding_docs()
            print("Sample onboarding documents created/verified")
        except Exception as e:
            print(f"Warning: Could not create sample docs: {e}")
    
    # Step 3: Initialize workflow orchestrator
    print("\nInitializing workflow orchestrator...")
    try:
        orchestrator = WorkflowOrchestrator(config)
        print("Workflow orchestrator initialized successfully")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        return
    
    print(f"\n{'='*70}")
    print("PROCESSING TEST QUERIES")
    print(f"{'='*70}")
    
    # Step 4: Test queries demonstrating both workflow types
    test_queries = [
        # Onboarding queries
        {
            "query": "What are the company's core values and mission?",
            "expected_type": "ONBOARDING",
            "description": "Company culture inquiry"
        },
        {
            "query": "How many PTO days do new employees get and what's the parental leave policy?",
            "expected_type": "ONBOARDING", 
            "description": "Benefits and time-off policy"
        },
        {
            "query": "What are the password requirements and remote work security policies?",
            "expected_type": "ONBOARDING",
            "description": "Security policy inquiry"
        },
        # General queries  
        {
            "query": "Write a Python function to calculate fibonacci numbers",
            "expected_type": "OTHER",
            "description": "Programming assistance"
        },
        {
            "query": "Explain the differences between supervised and unsupervised machine learning",
            "expected_type": "OTHER", 
            "description": "Technical explanation"
        }
    ]
    
    # Step 5: Process each test query
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected_type"]
        description = test_case["description"]
        
        print(f"\n{Fore.BLUE}Query {i}: {description}{Style.RESET_ALL}")
        print(f"Input: {query}")
        print(f"Expected Classification: {expected}")
        print("-" * 50)
        
        try:
            # Process through complete workflow
            result = orchestrator.process_query(query)
            
            # Display results
            classification = result["classification"]
            workflow_response = result["workflow_response"]
            processing_time = result["total_processing_time"]
            
            # Classification accuracy
            print(f"Actual Classification: {classification}")
            print(f"Processing Time: {processing_time:.3f}s")
            
            # Workflow response
            if "answer" in workflow_response:
                answer = workflow_response["answer"]
                confidence = workflow_response.get("confidence", 0.0)
                sources = workflow_response.get("sources", [])
                workflow_type = workflow_response.get("workflow", "unknown")
                
                print(f"Workflow: {workflow_type}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Sources: {', '.join(sources) if sources else 'None'}")
                print(f"Answer: {answer}")
                
                # Show retrieval info for onboarding queries
                if classification == "ONBOARDING" and "retrieved_documents_count" in workflow_response:
                    doc_count = workflow_response["retrieved_documents_count"]
                    print(f"Retrieved Documents: {doc_count}")
            else:
                print("No workflow response available")
                
        except Exception as e:
            print(f"✗ Error processing query: {e}")
        
        print("=" * 70)
    
    # Step 6: Display system summary
    print(f"\n{Fore.GREEN}SYSTEM SUMMARY{Style.RESET_ALL}")
    print(f"Router Agent: Operational")
    print(f"General Workflow: Operational")
    
    if orchestrator.onboarding_workflow:
        status = orchestrator.onboarding_workflow.get_system_status()
        print(f"Onboarding Workflow: Operational")
        print(f"  - Vector Store: {'Initialized' if status['vector_store_initialized'] else 'Not initialized'}")
        print(f"  - Self-Correction: {'Enabled' if status['config']['self_correction_enabled'] else 'Disabled'}")
        print(f"  - Top-K Retrieval: {status['config']['top_k_retrieval']}")
    else:
        print(f"Onboarding Workflow: Unavailable")
    
    print(f"\n{Fore.GREEN}System ready for production queries!{Style.RESET_ALL}")
    
    # Interactive mode option
    if config.DebugModeOn:
        print(f"\n{Fore.YELLOW}Interactive Mode Available{Style.RESET_ALL}")
        print("You can now test additional queries:")
        while True:
            try:
                user_query = input("\nEnter query (or 'quit' to exit): ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                if user_query:
                    result = orchestrator.process_query(user_query)
                    print(f"Classification: {result['classification']}")
                    print(f"Answer: {result['workflow_response']['answer']}")
            except KeyboardInterrupt:
                break
        
        print("Interactive session ended.")


if __name__ == "__main__":
    main()