import os
import json
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

try:
    from onboarding_component import (
        OnboardingWorkflow,
        OnboardingConfig,
        create_sample_onboarding_docs,
    )

    ONBOARDING_AVAILABLE = True
except ImportError:
    print("Onboarding component not available")
    ONBOARDING_AVAILABLE = False

try:
    from re_upskilling_component import (
        LeadershipDevelopmentWorkflow,
        LeadershipConfig,
        create_sample_leadership_docs,
        run_sample_assessment,
    )

    RE_UPSKILLING_AVAILABLE = True
except ImportError:
    print("Onboarding component not available")
    RE_UPSKILLING_AVAILABLE = False

# ========================================================================================
# CONFIGURATION AND STATE MANAGEMENT
# ========================================================================================


@dataclass
class SystemConfig:
    """System configuration for the multi-agent router system."""

    OpenAIAPIKey: str
    OpenAIModel: str = "gpt-4o-mini"
    EmbeddingModel: str = "text-embedding-ada-002"
    ModelTemperature: float = 0.7
    MaxTokens: int = 4096
    EnableEvaluation: bool = True
    DebugModeOn: bool = True


class QueryType(Enum):
    """Query routing categories."""

    ONBOARDING = "ONBOARDING"
    RE_UP_SKILLING = "RE_UP_SKILLING"
    OTHER = "OTHER"


class MasterAgentState(TypedDict):
    """LangGraph-compatible state management for the multi-agent routing system."""

    # Core state
    user_query: str
    query_type: Optional[str]
    llm: Optional[Any]
    deepeval_model: Optional[Any]

    # Workflow response
    workflow_response: Optional[Dict[str, Any]]
    final_response: str

    # Evaluation results
    evaluation_results: Optional[Dict[str, Any]]
    faithfulness_score: Optional[float]
    relevancy_score: Optional[float]
    hallucination_score: Optional[float]
    overall_quality_score: Optional[float]

    # Metadata
    active_agents: List[str]
    processing_time: Optional[float]
    timestamp: Optional[str]
    error_info: Optional[Dict[str, Any]]


# ========================================================================================
# CONFIGURATION MANAGEMENT
# ========================================================================================


def create_sample_config():
    """Create a sample configuration file for the system."""
    config_data = {
        "OpenAIAPIKey": "",
        "OpenAIModel": "gpt-4o-mini",
        "EmbeddingModel": "text-embedding-ada-002",
        "ModelTemperature": 0.7,
        "MaxTokens": 4096,
        "EnableEvaluation": True,
        "DebugModeOn": True,
    }

    config_path = "config.json"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"Created sample config file: {config_path}")
        print("Please add your OpenAI API key to the config file.")

    return config_path


def load_config(config_path: str = "config.json") -> SystemConfig:
    """Load system configuration from JSON file with API key from Streamlit secrets."""
    try:
        # Try to load existing config file for other parameters
        with open(config_path, "r") as file:
            config = json.load(file)

        # Override API key with Streamlit secrets
        try:
            api_key = st.secrets["OpenAIAPIKey"]
            config["OpenAIAPIKey"] = api_key
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                print("Warning: Empty OpenAI API key found in Streamlit secrets.")
        except KeyError:
            print("Warning: OpenAI API key not found in Streamlit secrets.")
            # Keep the API key from config file if it exists, otherwise empty string
            api_key = config.get("OpenAIAPIKey", "")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                print("Warning: No OpenAI API key found in config file either.")

        return SystemConfig(**config)

    except FileNotFoundError:
        print(f"Config file {config_path} not found. Creating default config.")

        # Create default config
        default_config = {
            "OpenAIAPIKey": "",  # Will be overridden by Streamlit secrets
            "OpenAIModel": "gpt-4o-mini",
            "EmbeddingModel": "text-embedding-ada-002",
            "ModelTemperature": 0.7,
            "MaxTokens": 4096,
            "EnableEvaluation": True,
            "DebugModeOn": False,
        }

        # Try to get API key from Streamlit secrets
        try:
            api_key = st.secrets["OpenAIAPIKey"]
            default_config["OpenAIAPIKey"] = api_key
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("Successfully loaded OpenAI API key from Streamlit secrets.")
            else:
                print("Warning: Empty OpenAI API key found in Streamlit secrets.")
        except KeyError:
            print("Warning: OpenAI API key not found in Streamlit secrets.")

        # Create the config file for future reference
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config file: {config_path}")

        return SystemConfig(**default_config)

    except json.JSONDecodeError as e:
        print(f"Error parsing config file {config_path}: {e}")
        raise
    except KeyError as e:
        print(f"Missing required config key: {e}. Check your config file.")
        raise


# ========================================================================================
# MOCK IMPLEMENTATIONS
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
    """Initialize Language Learning Model with configuration parameters."""
    if LANGCHAIN_AVAILABLE and config.OpenAIAPIKey:
        return ChatOpenAI(
            model=config.OpenAIModel,
            temperature=config.ModelTemperature,
            max_tokens=config.MaxTokens,
            api_key=config.OpenAIAPIKey,
        )
    else:
        return MockLLM()


# ========================================================================================
# ROUTER AGENT
# ========================================================================================


class RouterAgent:
    """Enhanced routing agent for LangGraph workflows."""

    def __init__(self, llm, config: SystemConfig):
        self.llm = llm
        self.config = config

        self.classification_prompt = """
        You are an advanced query classification agent for a multi-workflow system.
        Your role is to accurately categorize user requests for optimal routing.
        Classify user queries into two categories:
        
        ONBOARDING - Company/HR related queries:
        - Company policies, values, culture
        - Benefits, PTO, compensation
        - Security policies and procedures
        - Remote work guidelines
        - Employee handbook content

        RE/UPSKILLING - Softskill or Hardskill related queries:
        - Develop leadership ability
        - Help user improve hardskill/softskill
        
        OTHER - All other queries:
        - Technical/programming questions
        - General knowledge
        - Content creation
        - Non-company specific help

        EXAMPLES:
            "Create onboarding materials for new software engineers" → ONBOARDING
            "What is our company vacation policy?" → ONBOARDING
            "How do I reset my company password?" → ONBOARDING
            "What are the core values of our organization?" → ONBOARDING
            "Explain machine learning algorithms" → OTHER
            "Write a Python script for data analysis" → OTHER
            "Help me write a blog post about AI trends" → OTHER
            "What is the capital of France?" → OTHER
        
        Respond with exactly 'ONBOARDING' or 'RE/UPSKILLING' or 'OTHER'.
        
        User Query: {query}
        
        Classification:"""

    def classify_query(self, state: MasterAgentState) -> MasterAgentState:
        """Classify query and update state."""
        try:
            prompt = self.classification_prompt.format(query=state["user_query"])
            response = self.llm.invoke(prompt)

            # Extract classification
            if hasattr(response, "content"):
                classification = response.content.strip().upper()
            else:
                classification = str(response).strip().upper()

            # Validate classification
            if classification in ("ONBOARDING", "RE/UPSKILLING", "OTHER"):
                state["query_type"] = classification
            else:
                state["query_type"] = "OTHER"

            state["active_agents"] = state.get("active_agents", []) + ["RouterAgent"]

            if self.config.DebugModeOn:
                print(f"Query classified as: {state['query_type']}")

        except Exception as e:
            print(f"Router error: {e}")
            state["query_type"] = "OTHER"  # Safe fallback
            if state.get("error_info") is None:
                state["error_info"] = {}
            state["error_info"] = state.get("error_info", {})
            state["error_info"]["routing_error"] = str(e)

        return state


# ========================================================================================
# DEEPEVAL INTEGRATION
# ========================================================================================


from typing import List, Dict, Any


class DeepEvalManager:
    """Manages DeepEval integration for response evaluation."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.evaluator_model = None

        if DEEPEVAL_AVAILABLE and config.OpenAIAPIKey:
            try:
                from deepeval.models import GPTModel

                self.evaluator_model = GPTModel(model=config.OpenAIModel)
                if self.config.DebugModeOn:
                    print("DeepEval evaluator initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize DeepEval: {e}")

    def evaluate_response(self, state: MasterAgentState) -> MasterAgentState:
        """Evaluate response quality using DeepEval metrics (faithfulness, relevancy, hallucination)."""
        if not self.evaluator_model or not state.get("workflow_response"):
            return self._add_mock_evaluation(state)

        try:
            workflow_response = state["workflow_response"]
            query = state["user_query"]
            answer = workflow_response.get("answer", "")

            # Extract context from retrieved documents
            context = []
            if (
                "retrieved_docs" in workflow_response
                and workflow_response["retrieved_docs"]
            ):
                context = [
                    doc.page_content for doc in workflow_response["retrieved_docs"]
                ]
            elif "sources" in workflow_response and workflow_response["sources"]:
                context = workflow_response["sources"]

            # Create test case for evaluation
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context if context else None,
                context=context if context else None,
            )

            # Initialize metrics list
            metrics = []

            # Add Faithfulness metric if context available
            if context:
                faithfulness_metric = FaithfulnessMetric(
                    threshold=0.7,
                    model=self.evaluator_model,
                    include_reason=False,
                    verbose_mode=False,
                )
                metrics.append(faithfulness_metric)

            # Add Answer Relevancy metric (always available)
            relevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=self.evaluator_model,
                include_reason=False,
                verbose_mode=False,
            )
            metrics.append(relevancy_metric)

            # Add Hallucination metric if context available
            if context:
                hallucination_metric = HallucinationMetric(
                    threshold=0.6,
                    model=self.evaluator_model,
                    include_reason=False,
                    verbose_mode=False,
                )
                metrics.append(hallucination_metric)

            # Run evaluation with all metrics together
            evaluation_results = evaluate(test_cases=[test_case], metrics=metrics)

            # Extract scores from results
            evaluation_scores = {
                "faithfulness": None,
                "relevancy": None,
                "hallucination": None,
            }

            # Parse results from the single evaluation run
            test_result = evaluation_results.test_results[0]
            for metric_data in test_result.metrics_data:
                metric_name = metric_data.name.lower()
                score = float(metric_data.score)

                if "faithfulness" in metric_name:
                    evaluation_scores["faithfulness"] = score
                elif "answer relevancy" in metric_name or "relevancy" in metric_name:
                    evaluation_scores["relevancy"] = score
                elif "hallucination" in metric_name:
                    evaluation_scores["hallucination"] = score

            # Calculate overall quality score
            # Note: For hallucination, higher scores mean more hallucination (bad)
            # So we invert it for the overall quality calculation
            scores_for_average = []
            if evaluation_scores["faithfulness"] is not None:
                scores_for_average.append(evaluation_scores["faithfulness"])
            if evaluation_scores["relevancy"] is not None:
                scores_for_average.append(evaluation_scores["relevancy"])
            if evaluation_scores["hallucination"] is not None:
                # Invert hallucination score (1 - score) since lower hallucination is better
                scores_for_average.append(1.0 - evaluation_scores["hallucination"])

            # Calculate weighted average if we have scores
            if scores_for_average:
                if len(scores_for_average) == 3:
                    # All three metrics available
                    overall_quality = (
                        evaluation_scores["faithfulness"] * 0.4
                        + evaluation_scores["relevancy"] * 0.4
                        + (1.0 - evaluation_scores["hallucination"]) * 0.2
                    )
                else:
                    # Only some metrics available, use simple average
                    overall_quality = sum(scores_for_average) / len(scores_for_average)
            else:
                overall_quality = 0.0

            # Update state with comprehensive evaluation results
            state["evaluation_results"] = {
                "faithfulness": evaluation_scores["faithfulness"],
                "relevancy": evaluation_scores["relevancy"],
                "hallucination": evaluation_scores["hallucination"],
                "overall_quality": overall_quality,
            }

            # Update individual score fields for backward compatibility
            state["faithfulness_score"] = evaluation_scores["faithfulness"]
            state["relevancy_score"] = evaluation_scores["relevancy"]
            state["hallucination_score"] = evaluation_scores["hallucination"]
            state["overall_quality_score"] = overall_quality

            # Print scores in order if debug mode is on
            if self.config.DebugModeOn:
                self._print_evaluation_results(evaluation_scores, overall_quality)

        except Exception as e:
            print(f"DeepEval evaluation error: {e}")
            state = self._add_mock_evaluation(state)
            if state.get("error_info") is None:
                state["error_info"] = {}
            state["error_info"]["evaluation_error"] = str(e)

        return state

    def _print_evaluation_results(self, scores: dict, overall_quality: float):
        """Print evaluation results in order with clear formatting."""
        print("=" * 70)
        print("DEEPEVAL ASSESSMENT RESULTS")
        print("=" * 70)

        # Print scores in order: Faithfulness, Relevancy, Hallucination
        faithfulness = scores.get("faithfulness")
        relevancy = scores.get("relevancy")
        hallucination = scores.get("hallucination")

        # Format each score properly
        faithfulness_str = f"{faithfulness:.3f}" if faithfulness is not None else "N/A"
        relevancy_str = f"{relevancy:.3f}" if relevancy is not None else "N/A"
        hallucination_str = (
            f"{hallucination:.3f}" if hallucination is not None else "N/A"
        )

        print(
            f"1. Faithfulness:   {faithfulness_str:>8} (how grounded in retrieved context)"
        )
        print(f"2. Relevancy:      {relevancy_str:>8} (how relevant to user query)")
        print(
            f"3. Hallucination:  {hallucination_str:>8} (lower is better - fabricated info)"
        )
        print("-" * 70)
        print(f"Overall Quality:   {overall_quality:>8.3f}")
        print("=" * 70)

    def _add_mock_evaluation(self, state: MasterAgentState) -> MasterAgentState:
        """Add mock evaluation results when DeepEval unavailable or evaluation fails."""
        confidence = state.get("workflow_response", {}).get("confidence", 0.5)

        state["evaluation_results"] = {
            "faithfulness": confidence,
            "relevancy": confidence,
            "hallucination": 1.0
            - confidence,  # Invert for hallucination (lower is better)
            "overall_quality": confidence,
            "mock_evaluation": True,
        }

        state["faithfulness_score"] = confidence
        state["relevancy_score"] = confidence
        state["hallucination_score"] = 1.0 - confidence
        state["overall_quality_score"] = confidence

        if self.config.DebugModeOn:
            print("=" * 70)
            print("MOCK EVALUATION RESULTS (DeepEval unavailable)")
            print("=" * 70)

            # Format scores properly for mock evaluation
            faithfulness_str = f"{confidence:.3f}"
            relevancy_str = f"{confidence:.3f}"
            hallucination_str = f"{1.0 - confidence:.3f}"
            overall_str = f"{confidence:.3f}"

            print(f"1. Faithfulness:   {faithfulness_str:>8} (mock)")
            print(f"2. Relevancy:      {relevancy_str:>8} (mock)")
            print(f"3. Hallucination:  {hallucination_str:>8} (mock)")
            print("-" * 70)
            print(f"Overall Quality:   {overall_str:>8}")
            print("=" * 70)

        return state


# ========================================================================================
# LANGGRAPH WORKFLOW NODES
# ========================================================================================


def initialize_system_node(state: MasterAgentState) -> MasterAgentState:
    """Initialize system components and prepare for processing."""
    try:
        config = load_config()
        llm = initialize_llm(config)

        state["llm"] = llm
        state["active_agents"] = ["SystemInitializer"]
        state["timestamp"] = datetime.now().isoformat()

        if DEEPEVAL_AVAILABLE and config.OpenAIAPIKey:
            try:
                state["deepeval_model"] = GPTModel(model=config.OpenAIModel)
            except Exception as e:
                print(f"DeepEval init failed: {e}")

        print("System initialized successfully")

    except Exception as e:
        print(f"System initialization error: {e}")
        if state.get("error_info") is None:
            state["error_info"] = {}
        state["error_info"] = {"init_error": str(e)}

    return state


def router_node(state: MasterAgentState) -> MasterAgentState:
    """Route query to appropriate workflow based on classification."""
    try:
        config = load_config()
        llm = state.get("llm") or initialize_llm(config)
        router = RouterAgent(llm, config)
        state = router.classify_query(state)

    except Exception as e:
        print(f"Routing error: {e}")
        state["query_type"] = "OTHER"  # Safe fallback
        if state.get("error_info") is None:
            state["error_info"] = {}
        state["error_info"] = state.get("error_info", {})
        state["error_info"]["routing_error"] = str(e)

    return state


def onboarding_workflow_node(state: MasterAgentState) -> MasterAgentState:
    """Execute onboarding workflow for HR/company-related queries."""
    try:
        if ONBOARDING_AVAILABLE and LANGCHAIN_AVAILABLE:
            config = load_config()
            llm = state.get("llm") or initialize_llm(config)

            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(api_key=config.OpenAIAPIKey)
            onboarding_config = OnboardingConfig()

            workflow = OnboardingWorkflow(llm, embeddings, onboarding_config)
            response = workflow.handle_query(state["user_query"])

            state["workflow_response"] = response
            state["active_agents"].append("OnboardingWorkflow")

            print(
                f"Onboarding workflow completed with confidence: {response.get('confidence', 0.0):.2f}"
            )
        else:
            # Fallback response when onboarding unavailable
            state["workflow_response"] = {
                "answer": "Onboarding workflow is not available. Please contact HR directly for company policy and onboarding questions.",
                "confidence": 0.0,
                "sources": [],
                "workflow": "onboarding_unavailable",
            }

    except Exception as e:
        print(f"Onboarding workflow error: {e}")
        state["workflow_response"] = {
            "answer": "I apologize, but I'm experiencing technical difficulties with the onboarding system. Please contact HR directly for assistance.",
            "confidence": 0.0,
            "sources": [],
            "workflow": "onboarding_error",
            "error": str(e),
        }
        if state.get("error_info") is None:
            state["error_info"] = {}
        state["error_info"] = state.get("error_info", {})
        state["error_info"]["onboarding_error"] = str(e)

    return state


def re_upskilling_workflow_node(state: MasterAgentState) -> MasterAgentState:
    """Execute re_upskilling workflow for personal development."""
    try:
        if RE_UPSKILLING_AVAILABLE and LANGCHAIN_AVAILABLE:
            config = load_config()
            llm = state.get("llm") or initialize_llm(config)

            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(api_key=config.OpenAIAPIKey)
            leadership_config = LeadershipConfig()

            workflow = LeadershipDevelopmentWorkflow(llm, embeddings, leadership_config)
            response = workflow.handle_query(
                state["user_query"], user_id="test_user_001"
            )

            sample_responses = run_sample_assessment(llm)
            learning_path = workflow.conduct_assessment(
                "test_user_001", sample_responses
            )

            print(f"Overall Progress: {learning_path.overall_progress:.2%}")
            print(
                f"Focus Areas: {', '.join([c.value for c in learning_path.current_focus_areas])}"
            )
            print(
                f"Recommended Modules: {', '.join(learning_path.recommended_modules[:3])}"
            )
            print(f"Next Milestone: {learning_path.next_milestone}")

            state["workflow_response"] = response
            state["active_agents"].append("Re/UpskillingWorkflow")

            print(
                f"Re/Upskilling workflow completed with confidence: {response.get('confidence', 0.0):.2f}"
            )
        else:
            # Fallback response when onboarding unavailable
            state["workflow_response"] = {
                "answer": "Re/Upskilling workflow is not available.",
                "confidence": 0.0,
                "sources": [],
                "workflow": "re/upskilling_unavailable",
            }

    except Exception as e:
        print(f"Re/Upskilling workflow error: {e}")
        state["workflow_response"] = {
            "answer": "I apologize, but I'm experiencing technical difficulties with the re/upskilling system.",
            "confidence": 0.0,
            "sources": [],
            "workflow": "re/upskilling_error",
            "error": str(e),
        }
        if state.get("error_info") is None:
            state["error_info"] = {}
        state["error_info"] = state.get("error_info", {})
        state["error_info"]["re/upskilling_error"] = str(e)

    return state


def general_workflow_node(state: MasterAgentState) -> MasterAgentState:
    """Execute general workflow for non-onboarding queries."""
    try:
        llm = state.get("llm")
        if not llm:
            config = load_config()
            llm = initialize_llm(config)

        general_prompt = f"""
        You are a helpful AI assistant. Please provide a comprehensive and helpful response to this query:
        
        Query: {state['user_query']}
        
        Provide a well-structured, informative response that addresses the user's question directly.
        """

        response = llm.invoke(general_prompt)

        # Extract response text
        if hasattr(response, "content"):
            answer = response.content.strip()
        else:
            answer = str(response).strip()

        state["workflow_response"] = {
            "answer": answer,
            "confidence": 0.8,
            "sources": ["AI Assistant"],
            "workflow": "general",
        }
        state["active_agents"].append("GeneralWorkflow")

        print("General response generated")

    except Exception as e:
        print(f"General workflow error: {e}")
        state["workflow_response"] = {
            "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing your question.",
            "confidence": 0.0,
            "sources": [],
            "workflow": "general_error",
            "error": str(e),
        }
        if state.get("error_info") is None:
            state["error_info"] = {}
        state["error_info"] = state.get("error_info", {})
        state["error_info"]["general_error"] = str(e)

    return state


def evaluation_node(state: MasterAgentState) -> MasterAgentState:
    """Evaluate response quality using DeepEval metrics."""
    try:
        config = load_config()
        evaluator = DeepEvalManager(config)
        state = evaluator.evaluate_response(state)
        state["active_agents"].append("DeepEvalAssessment")

    except Exception as e:
        print(f"Evaluation error: {e}")
        workflow_response = state.get("workflow_response", {})
        confidence = workflow_response.get("confidence", 0.5)
        state["overall_quality_score"] = confidence
        if state.get("error_info") is None:
            state["error_info"] = {}
        state["error_info"] = state.get("error_info", {})
        state["error_info"]["evaluation_error"] = str(e)

    return state


def finalize_response_node(state: MasterAgentState) -> MasterAgentState:
    """Finalize the response and prepare output."""
    try:
        workflow_response = state.get("workflow_response", {})
        answer = workflow_response.get("answer", "No response generated.")

        final_response_data = {
            "answer": answer,
            "query_classification": state.get("query_type", "UNKNOWN"),
            "confidence": workflow_response.get("confidence", 0.0),
            "sources": workflow_response.get("sources", []),
            "workflow_type": workflow_response.get("workflow", "unknown"),
            "evaluation": {
                "overall_quality": state.get("overall_quality_score", 0.0),
                "faithfulness": state.get("faithfulness_score"),
                "relevancy": state.get("relevancy_score"),
                "precision": state.get("hallucination_score"),
            },
            "metadata": {
                "active_agents": state.get("active_agents", []),
                "processing_time": state.get("processing_time", 0.0),
                "timestamp": state.get("timestamp"),
            },
        }
        if state.get("error_info") is None:
            state["error_info"] = {}
        elif state.get("error_info"):
            final_response_data["errors"] = state["error_info"]

        state["final_response"] = json.dumps(final_response_data, indent=2)
        state["active_agents"].append("ResponseFinalizer")

    except Exception as e:
        print(f"Finalization error: {e}")
        state["final_response"] = json.dumps(
            {
                "answer": "System error occurred during response finalization.",
                "error": str(e),
            }
        )

    return state


# ========================================================================================
# LANGGRAPH WORKFLOW ORCHESTRATOR
# ========================================================================================


class LangGraphWorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for multi-agent routing system."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.workflow_app = None

        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
        else:
            print("LangGraph not available - workflow orchestrator disabled")

    def _build_workflow(self):
        """Build the LangGraph workflow with all nodes and edges."""
        try:
            workflow = StateGraph(MasterAgentState)

            # Add workflow nodes
            workflow.add_node("initialize_system", initialize_system_node)
            workflow.add_node("router", router_node)
            workflow.add_node("onboarding_workflow", onboarding_workflow_node)
            workflow.add_node("re/upskilling_workflow", re_upskilling_workflow_node)
            workflow.add_node("general_workflow", general_workflow_node)
            workflow.add_node("evaluation", evaluation_node)
            workflow.add_node("finalize_response", finalize_response_node)

            # Define workflow edges
            workflow.add_edge(START, "initialize_system")
            workflow.add_edge("initialize_system", "router")

            # Conditional routing
            workflow.add_conditional_edges(
                "router",
                self._route_to_workflow,
                {
                    "onboarding": "onboarding_workflow",
                    "re/upskilling": "re/upskilling_workflow",
                    "general": "general_workflow",
                },
            )

            workflow.add_edge("onboarding_workflow", "evaluation")
            workflow.add_edge("re/upskilling_workflow", "evaluation")
            workflow.add_edge("general_workflow", "evaluation")
            workflow.add_edge("evaluation", "finalize_response")
            workflow.add_edge("finalize_response", END)

            # Compile workflow
            self.workflow_app = workflow.compile()

            print("LangGraph workflow compiled successfully.")

        except Exception as e:
            print(f"Error building LangGraph workflow: {e}")
            self.workflow_app = None

    def _route_to_workflow(self, state: MasterAgentState) -> str:
        """Determine which workflow to execute based on query classification."""
        query_type = state.get("query_type", "OTHER")
        if query_type == "ONBOARDING":
            return "onboarding"
        elif query_type == "RE/UPSKILLING":
            return "re/upskilling"
        else:
            return "general"

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through complete LangGraph workflow."""
        if not self.workflow_app:
            return self._fallback_processing(query)

        start_time = datetime.now()

        try:
            # Initialize state
            initial_state: MasterAgentState = {
                "user_query": query,
                "query_type": None,
                "llm": None,
                "deepeval_model": None,
                "workflow_response": None,
                "final_response": "",
                "evaluation_results": None,
                "faithfulness_score": None,
                "relevancy_score": None,
                "hallucination_score": None,
                "overall_quality_score": None,
                "active_agents": [],
                "processing_time": None,
                "timestamp": None,
                "error_info": None,
            }

            # Execute workflow
            final_state = self.workflow_app.invoke(initial_state)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            final_state["processing_time"] = processing_time

            # Parse final response
            try:
                final_response_data = json.loads(final_state["final_response"])
                final_response_data["metadata"]["processing_time"] = processing_time
                return final_response_data
            except json.JSONDecodeError:
                return {
                    "answer": final_state["final_response"],
                    "processing_time": processing_time,
                    "workflow_app": "langgraph",
                }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"Workflow execution error: {e}")

            return {
                "answer": f"I encountered an error processing your request: {str(e)}",
                "error": str(e),
                "processing_time": processing_time,
                "workflow_app": "langgraph_error",
            }

    def _fallback_processing(self, query: str) -> Dict[str, Any]:
        """Fallback processing when LangGraph unavailable."""
        return {
            "answer": "LangGraph workflow is not available. The system cannot process queries at this time.",
            "error": "LangGraph unavailable",
            "workflow_app": "fallback",
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and configuration."""
        return {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "workflow_compiled": self.workflow_app is not None,
            "deepeval_available": DEEPEVAL_AVAILABLE,
            "onboarding_available": ONBOARDING_AVAILABLE,
        }


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================


def main():
    """Main execution function demonstrating the complete LangGraph multi-agent workflow system."""
    print("=" * 80)
    print("MULTI-AGENT HCM SYSTEM")
    print("=" * 80)

    # Load configuration
    try:
        create_sample_config()
        config = load_config()
        print(
            f"Configuration loaded - Model: {config.OpenAIModel}, Debug: {config.DebugModeOn}"
        )
    except Exception as e:
        print(f"Configuration error: {e}")
        return

    # Display system capabilities
    print(f"\nSystem Dependencies:")
    print(f"  LangChain Available: {LANGCHAIN_AVAILABLE}")
    print(f"  LangGraph Available: {LANGGRAPH_AVAILABLE}")
    print(f"  DeepEval Available: {DEEPEVAL_AVAILABLE}")
    print(f"  Onboarding Component: {ONBOARDING_AVAILABLE}")

    if not LANGGRAPH_AVAILABLE:
        print(
            f"\nERROR: LangGraph is required. Please install with: pip install langgraph"
        )
        return

    # Create sample onboarding documents
    if ONBOARDING_AVAILABLE:
        try:
            create_sample_onboarding_docs()
            print(f"\nSample onboarding documents created/verified")
        except Exception as e:
            print(f"Warning: Could not create sample docs: {e}")

    # Initialize workflow orchestrator
    print(f"\nInitializing LangGraph workflow orchestrator...")
    try:
        orchestrator = LangGraphWorkflowOrchestrator(config)
        status = orchestrator.get_workflow_status()

        print(f"\nWorkflow Status:")
        for key, value in status.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        if not status["workflow_compiled"]:
            print(f"ERROR: LangGraph workflow could not be compiled.")
            return

        print(f"LangGraph workflow orchestrator initialized successfully!")

    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        return

    # Test queries
    # test_queries = [
    #     {
    #         "query": "What are our company's core values and how do they guide daily work culture?",
    #         "expected_type": "ONBOARDING",
    #         "description": "Company culture inquiry",
    #     },
    #     {
    #         "query": "Explain our remote work policy including IT security requirements",
    #         "expected_type": "ONBOARDING",
    #         "description": "Remote work policy",
    #     },
    #     {
    #         "query": "Write a Python function to implement binary search",
    #         "expected_type": "OTHER",
    #         "description": "Programming assistance",
    #     },
    #     {
    #         "query": "What is the capital of France?",
    #         "expected_type": "OTHER",
    #         "description": "General knowledge",
    #     },
    # ]

    # print(f"\n{'='*80}")
    # print("PROCESSING TEST QUERIES THROUGH LANGGRAPH WORKFLOW")
    # print(f"{'='*80}")

    # # Process test queries
    # successful_queries = 0
    # classification_accuracy = 0

    # for i, test_case in enumerate(test_queries, 1):
    #     query = test_case["query"]
    #     expected = test_case["expected_type"]
    #     description = test_case["description"]

    #     print(f"\nQuery {i}: {description}")
    #     print(f"Input: {query}")
    #     print(f"Expected: {expected}")
    #     print("-" * 70)

    #     try:
    #         result = orchestrator.process_query(query)

    #         classification = result.get("query_classification", "UNKNOWN")
    #         answer = result.get("answer", "No answer provided")
    #         confidence = result.get("confidence", 0.0)
    #         processing_time = result.get("processing_time", 0.0)

    #         classification_correct = classification == expected
    #         if classification_correct:
    #             classification_accuracy += 1
    #             status = "CORRECT"
    #         else:
    #             status = "INCORRECT"

    #         print(f"Classification: {classification} ({status})")
    #         print(f"Processing Time: {processing_time:.3f}s")
    #         print(f"Confidence: {confidence:.2f}")

    #         evaluation = result.get("evaluation", {})
    #         if evaluation.get("overall_quality"):
    #             print(f"Quality Score: {evaluation['overall_quality']:.2f}")

    #         print(f"Final Answer:\n{answer}")

    #         successful_queries += 1

    #     except Exception as e:
    #         print(f"Error processing query: {e}")

    #     print("=" * 80)

    # # Display summary
    # print(f"\nSYSTEM PERFORMANCE SUMMARY")
    # print(f"{'='*80}")
    # print(f"Total Queries: {len(test_queries)}")
    # print(
    #     f"Successful: {successful_queries}/{len(test_queries)} ({successful_queries/len(test_queries)*100:.1f}%)"
    # )
    # print(
    #     f"Classification Accuracy: {classification_accuracy}/{len(test_queries)} ({classification_accuracy/len(test_queries)*100:.1f}%)"
    # )

    # print(f"\nCore Components:")
    # print(f"  Router Agent: Operational")
    # print(f"  General Workflow: Operational")
    # workflow_status = orchestrator.get_workflow_status()
    # if workflow_status["onboarding_available"]:
    #     print(f"  Onboarding Workflow: Operational")
    # if workflow_status["deepeval_available"]:
    #     print(f"  DeepEval Integration: Operational")

    # print(f"\nSYSTEM READY FOR PRODUCTION!")

    # Interactive mode
    workflow_status = orchestrator.get_workflow_status()
    if config.DebugModeOn and workflow_status["workflow_compiled"]:
        print(f"\nINTERACTIVE MODE AVAILABLE")
        print(f"Test additional queries (enter 'quit' to exit):")

        while True:
            try:
                user_query = input(f"\nEnter query: ").strip()

                if user_query.lower() in ["quit", "exit", "q"]:
                    break
                elif user_query:
                    result = orchestrator.process_query(user_query)
                    print(f"\nResults:")
                    print(
                        f"  Classification: {result.get('query_classification', 'Unknown')}"
                    )
                    print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
                    answer = result.get("answer", "No answer")
                    print(f"  Answer: {answer}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print(f"Interactive session ended.")


if __name__ == "__main__":
    main()
