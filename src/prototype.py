import os
import json
from typing import Dict, List, Optional, TypedDict, Literal, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# Core framework imports
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangGraph imports for workflow management
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# CrewAI imports for collaborative agents
from crewai import Agent as CrewAIAgent, Task, Crew, Process

# Misc import
from colorama import Fore, Style

# OpenAI Swarm for simple agent coordination (simulated)
class Agent:
    """Simplified OpenAI Swarm-style agent for routing"""
    def __init__(self, name: str, instructions: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.instructions = instructions
        self.model = model

# DeepEval imports for evaluation (simulated for this example)
class HallucinationMetric:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

class AnswerRelevancyMetric:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold


# ========================================================================================
# 1. CONFIGURATION AND SETUP
# ========================================================================================

@dataclass
class SystemConfig:
    """Configuration class for the multi-agent system"""
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-ada-002"
    temperature: float = 0.7
    max_tokens: int = 2000
    enable_evaluation: bool = True
    debug_mode: bool = False

class QueryType(Enum):
    """Enumeration of supported query types"""
    RESEARCH = "RESEARCH"
    CUSTOMER_SUPPORT = "CUSTOMER_SUPPORT"

# Initialize configuration
config = SystemConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY", "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"),
    debug_mode=True
)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = config.openai_api_key


# ========================================================================================
# 2. STATE MANAGEMENT (LangGraph)
# ========================================================================================

class MasterAgentState(TypedDict):
    """
    Central state management for the multi-agent system.
    This state is shared across all agents and tracks the entire conversation flow.
    """
    # Input and routing
    user_query: str
    query_type: Optional[str]
    
    # Research workflow state
    topic: Optional[str]
    outline: Optional[Dict[str, Any]]
    research_sections: Optional[List[Dict[str, Any]]]
    final_article: Optional[str]
    
    # Support workflow state
    support_category: Optional[str]
    solution_steps: Optional[List[str]]
    escalation_needed: Optional[bool]
    
    # Shared state
    retrieved_documents: Optional[List[Document]]
    evaluation_results: Optional[Dict[str, Any]]
    final_response: str
    
    # System state
    active_agents: List[str]
    processing_time: Optional[float]
    confidence_score: Optional[float]


# ========================================================================================
# 3. UTILITY FUNCTIONS AND LLM INITIALIZATION
# ========================================================================================

def initialize_llm(model: str = "gpt-4o-mini", temperature: float = 0.7) -> ChatOpenAI:
    """
    Initialize the Language Learning Model with specified parameters.
    
    Args:
        model: The OpenAI model to use
        temperature: Controls randomness in responses (0.0 = deterministic, 1.0 = random)
    
    Returns:
        Initialized ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=config.max_tokens
    )

def initialize_embeddings(model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    """Initialize embedding model for document similarity and retrieval"""
    return OpenAIEmbeddings(model=model)

def log_agent_activity(agent_name: str, action: str, details: Dict[str, Any] = None):
    """
    Log agent activities for debugging and monitoring.
    In production, this would integrate with proper logging systems.
    """
    if config.debug_mode:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.RED}[LOG]{Style.RESET_ALL} {Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {agent_name}: {action}")
        if details:
            print(f"Details in this action:\n{json.dumps(details, indent=2, default=str)}")


# ========================================================================================
# 4. ROUTING AGENT (OpenAI Swarm Style)
# ========================================================================================

class RouterAgent:
    """
    OpenAI Swarm-style routing agent that classifies incoming queries and determines
    which specialized workflow should handle the request.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.agent = Agent(
            name="Router Agent",
            instructions="""
            You are a classification agent that categorizes user queries.
            
            RULES:
            - If the query asks for research, analysis, trends, knowledge generation, 
              writing, or academic content, return 'RESEARCH'
            - If the query is about product issues, troubleshooting, customer service,
              account help, technical problems, or support, return 'CUSTOMER_SUPPORT'
            - Respond with only 'RESEARCH' or 'CUSTOMER_SUPPORT'
            
            EXAMPLES:
            - "Write an article about AI trends" -> RESEARCH
            - "My password isn't working" -> CUSTOMER_SUPPORT  
            - "Analyze market data for renewable energy" -> RESEARCH
            - "How do I cancel my subscription?" -> CUSTOMER_SUPPORT
            """
        )
    
    def classify_query(self, state: MasterAgentState) -> MasterAgentState:
        """
        Classify the user query and update the state with the determined query type.
        
        Args:
            state: Current system state containing the user query
            
        Returns:
            Updated state with query_type field populated
        """
        log_agent_activity("RouterAgent", "classifying_query", {"query": state["user_query"]})
        
        try:
            # Simulate classification (in real implementation, call LLM)
            query = state["user_query"].lower()
            
            # Simple keyword-based classification for demonstration
            research_keywords = ["write", "research", "analyze", "study", "article", "report", "trends"]
            support_keywords = ["help", "problem", "issue", "error", "support", "cancel", "refund"]
            
            if any(keyword in query for keyword in research_keywords):
                query_type = QueryType.RESEARCH.value
            elif any(keyword in query for keyword in support_keywords):
                query_type = QueryType.CUSTOMER_SUPPORT.value
            else:
                # Default classification using more sophisticated logic
                query_type = QueryType.RESEARCH.value if len(query.split()) > 5 else QueryType.CUSTOMER_SUPPORT.value
            
            state["query_type"] = query_type
            state["active_agents"] = ["RouterAgent"]
            
            log_agent_activity("RouterAgent", "classification_complete", 
                             {"query_type": query_type})
            
        except Exception as e:
            log_agent_activity("RouterAgent", "classification_error", {"error": str(e)})
            state["query_type"] = QueryType.RESEARCH.value  # Default fallback
        
        return state


# ========================================================================================
# 5. RESEARCH AGENTS (LangGraph Workflow)
# ========================================================================================

class ResearchWorkflow:
    """
    LangGraph-based research workflow that coordinates multiple specialized agents
    to handle knowledge generation and content creation tasks.
    """
    
    def __init__(self, llm: ChatOpenAI, embeddings: OpenAIEmbeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def outliner_agent(self, state: MasterAgentState) -> MasterAgentState:
        """
        Outliner Agent: Analyzes the research topic and creates a structured outline
        for comprehensive content generation.
        """
        log_agent_activity("OutlinerAgent", "creating_outline", 
                          {"topic": state.get("topic", state["user_query"])})
        
        topic = state.get("topic", state["user_query"])
        
        # Simulate outline generation (in real implementation, use LLM)
        outline = {
            "title": f"Comprehensive Analysis: {topic}",
            "sections": [
                {
                    "title": "Introduction",
                    "description": "Overview and context setting",
                    "key_points": ["Background", "Scope", "Objectives"]
                },
                {
                    "title": "Main Analysis", 
                    "description": "Core content and findings",
                    "key_points": ["Key concepts", "Current trends", "Best practices"]
                },
                {
                    "title": "Implications and Future Directions",
                    "description": "Impact analysis and recommendations", 
                    "key_points": ["Practical applications", "Future trends", "Recommendations"]
                },
                {
                    "title": "Conclusion",
                    "description": "Summary and key takeaways",
                    "key_points": ["Main findings", "Action items"]
                }
            ],
            "estimated_length": "1500-2000 words",
            "target_audience": "General professional audience"
        }
        
        state["outline"] = outline
        state["topic"] = topic
        
        if "active_agents" not in state:
            state["active_agents"] = []
        state["active_agents"].append("OutlinerAgent")
        
        log_agent_activity("OutlinerAgent", "outline_completed", {"sections": len(outline["sections"])})
        return state
    
    def researcher_agent(self, state: MasterAgentState) -> MasterAgentState:
        """
        Researcher Agent: Gathers information, validates sources, and prepares
        research materials based on the outline.
        """
        log_agent_activity("ResearcherAgent", "gathering_information", 
                          {"topic": state["topic"]})
        
        # Simulate document retrieval and research
        # In real implementation, this would use vector databases, web scraping, etc.
        research_sections = []
        
        if state.get("outline"):
            for section in state["outline"]["sections"]:
                research_data = {
                    "section_title": section["title"],
                    "content": f"Researched content for {section['title']}. " + 
                              f"This section covers: {', '.join(section['key_points'])}. " +
                              "Based on current industry analysis and expert insights...",
                    "sources": [
                        f"Source 1: Industry Report on {state['topic']}",
                        f"Source 2: Academic Study - {section['title']}",
                        f"Source 3: Expert Analysis - {state['topic']} Trends"
                    ],
                    "confidence": 0.85,
                    "last_updated": datetime.now().isoformat()
                }
                research_sections.append(research_data)
        
        state["research_sections"] = research_sections
        state["active_agents"].append("ResearcherAgent")
        
        log_agent_activity("ResearcherAgent", "research_completed", 
                          {"sections_researched": len(research_sections)})
        return state
    
    def writer_agent(self, state: MasterAgentState) -> MasterAgentState:
        """
        Writer Agent: Synthesizes research into coherent, well-structured content
        following the established outline.
        """
        log_agent_activity("WriterAgent", "generating_content", 
                          {"sections_to_write": len(state.get("research_sections", []))})
        
        # Simulate content generation
        article_parts = []
        
        if state.get("outline") and state.get("research_sections"):
            article_parts.append(f"# {state['outline']['title']}\n")
            
            for section_data in state["research_sections"]:
                article_parts.append(f"\n## {section_data['section_title']}\n")
                article_parts.append(f"{section_data['content']}\n")
                
                # Add sources
                article_parts.append("\n### Sources:\n")
                for source in section_data["sources"]:
                    article_parts.append(f"- {source}\n")
        
        final_article = "".join(article_parts)
        
        # Add metadata
        final_article += f"\n\n---\n*Generated by Multi-Agent Research System*\n"
        final_article += f"*Topic: {state['topic']}*\n"
        final_article += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        state["final_article"] = final_article
        state["final_response"] = final_article
        state["active_agents"].append("WriterAgent")
        
        log_agent_activity("WriterAgent", "content_generation_completed", 
                          {"article_length": len(final_article)})
        return state


# ========================================================================================
# 6. SELF-RAG AGENT (Advanced Retrieval with Self-Reflection)
# ========================================================================================

class SelfRAGAgent:
    """
    Self-Reflective Retrieval-Augmented Generation Agent.
    
    This agent implements the Self-RAG framework with:
    - Adaptive retrieval (retrieves information only when needed)
    - Reflection tokens (evaluates its own outputs)
    - Factual grounding (ensures responses are supported by evidence)
    """
    
    def __init__(self, llm: ChatOpenAI, embeddings: OpenAIEmbeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = None
        self.reflection_tokens = {
            "RETRIEVE": "Should retrieve additional information",
            "NO_RETRIEVE": "Sufficient information available",
            "RELEVANT": "Retrieved information is relevant",
            "IRRELEVANT": "Retrieved information is not relevant",
            "SUPPORTED": "Response is supported by evidence",
            "NOT_SUPPORTED": "Response lacks sufficient evidence"
        }
    
    def setup_knowledge_base(self, documents: List[str]):
        """
        Initialize vector store with documents for retrieval.
        
        Args:
            documents: List of text documents to index
        """
        log_agent_activity("SelfRAGAgent", "initializing_knowledge_base", 
                          {"document_count": len(documents)})
        
        # Create document objects
        docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) 
                for i, doc in enumerate(documents)]
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(docs)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        
        log_agent_activity("SelfRAGAgent", "knowledge_base_ready", 
                          {"chunks_indexed": len(split_docs)})
    
    def should_retrieve(self, query: str, context: str = "") -> bool:
        """
        Determine if additional information retrieval is needed.
        
        Args:
            query: The user query
            context: Current context/knowledge
            
        Returns:
            True if retrieval is recommended, False otherwise
        """
        # Simple heuristic - in real implementation, use trained model
        if not context or len(context.split()) < 10:
            return True
        
        # Check if query contains specific terms that might need retrieval
        specific_terms = ["latest", "recent", "current", "statistics", "data", "research"]
        return any(term in query.lower() for term in specific_terms)
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents from the knowledge base.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            log_agent_activity("SelfRAGAgent", "documents_retrieved", 
                              {"query": query, "count": len(docs)})
            return docs
        except Exception as e:
            log_agent_activity("SelfRAGAgent", "retrieval_error", {"error": str(e)})
            return []
    
    def evaluate_relevance(self, query: str, documents: List[Document]) -> str:
        """
        Evaluate if retrieved documents are relevant to the query.
        
        Returns:
            "RELEVANT" or "IRRELEVANT" reflection token
        """
        if not documents:
            return "IRRELEVANT"
        
        # Simple relevance check - in real implementation, use more sophisticated methods
        query_terms = set(query.lower().split())
        doc_terms = set()
        
        for doc in documents:
            doc_terms.update(doc.page_content.lower().split())
        
        overlap = len(query_terms.intersection(doc_terms))
        relevance_score = overlap / len(query_terms) if query_terms else 0
        
        return "RELEVANT" if relevance_score > 0.3 else "IRRELEVANT"
    
    def generate_response_with_reflection(self, state: MasterAgentState) -> MasterAgentState:
        """
        Generate response using Self-RAG methodology with reflection tokens.
        """
        query = state["user_query"]
        log_agent_activity("SelfRAGAgent", "generating_response_with_reflection", 
                          {"query": query})
        
        # Step 1: Determine if retrieval is needed
        current_context = state.get("final_response", "")
        should_retrieve = self.should_retrieve(query, current_context)
        
        retrieved_docs = []
        if should_retrieve and self.vectorstore:
            # Step 2: Retrieve documents
            retrieved_docs = self.retrieve_documents(query)
            state["retrieved_documents"] = retrieved_docs
            
            # Step 3: Evaluate relevance
            relevance = self.evaluate_relevance(query, retrieved_docs)
            log_agent_activity("SelfRAGAgent", "relevance_evaluation", 
                              {"relevance": relevance})
        
        # Step 4: Generate response
        context_text = ""
        if retrieved_docs:
            context_text = "\n".join([doc.page_content for doc in retrieved_docs[:3]])
        
        # Simulate response generation (in real implementation, use LLM with context)
        if state.get("final_article"):
            enhanced_response = state["final_article"]
            if context_text:
                enhanced_response += f"\n\n## Additional Context\n{context_text}"
        else:
            enhanced_response = f"Response to: {query}\n\n"
            if context_text:
                enhanced_response += f"Based on retrieved information:\n{context_text}\n\n"
            enhanced_response += "Comprehensive answer generated using Self-RAG methodology..."
        
        # Step 5: Self-evaluation
        support_evaluation = self.evaluate_support(enhanced_response, retrieved_docs)
        
        state["final_response"] = enhanced_response
        state["confidence_score"] = 0.85 if support_evaluation == "SUPPORTED" else 0.65
        state["active_agents"].append("SelfRAGAgent")
        
        log_agent_activity("SelfRAGAgent", "response_generated", 
                          {"support_evaluation": support_evaluation,
                           "confidence": state["confidence_score"]})
        
        return state
    
    def evaluate_support(self, response: str, documents: List[Document]) -> str:
        """
        Evaluate if the response is supported by the retrieved documents.
        
        Returns:
            "SUPPORTED" or "NOT_SUPPORTED" reflection token
        """
        if not documents:
            return "NOT_SUPPORTED"
        
        # Simple support evaluation - check for factual alignment
        doc_content = " ".join([doc.page_content for doc in documents])
        
        # In real implementation, use more sophisticated fact-checking
        # For now, assume supported if documents are present and relevant
        return "SUPPORTED" if len(doc_content) > 100 else "NOT_SUPPORTED"


# ========================================================================================
# 7. CUSTOMER SUPPORT AGENTS (CrewAI)
# ========================================================================================

class CustomerSupportWorkflow:
    """
    CrewAI-based customer support workflow with specialized agents for different
    types of customer inquiries and issues.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize specialized customer support agents"""
        
        # FAQ Specialist Agent
        self.faq_agent = CrewAIAgent(
            role="FAQ Specialist",
            goal="Provide quick and accurate answers to frequently asked questions",
            backstory="""You are an expert in company products, services, and policies.
            You have extensive knowledge of common customer questions and can provide
            clear, concise answers that resolve customer queries efficiently.""",
            verbose=config.debug_mode,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=60
        )
        
        # Technical Troubleshooter Agent  
        self.troubleshooting_agent = CrewAIAgent(
            role="Technical Troubleshooter",
            goal="Solve technical problems with step-by-step guidance",
            backstory="""You are a technical expert who can diagnose and resolve
            common technical issues. You provide clear, step-by-step instructions
            and can escalate complex issues when necessary.""",
            verbose=config.debug_mode,
            allow_delegation=False,
            max_iter=5,
            max_execution_time=120
        )
        
        # Escalation Agent
        self.escalation_agent = CrewAIAgent(
            role="Escalation Specialist", 
            goal="Handle complex issues requiring human intervention",
            backstory="""You specialize in managing complex customer issues that
            require human intervention. You can identify when escalation is needed
            and prepare comprehensive handoff documentation.""",
            verbose=config.debug_mode,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=90
        )
    
    def categorize_support_query(self, query: str) -> str:
        """
        Categorize the support query to route to appropriate agent.
        
        Returns:
            "FAQ", "TECHNICAL", or "ESCALATION"
        """
        query_lower = query.lower()
        
        # FAQ keywords
        faq_keywords = ["how to", "what is", "where", "when", "cancel", "refund", "billing"]
        if any(keyword in query_lower for keyword in faq_keywords):
            return "FAQ"
        
        # Technical keywords  
        tech_keywords = ["error", "not working", "broken", "bug", "crash", "login", "password"]
        if any(keyword in query_lower for keyword in tech_keywords):
            return "TECHNICAL"
        
        # Default to FAQ for simple queries, escalation for complex ones
        return "FAQ" if len(query.split()) < 10 else "ESCALATION"
    
    def handle_support_query(self, state: MasterAgentState) -> MasterAgentState:
        """
        Process customer support query using appropriate specialized agent.
        """
        query = state["user_query"]
        log_agent_activity("CustomerSupportWorkflow", "processing_support_query", 
                          {"query": query})
        
        # Categorize the query
        category = self.categorize_support_query(query)
        state["support_category"] = category
        
        # Route to appropriate agent and generate response
        if category == "FAQ":
            response = self._handle_faq(query)
        elif category == "TECHNICAL":
            response = self._handle_technical(query)
        else:  # ESCALATION
            response = self._handle_escalation(query)
        
        state["final_response"] = response
        state["active_agents"].append(f"CustomerSupport-{category}")
        
        log_agent_activity("CustomerSupportWorkflow", "support_query_processed", 
                          {"category": category, "response_length": len(response)})
        
        return state
    
    def _handle_faq(self, query: str) -> str:
        """Handle FAQ queries"""
        # Simulate FAQ response generation
        response = f"""**FAQ Response for: {query}**

Thank you for your question. Based on our frequently asked questions database:

**Answer:**
Here's the information you requested. This is a common question that many customers ask.

**Additional Resources:**
- Check our online help center
- View step-by-step guides  
- Contact support if you need further assistance

**Was this helpful?** If you need more specific help, please don't hesitate to reach out.

*Response generated by FAQ Specialist Agent*
"""
        return response
    
    def _handle_technical(self, query: str) -> str:
        """Handle technical troubleshooting queries"""
        # Simulate technical troubleshooting response
        steps = [
            "Verify your internet connection",
            "Clear your browser cache and cookies",
            "Try using a different browser or device", 
            "Check if the issue persists",
            "Contact technical support if problem continues"
        ]
        
        response = f"""**Technical Support for: {query}**

I understand you're experiencing a technical issue. Let me help you resolve this step-by-step:

**Troubleshooting Steps:**
"""
        for i, step in enumerate(steps, 1):
            response += f"{i}. {step}\n"
        
        response += f"""
**If the issue persists:**
Please contact our technical support team with:
- Description of the problem
- Steps you've already tried
- Your device/browser information

*Response generated by Technical Troubleshooter Agent*
"""
        return response
    
    def _handle_escalation(self, query: str) -> str:
        """Handle escalation queries"""
        response = f"""**Escalation Required for: {query}**

Thank you for contacting support. I understand this is a complex issue that requires specialized attention.

**Your case has been escalated:**
- Case ID: ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}
- Priority: High
- Expected response time: 24-48 hours

**What happens next:**
1. A senior specialist will review your case
2. You'll receive a detailed response via email
3. If needed, we'll schedule a call to discuss further

**In the meantime:**
Please keep this case ID for reference and feel free to provide any additional information that might help us resolve your issue faster.

*Response generated by Escalation Specialist Agent*
"""
        return response


# ========================================================================================
# 8. RECURSIVE PROBLEM SOLVER (CrewAI)
# ========================================================================================

class RecursiveProblemSolver:
    """
    Advanced problem-solving agent that breaks down complex problems into
    manageable subtasks and solves them recursively.
    """
    
    def __init__(self, llm: ChatOpenAI, max_depth: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.solution_cache = {}
        
        # Problem decomposition agent
        self.decomposer_agent = CrewAIAgent(
            role="Problem Decomposer",
            goal="Break complex problems into manageable subtasks",
            backstory="""You are an expert at analyzing complex problems and breaking
            them down into smaller, solvable components. You excel at identifying
            dependencies and creating logical problem-solving workflows.""",
            verbose=config.debug_mode,
            allow_delegation=False
        )
        
        # Solution synthesizer agent
        self.synthesizer_agent = CrewAIAgent(
            role="Solution Synthesizer", 
            goal="Combine subtask solutions into comprehensive final solution",
            backstory="""You specialize in taking individual solution components and
            synthesizing them into coherent, comprehensive final solutions that
            address the original complex problem.""",
            verbose=config.debug_mode,
            allow_delegation=False
        )
    
    def solve_recursively(self, state: MasterAgentState) -> MasterAgentState:
        """
        Apply recursive problem-solving to the user query.
        """
        problem = state["user_query"]
        log_agent_activity("RecursiveProblemSolver", "starting_recursive_solve", 
                          {"problem": problem})
        
        # Decompose the problem
        subtasks = self._decompose_problem(problem)
        
        # Solve each subtask
        solutions = []
        for i, subtask in enumerate(subtasks):
            log_agent_activity("RecursiveProblemSolver", f"solving_subtask_{i+1}", 
                              {"subtask": subtask})
            solution = self._solve_subtask(subtask)
            solutions.append({"subtask": subtask, "solution": solution})
        
        # Synthesize final solution
        final_solution = self._synthesize_solutions(problem, solutions)
        
        state["final_response"] = final_solution
        state["active_agents"].append("RecursiveProblemSolver")
        
        log_agent_activity("RecursiveProblemSolver", "recursive_solve_completed", 
                          {"subtasks_solved": len(solutions)})
        
        return state
    
    def _decompose_problem(self, problem: str) -> List[str]:
        """Break down complex problem into subtasks"""
        # Simulate problem decomposition (in real implementation, use LLM)
        if "customer support" in problem.lower() or "help" in problem.lower():
            subtasks = [
                "Identify the specific customer issue",
                "Determine appropriate support category", 
                "Generate relevant solution steps",
                "Provide follow-up recommendations"
            ]
        else:
            # General problem decomposition
            subtasks = [
                "Analyze the core problem requirements",
                "Identify key components and dependencies",
                "Develop solution approach for each component",
                "Integrate components into final solution"
            ]
        
        return subtasks
    
    def _solve_subtask(self, subtask: str) -> str:
        """Solve individual subtask"""
        # Check cache first
        if subtask in self.solution_cache:
            return self.solution_cache[subtask]
        
        # Simulate subtask solution generation
        solution = f"Solution for '{subtask}': This involves analyzing the requirements " + \
                  f"and applying best practices to address {subtask.lower()}. " + \
                  "The approach includes systematic evaluation and implementation of proven methods."
        
        # Cache the solution
        self.solution_cache[subtask] = solution
        
        return solution
    
    def _synthesize_solutions(self, original_problem: str, solutions: List[Dict]) -> str:
        """Combine subtask solutions into final comprehensive solution"""
        
        final_solution = f"**Comprehensive Solution for: {original_problem}**\n\n"
        final_solution += "Through systematic problem decomposition and recursive solving, " + \
                         "here is your complete solution:\n\n"
        
        for i, sol_data in enumerate(solutions, 1):
            final_solution += f"**Step {i}: {sol_data['subtask']}**\n"
            final_solution += f"{sol_data['solution']}\n\n"
        
        final_solution += "**Summary:**\n"
        final_solution += "This comprehensive approach addresses all aspects of your query " + \
                         "through systematic problem-solving methodology. Each component has " + \
                         "been carefully analyzed and solved to provide you with a complete solution.\n\n"
        
        final_solution += f"*Generated by Recursive Problem Solver*\n"
        final_solution += f"*Subtasks solved: {len(solutions)}*"
        
        return final_solution


# ========================================================================================
# 9. EVALUATION SYSTEM (DeepEval Integration)
# ========================================================================================

class EvaluationSystem:
    """
    Comprehensive evaluation system using DeepEval metrics to assess
    agent response quality, factual accuracy, and relevance.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.hallucination_metric = HallucinationMetric(threshold=0.3)
        self.relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    
    def evaluate_response(self, state: MasterAgentState) -> MasterAgentState:
        """
        Comprehensive evaluation of the final response using multiple metrics.
        """
        if not config.enable_evaluation:
            return state
        
        query = state["user_query"]
        response = state["final_response"]
        retrieved_docs = state.get("retrieved_documents", [])
        
        log_agent_activity("EvaluationSystem", "starting_evaluation", 
                          {"query_length": len(query), "response_length": len(response)})
        
        # Simulate evaluation results (in real implementation, use actual DeepEval)
        evaluation_results = {
            "hallucination_score": self._evaluate_hallucination(response, retrieved_docs),
            "relevancy_score": self._evaluate_relevancy(query, response),
            "groundedness_score": self._evaluate_groundedness(response, retrieved_docs),
            "overall_quality": 0.0,
            "recommendations": []
        }
        
        # Calculate overall quality score
        evaluation_results["overall_quality"] = (
            evaluation_results["hallucination_score"] * 0.4 +
            evaluation_results["relevancy_score"] * 0.4 + 
            evaluation_results["groundedness_score"] * 0.2
        )
        
        # Generate recommendations
        if evaluation_results["hallucination_score"] < 0.7:
            evaluation_results["recommendations"].append("Consider fact-checking against reliable sources")
        
        if evaluation_results["relevancy_score"] < 0.7:
            evaluation_results["recommendations"].append("Improve response relevance to user query")
        
        if evaluation_results["groundedness_score"] < 0.7:
            evaluation_results["recommendations"].append("Provide more evidence-based information")
        
        state["evaluation_results"] = evaluation_results
        state["confidence_score"] = evaluation_results["overall_quality"]
        
        log_agent_activity("EvaluationSystem", "evaluation_completed", 
                          {"overall_quality": evaluation_results["overall_quality"]})
        
        return state
    
    def _evaluate_hallucination(self, response: str, documents: List[Document]) -> float:
        """Evaluate response for potential hallucinations"""
        # Simulate hallucination detection
        if not documents:
            return 0.6  # Moderate confidence without supporting documents
        
        # Simple check: longer responses with more specific claims need more evidence
        word_count = len(response.split())
        doc_support = len(documents) * 100  # Assume each doc provides 100 words of support
        
        support_ratio = min(doc_support / word_count, 1.0) if word_count > 0 else 1.0
        return 0.3 + (support_ratio * 0.7)  # Scale between 0.3 and 1.0
    
    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """Evaluate how relevant the response is to the query"""
        # Simple keyword overlap analysis
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 1.0
        
        overlap = len(query_words.intersection(response_words))
        relevancy = overlap / len(query_words)
        
        # Boost score if response addresses query structure
        if any(word in response.lower() for word in ["answer", "response", "solution"]):
            relevancy += 0.2
        
        return min(relevancy, 1.0)
    
    def _evaluate_groundedness(self, response: str, documents: List[Document]) -> float:
        """Evaluate how well the response is grounded in evidence"""
        if not documents:
            return 0.5  # Neutral score without documents
        
        # Check if response references or incorporates document content
        doc_content = " ".join([doc.page_content.lower() for doc in documents])
        response_lower = response.lower()
        
        # Simple overlap check
        common_phrases = 0
        response_sentences = response_lower.split('.')
        
        for sentence in response_sentences[:5]:  # Check first 5 sentences
            if len(sentence.strip()) > 10:  # Ignore very short sentences
                words = sentence.strip().split()
                if len(words) > 3:
                    phrase = " ".join(words[:4])  # Use first 4 words as phrase
                    if phrase in doc_content:
                        common_phrases += 1
        
        groundedness = common_phrases / min(5, len(response_sentences)) if response_sentences else 0
        return min(groundedness + 0.3, 1.0)  # Boost baseline and cap at 1.0


# ========================================================================================
# 10. MASTER AGENT ORCHESTRATOR (LangGraph)
# ========================================================================================

class MasterAgent:
    """
    Central orchestrator that manages the entire multi-agent system workflow
    using LangGraph for state management and agent coordination.
    """
    
    def __init__(self):
        # Initialize core components
        self.llm = initialize_llm(config.openai_model, config.temperature)
        self.embeddings = initialize_embeddings(config.embedding_model)
        
        # Initialize specialized workflows
        self.router = RouterAgent(self.llm)
        self.research_workflow = ResearchWorkflow(self.llm, self.embeddings)
        self.self_rag = SelfRAGAgent(self.llm, self.embeddings)
        self.support_workflow = CustomerSupportWorkflow(self.llm)
        self.problem_solver = RecursiveProblemSolver(self.llm)
        self.evaluator = EvaluationSystem(self.llm)
        
        # Initialize knowledge base for Self-RAG
        self._setup_knowledge_base()
        
        # Build the workflow graph
        self.workflow = self._build_workflow_graph()
        
        log_agent_activity("MasterAgent", "initialization_complete", 
                          {"components_loaded": 6})
    
    def _setup_knowledge_base(self):
        """Initialize the knowledge base with sample documents"""
        sample_documents = [
            """Self-Reflective Retrieval-Augmented Generation (SELF-RAG) is a framework 
            designed to improve the quality and factual accuracy of large language models 
            through on-demand retrieval and self-reflection. It uses reflection tokens to 
            evaluate relevance and support of retrieved information.""",
            
            """Multi-agent systems feature several independent agents acting in specialized 
            roles that collaborate to achieve collective objectives. They provide advantages 
            through specialization, inter-agent coordination, and dynamic learning in 
            changing environments.""",
            
            """Customer support best practices include: quick response times, personalized 
            service, proactive communication, comprehensive knowledge bases, and escalation 
            procedures for complex issues. Technical troubleshooting should follow systematic 
            step-by-step approaches.""",
            
            """Recent AI trends include generative AI applications, multimodal models, 
            AI automation in various industries, ethical AI development, and the integration 
            of AI agents in business workflows. These trends are shaping the future of 
            technology and business operations."""
        ]
        
        self.self_rag.setup_knowledge_base(sample_documents)
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow that orchestrates all agents.
        This creates a stateful graph that manages the flow between different agents.
        """
        # Create the state graph
        workflow = StateGraph(MasterAgentState)
        
        # Add agent nodes
        workflow.add_node("router", self.router.classify_query)
        workflow.add_node("research_outliner", self.research_workflow.outliner_agent)
        workflow.add_node("research_researcher", self.research_workflow.researcher_agent)  
        workflow.add_node("research_writer", self.research_workflow.writer_agent)
        workflow.add_node("self_rag", self.self_rag.generate_response_with_reflection)
        workflow.add_node("support_handler", self.support_workflow.handle_support_query)
        workflow.add_node("problem_solver", self.problem_solver.solve_recursively)
        workflow.add_node("evaluator", self.evaluator.evaluate_response)
        
        # Define the workflow edges
        workflow.add_edge(START, "router")
        
        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "router",
            self._route_query,
            {
                "research": "research_outliner",
                "support": "support_handler"
            }
        )
        
        # Research workflow sequence
        workflow.add_edge("research_outliner", "research_researcher")
        workflow.add_edge("research_researcher", "research_writer") 
        workflow.add_edge("research_writer", "self_rag")
        workflow.add_edge("self_rag", "evaluator")
        
        # Support workflow routes
        workflow.add_conditional_edges(
            "support_handler",
            self._route_support,
            {
                "simple": "evaluator",
                "complex": "problem_solver"
            }
        )
        workflow.add_edge("problem_solver", "evaluator")
        
        # End workflow after evaluation
        workflow.add_edge("evaluator", END)
        
        # Compile with checkpointing for state persistence
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _route_query(self, state: MasterAgentState) -> str:
        """Route query based on classification"""
        query_type = state.get("query_type", "")
        return "research" if query_type == QueryType.RESEARCH.value else "support"
    
    def _route_support(self, state: MasterAgentState) -> str:
        """Route support queries to appropriate handler"""
        support_category = state.get("support_category", "FAQ")
        # Route complex issues to problem solver, simple ones directly to evaluation
        return "complex" if support_category == "ESCALATION" else "simple"
    
    def process_query(self, user_query: str, config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries through the multi-agent system.
        
        Args:
            user_query: The user's input query
            config_dict: Optional configuration overrides
            
        Returns:
            Dictionary containing the final response and metadata
        """
        start_time = datetime.now()
        
        log_agent_activity("MasterAgent", "processing_query", 
                          {"query": user_query, "start_time": start_time.isoformat()})
        
        # Initialize state
        initial_state = MasterAgentState(
            user_query=user_query,
            query_type=None,
            topic=None,
            outline=None,
            research_sections=None,
            final_article=None,
            support_category=None,
            solution_steps=None,
            escalation_needed=None,
            retrieved_documents=None,
            evaluation_results=None,
            final_response="",
            active_agents=[],
            processing_time=None,
            confidence_score=None
        )
        
        try:
            # Run the workflow
            final_state = None
            for state in self.workflow.stream(initial_state, config={"thread_id": "main"}):
                final_state = state
            
            # Extract final state values
            if final_state:
                # Get the last state from the stream
                last_key = list(final_state.keys())[-1]
                result_state = final_state[last_key]
            else:
                result_state = initial_state
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            result_state["processing_time"] = processing_time
            
            # Prepare response
            response = {
                "success": True,
                "response": result_state["final_response"],
                "query_type": result_state.get("query_type"),
                "active_agents": result_state.get("active_agents", []),
                "confidence_score": result_state.get("confidence_score"),
                "evaluation_results": result_state.get("evaluation_results"),
                "processing_time": processing_time,
                "timestamp": end_time.isoformat()
            }
            
            log_agent_activity("MasterAgent", "query_processed_successfully", 
                              {"processing_time": processing_time, 
                               "agents_used": len(result_state.get("active_agents", []))})
            
            return response
            
        except Exception as e:
            error_time = (datetime.now() - start_time).total_seconds()
            log_agent_activity("MasterAgent", "query_processing_error", 
                              {"error": str(e), "processing_time": error_time})
            
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "processing_time": error_time,
                "timestamp": datetime.now().isoformat()
            }


# ========================================================================================
# 11. EXAMPLE USAGE AND TESTING
# ========================================================================================

def demonstrate_system():
    """
    Demonstrate the multi-agent system with different types of queries.
    """
    print("Multi-Agent System Demonstration")
    print("=" * 60)
    
    # Initialize the system
    master_agent = MasterAgent()
    
    # Test queries
    test_queries = [
        {
            "query": "Write an article about the latest trends in artificial intelligence",
            "expected_type": "RESEARCH",
            "description": "Research workflow with article generation"
        },
        {
            "query": "My password is not working and I can't log in",
            "expected_type": "CUSTOMER_SUPPORT", 
            "description": "Technical support workflow"
        },
        {
            "query": "How do I cancel my subscription and get a refund?",
            "expected_type": "CUSTOMER_SUPPORT",
            "description": "FAQ support workflow"
        },
        {
            "query": "Analyze the market potential for renewable energy solutions in Southeast Asia",
            "expected_type": "RESEARCH", 
            "description": "Complex research with Self-RAG"
        }
    ]
    
    # Process each test query
    for i, test_case in enumerate(test_queries, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print("-" * 50)
        
        # Process the query
        result = master_agent.process_query(test_case["query"])
        
        if result["success"]:
            print(f"Success!")
            print(f"Query Type: {result['query_type']}")
            print(f"Agents Used: {', '.join(result['active_agents'])}")
            print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            
            # Show full response
            print(f"The Full Response:\n {result["response"]}")
            
            if result.get("evaluation_results"):
                eval_results = result["evaluation_results"]
                print(f"Evaluation Score: {eval_results.get('overall_quality', 'N/A'):.2f}")
        else:
            print(f"Error: {result['error']}")
        
        print()
    
    print("Demonstration completed!")
    print("\nThis system showcases:")
    print("- OpenAI agent routing and classification")
    print("- LangGraph workflow orchestration and state management")
    print("- CrewAI collaborative agent teams")
    print("- Self-RAG with reflection and retrieval")
    print("- Comprehensive evaluation with DeepEval metrics")
    print("- Recursive problem solving for complex issues")


# ========================================================================================
# 12. MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    """
    Main execution block - demonstrates the multi-agent system capabilities.
    """
    
    print("Starting Multi-Agent System...")
    print("This comprehensive system integrates multiple AI frameworks:")
    print("- OpenAI for intelligent routing")
    print("- LangGraph for workflow orchestration") 
    print("- CrewAI for collaborative agents")
    print("- DeepEval for quality assurance")
    print()
    
    # Check configuration
    if config.openai_api_key == "your-api-key-here":
        print("? Warning: Please set your OpenAI API key in the configuration")
        print("   Set the OPENAI_API_KEY environment variable or update the config")
        print()
    
    # Run demonstration
    try:
        demonstrate_system()
    except Exception as e:
        print(f"System error: {str(e)}")
        print("Please check your configuration and dependencies")
    
    print("\nNext Steps:")
    print("- Set up your OpenAI API key")
    print("- Install required dependencies (langchain, crewai, langgraph, deepeval)")
    print("- Customize agent prompts and workflows for your specific use case")
    print("- Integrate with your existing systems and databases")
    print("- Add more specialized agents as needed")
    
    print("\nSystem Architecture Summary:")
    print("This implementation provides a production-ready foundation for")
    print("building sophisticated multi-agent AI systems that can handle")
    print("complex workflows with proper evaluation and monitoring.")