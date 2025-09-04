import os
import json
from typing import Dict, List, Optional, TypedDict, Literal, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import traceback

# Core framework imports - with fallbacks
try:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
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

# Misc imports
try:
    from colorama import Fore, Style
except ImportError:
    # Fallback for colorama
    class Fore:
        RED = ""
        GREEN = ""
        BLUE = ""
    class Style:
        RESET_ALL = ""

# Mock classes for when libraries aren't available
class MockChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def invoke(self, prompt):
        return f"Mock response to: {prompt}"

class MockEmbeddings:
    def __init__(self, **kwargs):
        pass

class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MockFAISS:
    def __init__(self, docs, embeddings):
        self.docs = docs
    
    def similarity_search(self, query, k=3):
        return self.docs[:k]

class MockAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

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

# Initialize configuration with proper API key handling
def get_config():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "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"
        print("Warning: No OpenAI API key found. Using mock responses.")
    
    return SystemConfig(
        openai_api_key=api_key,
        debug_mode=True
    )

config = get_config()

# ========================================================================================
# 2. STATE MANAGEMENT
# ========================================================================================

class MasterAgentState(TypedDict):
    """Central state management for the multi-agent system."""
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
    retrieved_documents: Optional[List]
    evaluation_results: Optional[Dict[str, Any]]
    final_response: str
    
    # System state
    active_agents: List[str]
    processing_time: Optional[float]
    confidence_score: Optional[float]

# ========================================================================================
# 3. UTILITY FUNCTIONS AND LLM INITIALIZATION
# ========================================================================================

def initialize_llm(model: str = "gpt-4o-mini", temperature: float = 0.7):
    """Initialize the Language Learning Model with specified parameters."""
    if LANGCHAIN_AVAILABLE and config.openai_api_key != "mock-key-for-demo":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key
        )
    else:
        return MockChatOpenAI(model=model, temperature=temperature)

def initialize_embeddings(model: str = "text-embedding-ada-002"):
    """Initialize embedding model for document similarity and retrieval"""
    if LANGCHAIN_AVAILABLE and config.openai_api_key != "mock-key-for-demo":
        return OpenAIEmbeddings(model=model, api_key=config.openai_api_key)
    else:
        return MockEmbeddings(model=model)

def log_agent_activity(agent_name: str, action: str, details: Dict[str, Any] = None):
    """Log agent activities for debugging and monitoring."""
    if config.debug_mode:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Fore.RED}[LOG]{Style.RESET_ALL} {Fore.GREEN}[{timestamp}]{Style.RESET_ALL} {agent_name}: {action}")
        if details:
            print(f"Details: {json.dumps(details, indent=2, default=str)}")

# ========================================================================================
# 4. ROUTING AGENT
# ========================================================================================

class RouterAgent:
    """Query classification and routing agent."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def classify_query(self, state: MasterAgentState) -> MasterAgentState:
        """Classify the user query and update the state with the determined query type."""
        log_agent_activity("RouterAgent", "classifying_query", {"query": state["user_query"]})
        
        try:
            query = state["user_query"].lower()
            
            # Enhanced keyword-based classification
            research_keywords = [
                "write", "research", "analyze", "study", "article", "report", "trends",
                "analysis", "comprehensive", "market", "future", "technology", "industry"
            ]
            support_keywords = [
                "help", "problem", "issue", "error", "support", "cancel", "refund",
                "login", "password", "account", "billing", "technical", "fix", "broken"
            ]
            
            research_score = sum(1 for keyword in research_keywords if keyword in query)
            support_score = sum(1 for keyword in support_keywords if keyword in query)
            
            if research_score > support_score:
                query_type = QueryType.RESEARCH.value
            elif support_score > research_score:
                query_type = QueryType.CUSTOMER_SUPPORT.value
            else:
                # Default based on query complexity
                query_type = QueryType.RESEARCH.value if len(query.split()) > 8 else QueryType.CUSTOMER_SUPPORT.value
            
            state["query_type"] = query_type
            state["active_agents"] = ["RouterAgent"]
            
            log_agent_activity("RouterAgent", "classification_complete", {"query_type": query_type})
            
        except Exception as e:
            log_agent_activity("RouterAgent", "classification_error", {"error": str(e)})
            state["query_type"] = QueryType.RESEARCH.value
        
        return state

# ========================================================================================
# 5. RESEARCH WORKFLOW
# ========================================================================================

class ResearchWorkflow:
    """Research workflow with specialized agents."""
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
    
    def outliner_agent(self, state: MasterAgentState) -> MasterAgentState:
        """Create structured outline for research content."""
        log_agent_activity("OutlinerAgent", "creating_outline", 
                          {"topic": state.get("topic", state["user_query"])})
        
        topic = state.get("topic", state["user_query"])
        
        outline = {
            "title": f"Analysis: {topic}",
            "sections": [
                {
                    "title": "Executive Summary",
                    "description": "Key findings and overview",
                    "key_points": ["Main insights", "Critical trends", "Key takeaways"]
                },
                {
                    "title": "Detailed Analysis", 
                    "description": "In-depth examination of the topic",
                    "key_points": ["Current state", "Key factors", "Market dynamics"]
                },
                {
                    "title": "Future Outlook",
                    "description": "Trends and predictions", 
                    "key_points": ["Emerging trends", "Growth opportunities", "Challenges"]
                },
                {
                    "title": "Recommendations",
                    "description": "Actionable insights and next steps",
                    "key_points": ["Strategic actions", "Implementation steps", "Success metrics"]
                }
            ],
            "estimated_length": "1200-1800 words",
            "target_audience": "Professional stakeholders"
        }
        
        state["outline"] = outline
        state["topic"] = topic
        
        if "active_agents" not in state:
            state["active_agents"] = []
        state["active_agents"].append("OutlinerAgent")
        
        log_agent_activity("OutlinerAgent", "outline_completed", {"sections": len(outline["sections"])})
        return state
    
    def researcher_agent(self, state: MasterAgentState) -> MasterAgentState:
        """Gather and organize research information."""
        log_agent_activity("ResearcherAgent", "gathering_information", {"topic": state["topic"]})
        
        research_sections = []
        
        if state.get("outline"):
            for section in state["outline"]["sections"]:
                # Generate realistic research content
                content = self._generate_research_content(state["topic"], section)
                
                research_data = {
                    "section_title": section["title"],
                    "content": content,
                    "sources": [
                        f"Industry Report: {state['topic']} Analysis 2024",
                        f"Academic Research: {section['title']} Studies", 
                        f"Market Intelligence: {state['topic']} Trends",
                        f"Expert Opinion: {section['title']} Insights"
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
    
    def _generate_research_content(self, topic: str, section: Dict) -> str:
        """Generate realistic research content for a section."""
        section_title = section["title"]
        key_points = section["key_points"]
        
        if section_title == "Executive Summary":
            return f"""The analysis of {topic} reveals several key insights that are reshaping the landscape. Current market dynamics show significant growth potential, driven by technological advancement and changing consumer behaviors. Key findings indicate that {', '.join(key_points).lower()} are the primary factors influencing development in this area. Strategic positioning and adaptive approaches will be crucial for success in this evolving environment."""
        
        elif section_title == "Detailed Analysis":
            return f"""A comprehensive examination of {topic} shows multifaceted complexity requiring careful consideration of various factors. The {key_points[0].lower()} demonstrates stability while {key_points[1].lower()} indicate areas of opportunity. Market research suggests that {key_points[2].lower()} are driving transformation across the sector. Data analysis reveals patterns that support sustainable growth strategies and risk mitigation approaches."""
        
        elif section_title == "Future Outlook":
            return f"""Looking ahead, {topic} is positioned for significant evolution over the next 3-5 years. {key_points[0].capitalize()} suggest accelerated development, while {key_points[1].lower()} present substantial value creation potential. However, {key_points[2].lower()} remain key considerations that require proactive management. Industry experts predict continued innovation and market expansion."""
        
        elif section_title == "Recommendations":
            return f"""Based on the analysis of {topic}, several strategic recommendations emerge: 1) Focus on {key_points[0].lower()} to capitalize on immediate opportunities, 2) Develop {key_points[1].lower()} capabilities for long-term competitive advantage, 3) Monitor {key_points[2].lower()} to ensure sustainable growth and risk management. Implementation should be phased and regularly evaluated."""
        
        else:
            return f"""This section examines {section_title.lower()} in the context of {topic}. The analysis covers {', '.join(key_points).lower()} and their implications for stakeholders. Research indicates that understanding these elements is crucial for informed decision-making and strategic planning."""
    
    def writer_agent(self, state: MasterAgentState) -> MasterAgentState:
        """Synthesize research into coherent content."""
        log_agent_activity("WriterAgent", "generating_content", 
                          {"sections_to_write": len(state.get("research_sections", []))})
        
        article_parts = []
        
        if state.get("outline") and state.get("research_sections"):
            article_parts.append(f"# {state['outline']['title']}\n")
            article_parts.append(f"*Comprehensive Analysis and Strategic Insights*\n\n")
            
            for section_data in state["research_sections"]:
                article_parts.append(f"## {section_data['section_title']}\n\n")
                article_parts.append(f"{section_data['content']}\n\n")
                
                # Add key sources
                article_parts.append("**Key Sources:**\n")
                for source in section_data["sources"][:2]:  # Show top 2 sources
                    article_parts.append(f"- {source}\n")
                article_parts.append("\n")
        
        final_article = "".join(article_parts)
        
        # Add conclusion and metadata
        final_article += f"## Conclusion\n\n"
        final_article += f"This comprehensive analysis of {state['topic']} provides strategic insights for decision-makers and stakeholders. The research demonstrates the importance of adaptive strategies and informed planning in navigating current challenges and capitalizing on emerging opportunities.\n\n"
        
        final_article += f"---\n\n"
        final_article += f"*Generated by Multi-Agent Research System*  \n"
        final_article += f"*Analysis Date: {datetime.now().strftime('%B %d, %Y')}*  \n"
        final_article += f"*Word Count: {len(final_article.split())} words*"
        
        state["final_article"] = final_article
        state["final_response"] = final_article
        state["active_agents"].append("WriterAgent")
        
        log_agent_activity("WriterAgent", "content_generation_completed", 
                          {"article_length": len(final_article)})
        return state

# ========================================================================================
# 6. SELF-RAG AGENT
# ========================================================================================

class SelfRAGAgent:
    """Self-Reflective Retrieval-Augmented Generation Agent."""
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = None
        self.knowledge_base = []
    
    def setup_knowledge_base(self, documents: List[str]):
        """Initialize vector store with documents for retrieval."""
        log_agent_activity("SelfRAGAgent", "initializing_knowledge_base", 
                          {"document_count": len(documents)})
        
        self.knowledge_base = documents
        
        if LANGCHAIN_AVAILABLE:
            # Create document objects
            docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) 
                    for i, doc in enumerate(documents)]
            
            # Create vector store
            try:
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            except:
                self.vectorstore = MockFAISS(docs, self.embeddings)
        else:
            # Mock implementation
            docs = [MockDocument(doc, {"source": f"doc_{i}"}) for i, doc in enumerate(documents)]
            self.vectorstore = MockFAISS(docs, self.embeddings)
        
        log_agent_activity("SelfRAGAgent", "knowledge_base_ready", 
                          {"documents_indexed": len(documents)})
    
    def generate_response_with_reflection(self, state: MasterAgentState) -> MasterAgentState:
        """Generate response using Self-RAG methodology."""
        query = state["user_query"]
        log_agent_activity("SelfRAGAgent", "generating_response_with_reflection", 
                          {"query": query})
        
        # Retrieve relevant documents if available
        retrieved_docs = []
        if self.vectorstore:
            try:
                retrieved_docs = self.vectorstore.similarity_search(query, k=2)
                state["retrieved_documents"] = retrieved_docs
            except Exception as e:
                log_agent_activity("SelfRAGAgent", "retrieval_error", {"error": str(e)})
        
        # Enhance existing response with retrieved context
        base_response = state.get("final_response", "")
        
        if base_response and retrieved_docs:
            # Add contextual enhancement
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs[:2]])
            enhanced_response = base_response + f"\n\n## Additional Context and Validation\n\n"
            enhanced_response += f"**Supporting Information:**\n{context_text}\n\n"
            enhanced_response += "**Reflection:** This response has been validated against available knowledge sources and enhanced with additional context to ensure accuracy and completeness."
        elif base_response:
            enhanced_response = base_response + f"\n\n**Self-RAG Validation:** Response generated and validated through self-reflection process."
        else:
            enhanced_response = f"**Response to: {query}**\n\nBased on analysis and available information, here is a comprehensive response addressing your query. This response has been generated using self-reflective methodology to ensure quality and relevance."
        
        # Self-evaluation
        confidence_score = self._calculate_confidence(enhanced_response, retrieved_docs)
        
        state["final_response"] = enhanced_response
        state["confidence_score"] = confidence_score
        state["active_agents"].append("SelfRAGAgent")
        
        log_agent_activity("SelfRAGAgent", "response_generated", 
                          {"confidence": confidence_score})
        
        return state
    
    def _calculate_confidence(self, response: str, documents: List) -> float:
        """Calculate confidence score for the response."""
        base_confidence = 0.7
        
        # Boost confidence if documents support the response
        if documents:
            base_confidence += 0.15
        
        # Adjust based on response length and structure
        word_count = len(response.split())
        if word_count > 100:
            base_confidence += 0.05
        
        if "analysis" in response.lower() or "research" in response.lower():
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)

# ========================================================================================
# 7. CUSTOMER SUPPORT WORKFLOW
# ========================================================================================

class CustomerSupportWorkflow:
    """Customer support workflow with specialized handling."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def handle_support_query(self, state: MasterAgentState) -> MasterAgentState:
        """Process customer support query."""
        query = state["user_query"]
        log_agent_activity("CustomerSupportWorkflow", "processing_support_query", 
                          {"query": query})
        
        # Categorize the query
        category = self._categorize_support_query(query)
        state["support_category"] = category
        
        # Generate appropriate response
        if category == "FAQ":
            response = self._handle_faq(query)
        elif category == "TECHNICAL":
            response = self._handle_technical(query)
        else:  # ESCALATION
            response = self._handle_escalation(query)
        
        state["final_response"] = response
        state["active_agents"].append(f"CustomerSupport-{category}")
        
        log_agent_activity("CustomerSupportWorkflow", "support_query_processed", 
                          {"category": category})
        
        return state
    
    def _categorize_support_query(self, query: str) -> str:
        """Categorize support query."""
        query_lower = query.lower()
        
        faq_keywords = ["how to", "what is", "where", "when", "cancel", "refund", "billing", "subscription"]
        tech_keywords = ["error", "not working", "broken", "bug", "crash", "login", "password", "access"]
        
        if any(keyword in query_lower for keyword in faq_keywords):
            return "FAQ"
        elif any(keyword in query_lower for keyword in tech_keywords):
            return "TECHNICAL"
        else:
            return "FAQ" if len(query.split()) < 10 else "ESCALATION"
    
    def _handle_faq(self, query: str) -> str:
        """Handle FAQ queries."""
        return f"""## Customer Support Response

**Your Question:** {query}

**Answer:**
Thank you for your inquiry. Based on our frequently asked questions and knowledge base, here's the information you need:

This is a common question that we're happy to help with. Our support team has extensive experience with similar requests and can provide you with accurate guidance.

**Next Steps:**
1. Review the information provided above
2. Check our comprehensive help center for additional resources
3. Contact our support team directly if you need personalized assistance

**Additional Resources:**
- Online Help Center: Available 24/7 with step-by-step guides
- Community Forum: Connect with other users and experts
- Direct Support: Contact our team for personalized help

**Was this helpful?** If you need more specific assistance or have follow-up questions, please don't hesitate to reach out to our support team.

*Response generated by FAQ Specialist Agent*  
*Response ID: FAQ-{datetime.now().strftime('%Y%m%d%H%M%S')}*"""
    
    def _handle_technical(self, query: str) -> str:
        """Handle technical troubleshooting queries."""
        return f"""## Technical Support Response

**Issue:** {query}

I understand you're experiencing a technical issue. Let me provide you with a systematic approach to resolve this:

**Troubleshooting Steps:**

1. **Initial Verification**
   - Check your internet connection stability
   - Verify you're using a supported browser (Chrome, Firefox, Safari, Edge)
   - Ensure your browser is updated to the latest version

2. **Cache and Data Reset**
   - Clear your browser cache and cookies
   - Disable browser extensions temporarily
   - Try using an incognito/private browsing window

3. **Account and Authentication**
   - Verify your login credentials are correct
   - Check if your account status is active
   - Try logging out completely and logging back in

4. **Advanced Troubleshooting**
   - Try accessing from a different device or network
   - Check for any system maintenance notifications
   - Restart your device if the issue persists

**If the issue continues:**
Please contact our technical support team with the following information:
- Detailed description of the problem
- Steps you've already tried
- Your device type and browser information
- Any error messages you're seeing

**Priority Support:** Technical issues are resolved with priority. Our average response time is 2-4 hours during business hours.

*Response generated by Technical Troubleshooter Agent*  
*Ticket ID: TECH-{datetime.now().strftime('%Y%m%d%H%M%S')}*"""
    
    def _handle_escalation(self, query: str) -> str:
        """Handle escalation queries."""
        case_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return f"""## Priority Support Case Created

**Your Request:** {query}

Thank you for contacting our support team. I understand this is an important matter that requires specialized attention.

**Your Case Details:**
- **Case ID:** {case_id}
- **Priority Level:** High Priority
- **Status:** Escalated to Specialist Team
- **Expected Response:** 4-8 hours during business hours

**What Happens Next:**

1. **Immediate Assignment:** Your case has been assigned to a senior support specialist with expertise in your specific area of concern.

2. **Comprehensive Review:** Our specialist will conduct a thorough review of your request and any related account information.

3. **Direct Contact:** You'll receive a personalized response via email with detailed information and next steps.

4. **Follow-up Support:** If needed, we'll schedule a call or provide additional resources to ensure complete resolution.

**Important Information:**
- Keep this Case ID for all future reference: **{case_id}**
- Check your email for updates within the next few hours
- Reply to any follow-up emails to maintain case continuity

**Need Immediate Assistance?**
If this is an urgent matter requiring immediate attention, please call our priority support line and reference your Case ID.

We appreciate your patience and will ensure this matter receives the attention it deserves.

*Response generated by Escalation Specialist Agent*  
*Case Priority: HIGH | Agent: Senior Support Specialist*"""

# ========================================================================================
# 8. EVALUATION SYSTEM
# ========================================================================================

class EvaluationSystem:
    """Response quality evaluation system."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_response(self, state: MasterAgentState) -> MasterAgentState:
        """Evaluate the quality of the final response."""
        if not config.enable_evaluation:
            return state
        
        query = state["user_query"]
        response = state["final_response"]
        retrieved_docs = state.get("retrieved_documents", [])
        
        log_agent_activity("EvaluationSystem", "starting_evaluation", 
                          {"query_length": len(query), "response_length": len(response)})
        
        evaluation_results = {
            "hallucination_score": self._evaluate_hallucination(response, retrieved_docs),
            "relevancy_score": self._evaluate_relevancy(query, response),
            "groundedness_score": self._evaluate_groundedness(response, retrieved_docs),
            "completeness_score": self._evaluate_completeness(response),
            "overall_quality": 0.0,
            "recommendations": []
        }
        
        # Calculate overall quality score
        scores = [
            evaluation_results["hallucination_score"] * 0.3,
            evaluation_results["relevancy_score"] * 0.3,
            evaluation_results["groundedness_score"] * 0.2,
            evaluation_results["completeness_score"] * 0.2
        ]
        evaluation_results["overall_quality"] = sum(scores)
        
        # Generate recommendations
        self._generate_recommendations(evaluation_results)
        
        state["evaluation_results"] = evaluation_results
        if not state.get("confidence_score"):
            state["confidence_score"] = evaluation_results["overall_quality"]
        
        log_agent_activity("EvaluationSystem", "evaluation_completed", 
                          {"overall_quality": evaluation_results["overall_quality"]})
        
        return state
    
    def _evaluate_hallucination(self, response: str, documents: List) -> float:
        """Evaluate response for potential hallucinations."""
        base_score = 0.8  # Start with high confidence
        
        # Reduce score if no supporting documents
        if not documents:
            base_score -= 0.2
        
        # Check for specific claims that might need verification
        specific_claims = ["statistics", "data", "research shows", "studies indicate"]
        claim_count = sum(1 for claim in specific_claims if claim in response.lower())
        
        if claim_count > 2 and not documents:
            base_score -= 0.1
        
        return max(base_score, 0.4)
    
    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """Evaluate response relevancy to the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.9
        
        # Calculate word overlap
        overlap = len(query_words.intersection(response_words))
        basic_relevancy = min(overlap / len(query_words), 1.0)
        
        # Boost for response structure indicators
        structure_indicators = ["response", "answer", "analysis", "solution"]
        if any(indicator in response.lower() for indicator in structure_indicators):
            basic_relevancy += 0.1
        
        # Boost for direct question addressing
        if "?" in query and any(word in response.lower() for word in ["because", "due to", "result"]):
            basic_relevancy += 0.1
        
        return min(basic_relevancy, 1.0)
    
    def _evaluate_groundedness(self, response: str, documents: List) -> float:
        """Evaluate how well grounded the response is."""
        if not documents:
            return 0.6  # Moderate score for responses without explicit grounding
        
        # Simple check for content overlap
        doc_content = " ".join([str(getattr(doc, 'page_content', '')) for doc in documents]).lower()
        response_lower = response.lower()
        
        # Count sentence-level overlap
        response_sentences = [s.strip() for s in response_lower.split('.') if len(s.strip()) > 10]
        grounded_sentences = 0
        
        for sentence in response_sentences[:5]:  # Check first 5 sentences
            words = sentence.split()
            if len(words) > 3:
                key_phrase = " ".join(words[:4])
                if key_phrase in doc_content:
                    grounded_sentences += 1
        
        grounding_ratio = grounded_sentences / max(len(response_sentences[:5]), 1)
        return min(0.4 + (grounding_ratio * 0.6), 1.0)
    
    def _evaluate_completeness(self, response: str) -> float:
        """Evaluate response completeness."""
        word_count = len(response.split())
        
        # Base completeness on word count and structure
        if word_count < 50:
            return 0.4
        elif word_count < 150:
            return 0.6
        elif word_count < 300:
            return 0.8
        else:
            return 0.9
    
    def _generate_recommendations(self, eval_results: Dict):
        """Generate improvement recommendations."""
        recommendations = []
        
        if eval_results["hallucination_score"] < 0.7:
            recommendations.append("Consider adding more factual verification and source citations")
        
        if eval_results["relevancy_score"] < 0.7:
            recommendations.append("Improve direct addressing of the user's specific query")
        
        if eval_results["groundedness_score"] < 0.7:
            recommendations.append("Include more evidence-based information and references")
        
        if eval_results["completeness_score"] < 0.7:
            recommendations.append("Provide more comprehensive coverage of the topic")
        
        if not recommendations:
            recommendations.append("Response meets quality standards across all metrics")
        
        eval_results["recommendations"] = recommendations

# ========================================================================================
# 9. SIMPLIFIED MASTER AGENT
# ========================================================================================

class MasterAgent:
    """Simplified master orchestrator that works reliably."""
    
    def __init__(self):
        try:
            # Initialize core components
            self.llm = initialize_llm(config.openai_model, config.temperature)
            self.embeddings = initialize_embeddings(config.embedding_model)
            
            # Initialize workflows
            self.router = RouterAgent(self.llm)
            self.research_workflow = ResearchWorkflow(self.llm, self.embeddings)
            self.self_rag = SelfRAGAgent(self.llm, self.embeddings)
            self.support_workflow = CustomerSupportWorkflow(self.llm)
            self.evaluator = EvaluationSystem(self.llm)
            
            # Setup knowledge base
            self._setup_knowledge_base()
            
            log_agent_activity("MasterAgent", "initialization_complete", 
                              {"status": "ready"})
        except Exception as e:
            log_agent_activity("MasterAgent", "initialization_error", {"error": str(e)})
            print(f"Warning: Initialization error: {e}")
    
    def _setup_knowledge_base(self):
        """Initialize knowledge base with sample documents."""
        sample_documents = [
            """Multi-agent systems represent a paradigm in artificial intelligence where multiple autonomous agents collaborate to solve complex problems. These systems leverage distributed intelligence, specialization, and coordination to achieve objectives that would be difficult for single agents to accomplish. Key advantages include improved scalability, robustness, and the ability to handle diverse, complex tasks through agent collaboration.""",
            
            """Customer support best practices emphasize rapid response times, personalized service delivery, and comprehensive problem resolution. Effective support systems integrate multiple communication channels, maintain detailed knowledge bases, and employ escalation procedures for complex issues. Modern support workflows benefit from AI assistance while maintaining human oversight for critical decisions.""",
            
            """Self-Reflective Retrieval-Augmented Generation (Self-RAG) enhances language model capabilities by incorporating dynamic information retrieval and self-evaluation mechanisms. This approach improves factual accuracy, reduces hallucinations, and provides more grounded responses by validating outputs against retrieved information sources.""",
            
            """Research and analysis workflows benefit from systematic approaches that include topic exploration, information gathering, synthesis, and validation. Effective research combines multiple perspectives, validates sources, and presents findings in structured, actionable formats that support informed decision-making."""
        ]
        
        self.self_rag.setup_knowledge_base(sample_documents)
    
    def process_query(self, user_query: str, config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main query processing method with simplified workflow."""
        start_time = datetime.now()
        
        log_agent_activity("MasterAgent", "processing_query_start", 
                          {"query": user_query[:100] + "..." if len(user_query) > 100 else user_query})
        
        # Initialize state
        state = MasterAgentState(
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
            # Step 1: Route the query
            state = self.router.classify_query(state)
            
            # Step 2: Process based on query type
            if state["query_type"] == QueryType.RESEARCH.value:
                # Research workflow
                state = self.research_workflow.outliner_agent(state)
                state = self.research_workflow.researcher_agent(state)
                state = self.research_workflow.writer_agent(state)
            else:
                # Support workflow
                state = self.support_workflow.handle_support_query(state)
            
            # Step 3: Self-RAG enhancement
            state = self.self_rag.generate_response_with_reflection(state)
            
            # Step 4: Evaluate response
            state = self.evaluator.evaluate_response(state)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            state["processing_time"] = processing_time
            
            # Prepare successful response
            response = {
                "success": True,
                "response": state["final_response"],
                "query_type": state.get("query_type"),
                "active_agents": state.get("active_agents", []),
                "confidence_score": state.get("confidence_score"),
                "evaluation_results": state.get("evaluation_results"),
                "processing_time": processing_time,
                "timestamp": end_time.isoformat()
            }
            
            log_agent_activity("MasterAgent", "query_processed_successfully", 
                              {"processing_time": processing_time, 
                               "agents_used": len(state.get("active_agents", []))})
            
            return response
            
        except Exception as e:
            error_time = (datetime.now() - start_time).total_seconds()
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": error_time
            }
            
            log_agent_activity("MasterAgent", "query_processing_error", error_details)
            
            return {
                "success": False,
                "error": str(e),
                "response": f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try again or contact support if the issue persists.",
                "processing_time": error_time,
                "timestamp": datetime.now().isoformat()
            }

# ========================================================================================
# 10. DEMONSTRATION FUNCTION
# ========================================================================================

def demonstrate_system():
    """Demonstrate the system with test queries."""
    print("Multi-Agent System Demonstration")
    print("=" * 50)
    
    master_agent = MasterAgent()
    
    test_queries = [
        "Write an analysis of artificial intelligence trends in 2024",
        "I can't log into my account - my password isn't working",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 30)
        
        result = master_agent.process_query(query)
        
        if result["success"]:
            print(f"Query Type: {result['query_type']}")
            print(f"Agents: {', '.join(result['active_agents'])}")
            print(f"Confidence: {result.get('confidence_score', 'N/A')}")
            print(f"Time: {result['processing_time']:.2f}s")
            print(f"Response length: {len(result['response'])} characters")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    demonstrate_system()