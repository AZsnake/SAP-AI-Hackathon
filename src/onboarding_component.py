import os
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

# Optional imports with fallback handling
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("LangChain not available - using mock implementations")
    LANGCHAIN_AVAILABLE = False

try:
    from colorama import Fore, Style
except ImportError:
    class Fore:
        RED = GREEN = BLUE = YELLOW = ""
    class Style:
        RESET_ALL = ""


# ========================================================================================
# CONFIGURATION AND DATA STRUCTURES
# ========================================================================================

@dataclass
class OnboardingConfig:
    """Configuration for onboarding workflow component."""
    documents_path: str = "onboarding_docs"
    vector_store_path: str = "vector_stores/onboarding_faiss"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    relevance_threshold: float = 0.7
    enable_self_correction: bool = True


@dataclass
class RAGResponse:
    """Structured response from RAG pipeline."""
    answer: str
    retrieved_docs: List[Document]
    confidence_score: float
    sources: List[str]
    processing_steps: List[str]


# ========================================================================================
# MOCK IMPLEMENTATIONS
# ========================================================================================

class MockDocument:
    """Mock document class when LangChain unavailable."""
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockVectorStore:
    """Mock vector store for testing without LangChain."""
    def __init__(self, documents: List):
        self.documents = documents
    
    def similarity_search(self, query: str, k: int = 5) -> List[MockDocument]:
        """Return mock relevant documents."""
        return self.documents[:k]
    
    def save_local(self, path: str):
        """Mock save operation."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def load_local(cls, path: str, embeddings):
        """Mock load operation."""
        return cls([MockDocument("Mock company policy document", {"source": "mock.pdf"})])


class MockEmbeddings:
    """Mock embeddings when OpenAI unavailable."""
    def __init__(self):
        pass


class MockLLM:
    """Mock LLM for testing."""
    def invoke(self, prompt: str) -> str:
        if "company policy" in prompt.lower():
            return "According to company policies, employees should follow standard procedures."
        return "I can help you with onboarding questions based on company documentation."


# ========================================================================================
# DOCUMENT MANAGEMENT
# ========================================================================================

class DocumentManager:
    """
    Manages onboarding documents and their indexing.
    Handles document loading, chunking, and change detection.
    """
    
    def __init__(self, config: OnboardingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        ) if LANGCHAIN_AVAILABLE else None
    
    def load_documents(self) -> List[Document]:
        """
        Load all onboarding documents from configured directory.
        
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        docs_path = Path(self.config.documents_path)
        
        if not docs_path.exists():
            print(f"Warning: Documents directory {docs_path} not found.")
            return self._get_sample_documents()
        
        # Supported file extensions
        supported_extensions = {'.txt', '.md', '.pdf', '.docx'}
        
        for file_path in docs_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    content = self._read_file(file_path)
                    if content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "last_modified": file_path.stat().st_mtime,
                                "file_type": file_path.suffix[1:]
                            }
                        ) if LANGCHAIN_AVAILABLE else MockDocument(content, {"source": str(file_path)})
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents for onboarding knowledge base.")
        return documents
    
    def _read_file(self, file_path: Path) -> str:
        """Read content from various file types."""
        try:
            if file_path.suffix.lower() in ['.txt', '.md']:
                return file_path.read_text(encoding='utf-8')
            elif file_path.suffix.lower() == '.pdf':
                # In production, use PyPDF2 or similar
                return f"PDF content from {file_path.name} would be extracted here"
            elif file_path.suffix.lower() == '.docx':
                # In production, use python-docx
                return f"DOCX content from {file_path.name} would be extracted here"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def _get_sample_documents(self) -> List[Document]:
        """Generate sample onboarding documents for demonstration."""
        sample_docs = [
            {
                "content": """
                COMPANY HANDBOOK - REMOTE WORK POLICY
                
                Our company supports flexible remote work arrangements. Employees may work 
                remotely up to 3 days per week with manager approval. Remote workers must:
                - Maintain regular communication with their team
                - Attend mandatory meetings via video conference
                - Ensure reliable internet and appropriate workspace
                - Follow all company security protocols for remote access
                
                Equipment provided: Laptop, monitor, and necessary software licenses.
                """,
                "metadata": {"source": "remote_work_policy.md", "category": "policies"}
            },
            {
                "content": """
                NEW EMPLOYEE ONBOARDING CHECKLIST
                
                Week 1:
                - Complete IT setup and security training
                - Meet with direct manager and team members
                - Review company mission, values, and organizational chart
                - Complete required HR paperwork and benefits enrollment
                
                Week 2-4:
                - Shadow experienced team members
                - Complete job-specific training modules
                - Set initial goals and expectations with manager
                - Join relevant team meetings and standups
                """,
                "metadata": {"source": "onboarding_checklist.md", "category": "process"}
            },
            {
                "content": """
                COMPANY BENEFITS OVERVIEW
                
                Health Insurance:
                - Medical, dental, and vision coverage
                - Company pays 80% of premiums
                - Open enrollment in November
                
                Time Off:
                - 15 PTO days (increasing with tenure)
                - 10 holidays per year
                - Sick leave as needed
                
                Professional Development:
                - $2000 annual learning budget
                - Conference attendance encouraged
                - Mentorship program available
                """,
                "metadata": {"source": "benefits_guide.md", "category": "benefits"}
            }
        ]
        
        return [
            Document(page_content=doc["content"], metadata=doc["metadata"]) 
            if LANGCHAIN_AVAILABLE else MockDocument(doc["content"], doc["metadata"])
            for doc in sample_docs
        ]
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        if not self.text_splitter:
            return documents
        
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")
        return chunked_docs
    
    def get_documents_hash(self, documents: List[Document]) -> str:
        """Generate hash of documents to detect changes."""
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()


# ========================================================================================
# VECTOR STORE MANAGEMENT
# ========================================================================================

class VectorStoreManager:
    """
    Manages vector store operations with persistence and incremental updates.
    Implements best practices for efficient vector storage and retrieval.
    """
    
    def __init__(self, config: OnboardingConfig, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.vector_store = None
        self.documents_hash = None
        
    def get_or_create_vector_store(self, documents: List[Document]) -> Any:
        """
        Load existing vector store or create new one if needed.
        
        Args:
            documents: Documents to index if creating new store
            
        Returns:
            Configured vector store instance
        """
        store_path = Path(self.config.vector_store_path)
        hash_path = store_path.parent / "documents.hash"
        
        current_hash = self._calculate_documents_hash(documents)
        
        # Check if existing store is valid
        if self._should_use_existing_store(store_path, hash_path, current_hash):
            print("Loading existing vector store...")
            try:
                if LANGCHAIN_AVAILABLE:
                    self.vector_store = FAISS.load_local(
                        str(store_path), 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    self.vector_store = MockVectorStore.load_local(str(store_path), self.embeddings)
                
                self.documents_hash = current_hash
                print("Vector store loaded successfully.")
                return self.vector_store
                
            except Exception as e:
                print(f"Error loading existing store: {e}. Creating new one...")
        
        # Create new vector store
        print("Creating new vector store...")
        self.vector_store = self._create_new_store(documents, store_path, current_hash)
        return self.vector_store
    
    def _should_use_existing_store(self, store_path: Path, hash_path: Path, current_hash: str) -> bool:
        """Check if existing vector store should be used."""
        if not store_path.exists() or not hash_path.exists():
            return False
        
        try:
            stored_hash = hash_path.read_text().strip()
            return stored_hash == current_hash
        except Exception:
            return False
    
    def _create_new_store(self, documents: List[Document], store_path: Path, current_hash: str) -> Any:
        """Create and persist new vector store."""
        try:
            # Create vector store
            if LANGCHAIN_AVAILABLE and documents:
                vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                vector_store = MockVectorStore(documents)
            
            # Ensure directory exists
            store_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save vector store and hash
            vector_store.save_local(str(store_path))
            (store_path.parent / "documents.hash").write_text(current_hash)
            
            self.documents_hash = current_hash
            print(f"Vector store created and saved with {len(documents)} documents.")
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return MockVectorStore(documents)
    
    def _calculate_documents_hash(self, documents: List[Document]) -> str:
        """Calculate hash of all documents for change detection."""
        content = "".join(sorted([doc.page_content for doc in documents]))
        return hashlib.md5(content.encode()).hexdigest()
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search on vector store."""
        if not self.vector_store:
            print("Warning: Vector store not initialized")
            return []
        
        k = k or self.config.top_k_retrieval
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []


# ========================================================================================
# SELF-RAG IMPLEMENTATION
# ========================================================================================

class SelfRAGPipeline:
    """
    Self-Reflective Retrieval-Augmented Generation pipeline.
    Implements retrieve -> generate -> evaluate -> refine workflow.
    """
    
    def __init__(self, llm, vector_manager: VectorStoreManager, config: OnboardingConfig):
        self.llm = llm
        self.vector_manager = vector_manager
        self.config = config
        
        # Define prompts for different stages
        self.generation_prompt = PromptTemplate(
            template="""You are a helpful HR assistant answering employee onboarding questions.

Context Documents:
{context}

Employee Question: {question}

Instructions:
- Provide accurate information based only on the provided context
- If information is not available in the context, clearly state this
- Be helpful and professional in your response
- Include specific policy details when relevant
- If multiple sources apply, synthesize the information clearly

Answer:""",
            input_variables=["context", "question"]
        ) if LANGCHAIN_AVAILABLE else None
        
        self.evaluation_prompt = PromptTemplate(
            template="""Evaluate this HR response for accuracy and completeness.

Question: {question}
Response: {response}
Available Context: {context}

Evaluation Criteria:
1. Accuracy: Is the response factually correct based on the context?
2. Completeness: Does it fully address the question?
3. Relevance: Is the information relevant to the question?
4. Clarity: Is the response clear and professional?

Provide a score from 0.0 to 1.0 and brief explanation.
Format: SCORE: X.X | EXPLANATION: [brief explanation]""",
            input_variables=["question", "response", "context"]
        ) if LANGCHAIN_AVAILABLE else None
    
    def process_query(self, query: str) -> RAGResponse:
        """
        Process onboarding query through complete self-RAG pipeline.
        
        Args:
            query: Employee's onboarding question
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        processing_steps = []
        start_time = datetime.now()
        
        # Step 1: Retrieve relevant documents
        processing_steps.append("Retrieving relevant documents...")
        retrieved_docs = self.vector_manager.similarity_search(query)
        
        if not retrieved_docs:
            return RAGResponse(
                answer="I don't have specific information about that topic in our onboarding materials. Please contact HR directly for assistance.",
                retrieved_docs=[],
                confidence_score=0.0,
                sources=[],
                processing_steps=processing_steps + ["No relevant documents found"]
            )
        
        # Step 2: Generate initial response
        processing_steps.append("Generating response...")
        context = self._format_context(retrieved_docs)
        initial_response = self._generate_response(query, context)
        
        # Step 3: Self-evaluation (if enabled)
        confidence_score = 0.8  # Default confidence
        final_response = initial_response
        
        if self.config.enable_self_correction:
            processing_steps.append("Evaluating response quality...")
            confidence_score = self._evaluate_response(query, initial_response, context)
            
            # Step 4: Refine if needed (low confidence)
            if confidence_score < self.config.relevance_threshold:
                processing_steps.append("Refining response...")
                final_response = self._refine_response(query, initial_response, context, confidence_score)
        
        # Extract source information
        sources = list(set([
            doc.metadata.get("source", "Unknown") 
            for doc in retrieved_docs
        ]))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        processing_steps.append(f"Completed in {processing_time:.2f} seconds")
        
        return RAGResponse(
            answer=final_response,
            retrieved_docs=retrieved_docs,
            confidence_score=confidence_score,
            sources=sources,
            processing_steps=processing_steps
        )
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM with retrieved context."""
        try:
            if self.generation_prompt and LANGCHAIN_AVAILABLE:
                prompt = self.generation_prompt.format(question=query, context=context)
            else:
                prompt = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content.strip()
            return str(response).strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your question. Please try again or contact HR directly."
    
    def _evaluate_response(self, query: str, response: str, context: str) -> float:
        """Evaluate response quality and return confidence score."""
        try:
            if self.evaluation_prompt and LANGCHAIN_AVAILABLE:
                eval_prompt = self.evaluation_prompt.format(
                    question=query, 
                    response=response, 
                    context=context
                )
            else:
                eval_prompt = f"Rate this response from 0.0 to 1.0: {response}"
            
            evaluation = self.llm.invoke(eval_prompt)
            
            if hasattr(evaluation, 'content'):
                eval_text = evaluation.content
            else:
                eval_text = str(evaluation)
            
            # Extract score from evaluation
            if "SCORE:" in eval_text:
                score_part = eval_text.split("SCORE:")[1].split("|")[0].strip()
                try:
                    return float(score_part)
                except ValueError:
                    pass
            
            return 0.7  # Default moderate confidence
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return 0.5  # Default neutral confidence
    
    def _refine_response(self, query: str, initial_response: str, context: str, confidence: float) -> str:
        """Refine response if confidence is low."""
        refine_prompt = f"""
        The initial response to this onboarding question had low confidence ({confidence:.2f}).
        Please provide a better, more accurate response.

        Question: {query}
        Initial Response: {initial_response}
        Context: {context}

        Improved Response:"""
        
        try:
            refined = self.llm.invoke(refine_prompt)
            if hasattr(refined, 'content'):
                return refined.content.strip()
            return str(refined).strip()
        except Exception as e:
            print(f"Error refining response: {e}")
            return initial_response  # Fallback to original


# ========================================================================================
# MAIN ONBOARDING WORKFLOW
# ========================================================================================

class OnboardingWorkflow:
    """
    Main orchestrator for onboarding-related queries.
    Combines document management, vector storage, and self-RAG pipeline.
    """
    
    def __init__(self, llm, embeddings, config: OnboardingConfig = None):
        """
        Initialize onboarding workflow with all components.
        
        Args:
            llm: Language model for generation and evaluation
            embeddings: Embedding model for vector operations
            config: Configuration object (uses defaults if None)
        """
        self.config = config or OnboardingConfig()
        self.llm = llm
        
        # Initialize components
        self.doc_manager = DocumentManager(self.config)
        self.vector_manager = VectorStoreManager(self.config, embeddings)
        
        # Load documents and initialize vector store
        print("Initializing onboarding knowledge base...")
        documents = self.doc_manager.load_documents()
        chunked_docs = self.doc_manager.chunk_documents(documents)
        self.vector_store = self.vector_manager.get_or_create_vector_store(chunked_docs)
        
        # Initialize RAG pipeline
        self.rag_pipeline = SelfRAGPipeline(llm, self.vector_manager, self.config)
        
        print("Onboarding workflow initialized successfully.")
    
    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Handle onboarding query and return comprehensive response.
        
        Args:
            query: Employee's onboarding question
            
        Returns:
            Dictionary containing response and metadata
        """
        print(f"{Fore.BLUE}Processing onboarding query:{Style.RESET_ALL} {query[:100]}...")
        
        try:
            # Process through RAG pipeline
            rag_response = self.rag_pipeline.process_query(query)
            
            # Format response for display
            response = {
                "answer": rag_response.answer,
                "confidence": rag_response.confidence_score,
                "sources": rag_response.sources,
                "retrieved_documents_count": len(rag_response.retrieved_docs),
                "processing_steps": rag_response.processing_steps,
                "workflow": "onboarding",
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"{Fore.GREEN}Response generated with {rag_response.confidence_score:.2f} confidence{Style.RESET_ALL}")
            return response
            
        except Exception as e:
            print(f"{Fore.RED}Error in onboarding workflow: {e}{Style.RESET_ALL}")
            return {
                "answer": "I apologize, but I encountered an error processing your onboarding question. Please contact HR directly for assistance.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_documents_count": 0,
                "processing_steps": [f"Error: {str(e)}"],
                "workflow": "onboarding",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        return {
            "vector_store_initialized": self.vector_store is not None,
            "config": {
                "documents_path": self.config.documents_path,
                "vector_store_path": self.config.vector_store_path,
                "chunk_size": self.config.chunk_size,
                "top_k_retrieval": self.config.top_k_retrieval,
                "relevance_threshold": self.config.relevance_threshold,
                "self_correction_enabled": self.config.enable_self_correction
            },
            "documents_hash": self.vector_manager.documents_hash,
            "langchain_available": LANGCHAIN_AVAILABLE
        }


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def create_sample_onboarding_docs(docs_path: str = "onboarding_docs"):
    """Create sample onboarding documents for testing."""
    docs_dir = Path(docs_path)
    docs_dir.mkdir(exist_ok=True)
    
    sample_files = {
        "employee_handbook.md": """
# Employee Handbook

## Company Mission
We strive to create innovative solutions that make a positive impact on the world.

## Core Values
- Integrity: We act with honesty and transparency
- Innovation: We embrace new ideas and creative solutions  
- Collaboration: We work together to achieve common goals
- Excellence: We deliver high-quality work and continuous improvement

## Communication Guidelines
- Use Slack for daily communication
- Email for formal communications
- All-hands meetings every Friday at 2 PM
- Open door policy with management
        """,
        
        "benefits_guide.md": """
# Employee Benefits Guide

## Health & Wellness
- Comprehensive health insurance (medical, dental, vision)
- Company pays 80% of premiums
- $500 wellness stipend annually
- On-site gym membership

## Time Off
- 15 PTO days for first year (increases with tenure)
- 12 company holidays
- Unlimited sick leave
- Parental leave: 12 weeks paid

## Professional Development  
- $3000 annual learning budget
- Conference attendance supported
- Internal mentorship program
- Career development planning
        """,
        
        "it_security_policy.md": """
# IT Security Policy

## Password Requirements
- Minimum 12 characters
- Must include uppercase, lowercase, numbers, symbols
- Changed every 90 days
- No password reuse for last 12 passwords

## Remote Work Security
- VPN required for all remote connections
- Encrypted laptops only
- No public WiFi for sensitive work
- Screen locks after 5 minutes of inactivity

## Data Protection
- All company data must be stored in approved systems
- No personal cloud storage for work files  
- Report security incidents immediately
- Regular security training required
        """
    }
    
    for filename, content in sample_files.items():
        file_path = docs_dir / filename
        file_path.write_text(content.strip())
    
    print(f"Created {len(sample_files)} sample onboarding documents in {docs_path}/")


if __name__ == "__main__":
    # Demo the onboarding component
    print("Onboarding Component Demo")
    print("=" * 50)
    
    # Create sample documents
    create_sample_onboarding_docs()
    
    # Initialize mock components
    llm = MockLLM()
    embeddings = MockEmbeddings()
    
    # Initialize workflow
    workflow = OnboardingWorkflow(llm, embeddings)
    
    # Test queries
    test_queries = [
        "What are the company's core values?",
        "How many PTO days do I get as a new employee?",
        "What are the password requirements for company systems?",
        "Tell me about the professional development budget"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = workflow.handle_query(query)
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {', '.join(response['sources'])}")