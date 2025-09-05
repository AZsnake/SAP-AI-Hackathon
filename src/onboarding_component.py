import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
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
    workflow_metadata: Dict[str, Any]


# ========================================================================================
# MOCK IMPLEMENTATIONS
# ========================================================================================


class MockDocument:
    """Mock document class for fallback when LangChain unavailable."""

    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockVectorStore:
    """Mock vector store for fallback when LangChain unavailable."""

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
        mock_docs = [
            MockDocument(
                "Mock company policy document about remote work and security protocols",
                {"source": "remote_policy.pdf"},
            ),
            MockDocument(
                "Mock employee handbook with core values and benefits information",
                {"source": "handbook.pdf"},
            ),
        ]
        return cls(mock_docs)


class MockLLM:
    """Mock LLM for fallback when LangChain unavailable."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt: str) -> str:
        """Return mock responses based on prompt content."""
        prompt_lower = prompt.lower()

        if "values" in prompt_lower or "culture" in prompt_lower:
            return "Our company values integrity, innovation, and collaboration. These guide our daily work through transparent communication, creative problem-solving, and team-based decision making."
        elif "benefits" in prompt_lower or "pto" in prompt_lower:
            return "New employees receive 15 PTO days annually, comprehensive health insurance with 80% company contribution, and access to our $3000 annual learning budget."
        elif "security" in prompt_lower or "password" in prompt_lower:
            return "Company security requires 12-character passwords with mixed case, numbers, and symbols. VPN access is mandatory for remote work."
        elif "evaluate" in prompt_lower or "score" in prompt_lower:
            return "SCORE: 0.8 | EXPLANATION: Response is comprehensive and addresses the query based on available context."
        else:
            return "I can help you with onboarding questions based on our company documentation and policies."


# ========================================================================================
# DOCUMENT MANAGEMENT
# ========================================================================================


class DocumentManager:
    """Document manager for loading and processing onboarding documents."""

    def __init__(self, config: OnboardingConfig):
        self.config = config
        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            if LANGCHAIN_AVAILABLE
            else None
        )

    def load_documents(self) -> List[Document]:
        """Load documents from the configured path."""
        documents = []
        docs_path = Path(self.config.documents_path)

        if not docs_path.exists():
            print(f"Documents directory not found. Creating sample documents...")
            self._create_sample_documents(docs_path)

        supported_extensions = {".txt", ".md"}

        try:
            for file_path in docs_path.rglob("*"):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if content.strip():
                            doc = (
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": str(file_path),
                                        "filename": file_path.name,
                                    },
                                )
                                if LANGCHAIN_AVAILABLE
                                else MockDocument(
                                    content,
                                    {
                                        "source": str(file_path),
                                        "filename": file_path.name,
                                    },
                                )
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        except Exception as e:
            print(f"Error in document loading: {e}")
            return self._get_sample_documents()

        print(f"Loaded {len(documents)} documents for onboarding knowledge base.")
        return documents

    def _create_sample_documents(self, docs_path: Path):
        """Create sample onboarding documents."""
        docs_path.mkdir(parents=True, exist_ok=True)

        sample_content = {
            "company_handbook.md": """
# Company Handbook

## Mission Statement
Our mission is to create innovative technology solutions that make a positive impact while fostering a collaborative workplace.

## Core Values
1. **Integrity**: We act with honesty and transparency in all interactions
2. **Innovation**: We embrace creativity and new solutions to complex problems  
3. **Collaboration**: We work together to achieve shared goals
4. **Excellence**: We strive for high quality and continuous improvement
5. **Diversity & Inclusion**: We value diverse perspectives and create an inclusive environment

## Communication Guidelines
- Primary communication: Slack for daily coordination
- Formal communications: Email for official announcements
- Team meetings: All-hands meetings every Friday at 2:00 PM
- Open door policy: Management accessible for questions
- Documentation: Important decisions must be documented

## Work Environment
We support a hybrid work model balancing flexibility with collaboration.
            """,
            "benefits_guide.md": """
# Employee Benefits Guide

## Health & Wellness Benefits
- **Medical Insurance**: Comprehensive coverage, company pays 80% of premiums
- **Dental & Vision**: Full coverage included
- **Mental Health**: Access to counseling services
- **Wellness Stipend**: $500 annually for fitness activities

## Time Off Policies
- **PTO**: 15 days first year, 20 days after 2 years, 25 days after 5 years
- **Company Holidays**: 12 official holidays plus floating holidays
- **Sick Leave**: Unlimited sick leave for health needs
- **Parental Leave**: 12 weeks paid for primary caregivers, 6 weeks for secondary

## Professional Development
- **Learning Budget**: $3,000 annually for courses and certifications
- **Conference Attendance**: Company-sponsored industry conferences
- **Mentorship Program**: Formal mentoring with senior staff
- **Career Development**: Quarterly planning sessions with managers

## Financial Benefits
- **401(k) Plan**: Company matches up to 6% of salary
- **Stock Options**: Equity participation for full-time employees
- **Performance Bonuses**: Annual performance-based compensation
- **Referral Bonuses**: $2,000 for successful employee referrals
            """,
            "it_security_policy.md": """
# IT Security Policy

## Password and Authentication
- **Password Complexity**: Minimum 12 characters with uppercase, lowercase, numbers, and symbols
- **Password Rotation**: Must be changed every 90 days
- **Multi-Factor Authentication**: Required for all company systems
- **Password Manager**: Company-provided password manager mandatory

## Remote Work Security
- **VPN Access**: Mandatory VPN for all remote work
- **Device Encryption**: Full disk encryption required on all laptops
- **Network Security**: No public WiFi for sensitive work
- **Screen Security**: Automatic locks after 5 minutes
- **Physical Security**: Secure storage of company devices

## Data Protection
- **Data Classification**: Public, Internal, Confidential, or Restricted
- **Storage Requirements**: Company data only in approved systems
- **Data Sharing**: No personal cloud storage for company files
- **Email Security**: Encrypted email for sensitive communications
- **Incident Reporting**: Immediate reporting of security incidents

## Compliance and Training
- **Security Training**: Mandatory training every 6 months
- **Phishing Tests**: Regular simulated attacks
- **Policy Updates**: Notifications within 48 hours
- **Compliance Audits**: Regular security audits
            """,
        }

        for filename, content in sample_content.items():
            file_path = docs_path / filename
            file_path.write_text(content.strip())

        print(f"Created {len(sample_content)} sample documents.")

    def _get_sample_documents(self) -> List[Document]:
        """Generate sample onboarding documents."""
        sample_docs = [
            {
                "content": """
                COMPANY REMOTE WORK POLICY
                
                Our flexible remote work program supports productivity and work-life balance:
                
                **Eligibility:**
                - All full-time employees after 90-day period
                - Up to 3 days per week with manager approval
                - Fully remote arrangements case-by-case
                
                **Requirements:**
                - Regular communication via Slack and video calls
                - Attend mandatory meetings with camera enabled
                - Respond to messages within 4 hours during business hours
                - Quiet, professional workspace for video calls
                
                **Technology:**
                - Company laptop with security configurations
                - High-speed internet (minimum 25 Mbps)
                - VPN connection mandatory for company systems
                """,
                "metadata": {"source": "remote_policy.md", "category": "policies"},
            },
            {
                "content": """
                NEW EMPLOYEE ONBOARDING CHECKLIST
                
                **Week 1: Foundation**
                - Complete IT setup and security training
                - Meet with manager for role expectations
                - Review company mission, values, and structure
                - Complete HR paperwork and benefits enrollment
                - Attend new employee orientation
                
                **Week 2-3: Integration**
                - Shadow team members in relevant departments
                - Complete job-specific training modules
                - Begin starter projects with mentorship
                - Join team meetings and collaborative sessions
                
                **Week 4: Assessment**
                - 30-day check-in with manager and HR
                - Set 90-day performance goals
                - Provide onboarding feedback
                - Begin independent project work
                """,
                "metadata": {
                    "source": "onboarding_checklist.md",
                    "category": "process",
                },
            },
        ]

        return [
            (
                Document(page_content=doc["content"], metadata=doc["metadata"])
                if LANGCHAIN_AVAILABLE
                else MockDocument(doc["content"], doc["metadata"])
            )
            for doc in sample_docs
        ]

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces for better retrieval."""
        if not self.text_splitter:
            return documents

        chunked_docs = []
        for doc in documents:
            try:
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update(
                        {
                            "chunk_id": f"{doc.metadata.get('filename', 'unknown')}_{i}",
                            "chunk_index": i,
                            "original_source": doc.metadata.get("source", "unknown"),
                        }
                    )
                chunked_docs.extend(chunks)
            except Exception as e:
                print(
                    f"Error chunking document {doc.metadata.get('source', 'unknown')}: {e}"
                )
                chunked_docs.append(doc)

        print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")
        return chunked_docs


# ========================================================================================
# VECTOR STORE MANAGEMENT
# ========================================================================================


class VectorStoreManager:
    """Vector store management for document retrieval."""

    def __init__(self, config: OnboardingConfig, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.vector_store = None

    def get_or_create_vector_store(self, documents: List[Document]) -> Any:
        """Get existing vector store or create new one."""
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
                        allow_dangerous_deserialization=True,
                    )
                else:
                    self.vector_store = MockVectorStore.load_local(
                        str(store_path), self.embeddings
                    )

                print("Vector store loaded successfully.")
                return self.vector_store

            except Exception as e:
                print(f"Error loading existing store: {e}. Creating new one...")

        # Create new vector store
        print("Creating new vector store...")
        self.vector_store = self._create_new_store(documents, store_path, current_hash)
        return self.vector_store

    def _should_use_existing_store(
        self, store_path: Path, hash_path: Path, current_hash: str
    ) -> bool:
        """Check if existing vector store is valid."""
        if not store_path.exists() or not hash_path.exists():
            return False

        try:
            stored_hash = hash_path.read_text().strip()
            return stored_hash == current_hash
        except Exception:
            return False

    def _create_new_store(
        self, documents: List[Document], store_path: Path, current_hash: str
    ) -> Any:
        """Create new vector store."""
        try:
            # Create vector store
            if LANGCHAIN_AVAILABLE and documents:
                vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                vector_store = MockVectorStore(documents)

            # Save vector store and hash
            store_path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(store_path))
            (store_path.parent / "documents.hash").write_text(current_hash)

            print(f"Vector store created with {len(documents)} documents.")
            return vector_store

        except Exception as e:
            print(f"Error creating vector store: {e}")
            return MockVectorStore(documents)

    def _calculate_documents_hash(self, documents: List[Document]) -> str:
        """Calculate hash of documents for change detection."""
        content = "".join(sorted([doc.page_content for doc in documents]))
        return hashlib.md5(content.encode()).hexdigest()

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Search for similar documents."""
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
# SELF-RAG PIPELINE
# ========================================================================================


class SelfRAGPipeline:
    """Self-Reflective Retrieval-Augmented Generation pipeline."""

    def __init__(
        self, llm, vector_manager: VectorStoreManager, config: OnboardingConfig
    ):
        self.llm = llm
        self.vector_manager = vector_manager
        self.config = config

        # Generation prompt
        self.generation_prompt = (
            PromptTemplate(
                template="""You are an expert HR assistant helping employees with onboarding questions.

Context Information:
{context}

Employee Question: {question}

Instructions:
- Provide accurate information based strictly on the provided context
- If specific information is not available, clearly state this limitation
- Be professional, helpful, and detailed in your response
- Include relevant policy details when applicable
- Format your response clearly

Professional Response:""",
                input_variables=["context", "question"],
            )
            if LANGCHAIN_AVAILABLE
            else None
        )

        # Evaluation prompt
        self.evaluation_prompt = (
            PromptTemplate(
                template="""Evaluate the quality of this HR response.

Original Question: {question}
Generated Response: {response}
Available Context: {context}

Evaluation Criteria:
1. Accuracy: Is the response factually correct based on context?
2. Completeness: Does it address all aspects of the question?
3. Relevance: Is all information relevant to the question?
4. Clarity: Is the response clear and professional?

Provide a score from 0.0 to 1.0 and explanation.
Format: SCORE: [0.0-1.0] | EXPLANATION: [reasoning]""",
                input_variables=["question", "response", "context"],
            )
            if LANGCHAIN_AVAILABLE
            else None
        )

    def process_query(self, query: str) -> RAGResponse:
        """Process query through complete self-RAG pipeline."""
        processing_steps = []
        start_time = datetime.now()

        try:
            # Step 1: Document retrieval
            processing_steps.append("Retrieving relevant documents...")
            retrieved_docs = self.vector_manager.similarity_search(query)

            if not retrieved_docs:
                return self._create_no_documents_response(processing_steps)

            # Step 2: Response generation
            processing_steps.append("Generating response...")
            context = self._format_context(retrieved_docs)
            initial_response = self._generate_response(query, context)

            # Step 3: Self-evaluation
            confidence_score = 0.8  # Default confidence
            final_response = initial_response

            if self.config.enable_self_correction:
                processing_steps.append("Evaluating response quality...")
                confidence_score = self._evaluate_response_quality(
                    query, initial_response, context
                )

                # Step 4: Refinement if needed
                if confidence_score < self.config.relevance_threshold:
                    processing_steps.append("Refining response...")
                    final_response = self._refine_response(
                        query, initial_response, context
                    )
                    confidence_score = min(confidence_score + 0.1, 1.0)

            # Extract sources
            sources = self._extract_sources(retrieved_docs)

            # Processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_steps.append(
                f"Processing completed in {processing_time:.2f} seconds"
            )

            return RAGResponse(
                answer=final_response,
                retrieved_docs=retrieved_docs,
                confidence_score=confidence_score,
                sources=sources,
                processing_steps=processing_steps,
                workflow_metadata={
                    "processing_time": processing_time,
                    "document_count": len(retrieved_docs),
                    "self_correction_applied": confidence_score
                    < self.config.relevance_threshold,
                },
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_error_response(
                query, str(e), processing_steps, processing_time
            )

    def _create_no_documents_response(self, processing_steps: List[str]) -> RAGResponse:
        """Create response when no relevant documents found."""
        processing_steps.append("No relevant documents found")

        return RAGResponse(
            answer="I don't have specific information about that topic in our onboarding materials. Please contact HR directly for assistance.",
            retrieved_docs=[],
            confidence_score=0.0,
            sources=[],
            processing_steps=processing_steps,
            workflow_metadata={"no_documents_found": True},
        )

    def _create_error_response(
        self,
        query: str,
        error: str,
        processing_steps: List[str],
        processing_time: float,
    ) -> RAGResponse:
        """Create response for error scenarios."""
        processing_steps.append(f"Error encountered: {error}")

        return RAGResponse(
            answer="I apologize, but I encountered a technical issue. Please try again or contact HR directly.",
            retrieved_docs=[],
            confidence_score=0.0,
            sources=[],
            processing_steps=processing_steps,
            workflow_metadata={
                "error_occurred": True,
                "error_message": error,
                "processing_time": processing_time,
            },
        )

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context."""
        if not documents:
            return "No relevant context available."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown Source")
            content = doc.page_content.strip()
            context_parts.append(f"Document {i} ({source}):\n{content}\n")

        return "\n".join(context_parts)

    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM."""
        try:
            if self.generation_prompt and LANGCHAIN_AVAILABLE:
                prompt = self.generation_prompt.format(question=query, context=context)
            else:
                prompt = f"""
                As an HR assistant, provide a comprehensive answer to this question:
                
                Question: {query}
                
                Available Information:
                {context}
                
                Response:
                """

            response = self.llm.invoke(prompt)

            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating your response. Please contact HR directly."

    def _evaluate_response_quality(
        self, query: str, response: str, context: str
    ) -> float:
        """Evaluate response quality."""
        try:
            if self.evaluation_prompt and LANGCHAIN_AVAILABLE:
                eval_prompt = self.evaluation_prompt.format(
                    question=query, response=response, context=context
                )
            else:
                eval_prompt = f"""
                Rate this response quality from 0.0 to 1.0:
                
                Question: {query}
                Response: {response}
                
                Score (0.0-1.0):
                """

            evaluation = self.llm.invoke(eval_prompt)

            if hasattr(evaluation, "content"):
                eval_text = evaluation.content
            else:
                eval_text = str(evaluation)

            # Extract score
            if "SCORE:" in eval_text:
                try:
                    score_part = eval_text.split("SCORE:")[1].split("|")[0].strip()
                    score = float(score_part)
                    return max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass

            # Fallback scoring
            return self._calculate_fallback_confidence(query, response, context)

        except Exception as e:
            print(f"Error evaluating response: {e}")
            return 0.6  # Default confidence

    def _calculate_fallback_confidence(
        self, query: str, response: str, context: str
    ) -> float:
        """Calculate fallback confidence score."""
        confidence = 0.5

        # Length-based confidence
        if len(response) > 100:
            confidence += 0.1
        if len(response) > 300:
            confidence += 0.1

        # Context utilization
        if context and any(
            word in response.lower() for word in ["policy", "benefit", "company"]
        ):
            confidence += 0.2

        # Query coverage
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        if len(query_words.intersection(response_words)) / len(query_words) > 0.5:
            confidence += 0.1

        return min(confidence, 1.0)

    def _refine_response(self, query: str, initial_response: str, context: str) -> str:
        """Refine response for better quality."""
        refine_prompt = f"""
        Please improve this response to make it more comprehensive and accurate:

        Question: {query}
        Initial Response: {initial_response}
        Available Context: {context}
        
        Improved Response:"""

        try:
            refined = self.llm.invoke(refine_prompt)
            if hasattr(refined, "content"):
                return refined.content.strip()
            return str(refined).strip()
        except Exception as e:
            print(f"Error refining response: {e}")
            return initial_response

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            source_info = doc.metadata.get(
                "filename", doc.metadata.get("source", "Unknown")
            )
            sources.append(source_info)
        return list(set(sources))  # Remove duplicates


# ========================================================================================
# MAIN ONBOARDING WORKFLOW
# ========================================================================================


class OnboardingWorkflow:
    """Main orchestrator for onboarding-related queries."""

    def __init__(self, llm, embeddings, config: OnboardingConfig = None):
        self.config = config or OnboardingConfig()
        self.llm = llm
        self.initialization_time = datetime.now()

        try:
            print("Initializing onboarding knowledge base...")

            self.doc_manager = DocumentManager(self.config)
            self.vector_manager = VectorStoreManager(self.config, embeddings)

            # Load and process documents
            documents = self.doc_manager.load_documents()
            chunked_docs = self.doc_manager.chunk_documents(documents)
            self.vector_store = self.vector_manager.get_or_create_vector_store(
                chunked_docs
            )

            # Initialize RAG pipeline
            self.rag_pipeline = SelfRAGPipeline(llm, self.vector_manager, self.config)

            self.initialization_successful = True
            print("Onboarding workflow initialized successfully.")

        except Exception as e:
            print(f"Error initializing onboarding workflow: {e}")
            self.initialization_successful = False
            self.initialization_error = str(e)

    def handle_query(self, query: str) -> Dict[str, Any]:
        """Handle onboarding query and return structured response."""
        if not self.initialization_successful:
            return self._create_initialization_error_response()

        print(f"Processing onboarding query: {query[:100]}...")

        try:
            # Process through RAG pipeline
            rag_response = self.rag_pipeline.process_query(query)

            # Format response
            response = {
                "answer": rag_response.answer,
                "confidence": rag_response.confidence_score,
                "sources": rag_response.sources,
                "retrieved_documents_count": len(rag_response.retrieved_docs),
                "processing_steps": rag_response.processing_steps,
                "workflow": "onboarding",
                "timestamp": datetime.now().isoformat(),
                "workflow_metadata": rag_response.workflow_metadata,
                "retrieved_docs": rag_response.retrieved_docs,  # For DeepEval
            }

            print(
                f"Response generated with {rag_response.confidence_score:.2f} confidence"
            )
            return response

        except Exception as e:
            print(f"Error in onboarding workflow: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your onboarding question. Please contact HR directly.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_documents_count": 0,
                "processing_steps": [f"Error: {str(e)}"],
                "workflow": "onboarding_error",
                "timestamp": datetime.now().isoformat(),
                "error_details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            }

    def _create_initialization_error_response(self) -> Dict[str, Any]:
        """Create error response for initialization failures."""
        return {
            "answer": "The onboarding system is currently unavailable. Please contact HR directly.",
            "confidence": 0.0,
            "sources": [],
            "retrieved_documents_count": 0,
            "processing_steps": ["Initialization failed"],
            "workflow": "onboarding_init_error",
            "timestamp": datetime.now().isoformat(),
            "error_details": {
                "initialization_error": getattr(
                    self, "initialization_error", "Unknown error"
                )
            },
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        base_status = {
            "initialization_successful": self.initialization_successful,
            "initialization_time": self.initialization_time.isoformat(),
            "langchain_available": LANGCHAIN_AVAILABLE,
        }

        if self.initialization_successful:
            base_status.update(
                {
                    "config": {
                        "documents_path": self.config.documents_path,
                        "vector_store_path": self.config.vector_store_path,
                        "chunk_size": self.config.chunk_size,
                        "top_k_retrieval": self.config.top_k_retrieval,
                        "self_correction_enabled": self.config.enable_self_correction,
                    },
                }
            )
        else:
            base_status.update(
                {
                    "initialization_error": getattr(
                        self, "initialization_error", "Unknown error"
                    )
                }
            )

        return base_status


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================


def create_sample_onboarding_docs(docs_path: str = "onboarding_docs"):
    """Create sample onboarding documents."""
    docs_dir = Path(docs_path)
    docs_dir.mkdir(exist_ok=True)

    doc_manager = DocumentManager(OnboardingConfig(documents_path=docs_path))
    doc_manager._create_sample_documents(docs_dir)


if __name__ == "__main__":
    # Demo of the onboarding component
    print("Onboarding Component Demo")
    print("=" * 40)

    # Create sample documents
    create_sample_onboarding_docs()

    # Initialize mock components
    llm = MockLLM()

    class MockEmbeddings:
        def __init__(self):
            self.model_name = "mock-embeddings"

    embeddings = MockEmbeddings()

    # Initialize workflow
    config = OnboardingConfig()
    workflow = OnboardingWorkflow(llm, embeddings, config)

    # Test queries
    test_queries = [
        "What are the company's core values?",
        "Explain the benefits package for new employees",
        "What are the password requirements?",
        "Tell me about professional development opportunities",
        "What is the onboarding process?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = workflow.handle_query(query)
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Sources: {', '.join(response['sources'])}")

    # Display system status
    print(f"\n{'='*40}")
    print("System Status:")
    status = workflow.get_system_status()
    print(
        f"Initialization: {'Success' if status['initialization_successful'] else 'Failed'}"
    )
    print(f"LangChain Available: {status['langchain_available']}")
