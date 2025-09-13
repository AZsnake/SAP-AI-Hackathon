import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import random

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


class LeadershipCompetency(Enum):
    """Core leadership competencies for early talent development."""

    COMMUNICATION = "COMMUNICATION"
    EMOTIONAL_INTELLIGENCE = "EMOTIONAL_INTELLIGENCE"
    DECISION_MAKING = "DECISION_MAKING"
    STRATEGIC_THINKING = "STRATEGIC_THINKING"
    TEAM_BUILDING = "TEAM_BUILDING"
    ADAPTABILITY = "ADAPTABILITY"
    INFLUENCE = "INFLUENCE"
    ACCOUNTABILITY = "ACCOUNTABILITY"


class ProficiencyLevel(Enum):
    """Proficiency levels for competency assessment."""

    NOVICE = 1
    DEVELOPING = 2
    PROFICIENT = 3
    ADVANCED = 4
    EXPERT = 5


@dataclass
class LeadershipConfig:
    """Configuration for leadership development workflow component."""

    documents_path: str = "leadership_docs"
    vector_store_path: str = "vector_stores/leadership_faiss"
    assessment_store_path: str = "assessments/leadership"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    relevance_threshold: float = 0.7
    enable_self_correction: bool = True
    enable_adaptive_learning: bool = True
    assessment_frequency_days: int = 30
    minimum_confidence_threshold: float = 0.6


@dataclass
class CompetencyAssessment:
    """Individual competency assessment."""

    competency: LeadershipCompetency
    current_level: ProficiencyLevel
    target_level: ProficiencyLevel
    score: float  # 0.0 to 1.0
    last_assessed: datetime
    improvement_rate: float  # Rate of improvement over time
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class LearningPath:
    """Personalized learning path for leadership development."""

    user_id: str
    created_at: datetime
    updated_at: datetime
    competency_assessments: Dict[LeadershipCompetency, CompetencyAssessment]
    current_focus_areas: List[LeadershipCompetency]
    completed_modules: List[str]
    in_progress_modules: List[str]
    recommended_modules: List[str]
    overall_progress: float  # 0.0 to 1.0
    learning_style: str  # visual, auditory, kinesthetic, reading/writing
    pace_preference: str  # accelerated, standard, gradual
    next_milestone: Optional[str] = None
    estimated_completion: Optional[datetime] = None


@dataclass
class LeadershipRAGResponse:
    """Structured response from Leadership RAG pipeline."""

    answer: str
    retrieved_docs: List[Document]
    confidence_score: float
    sources: List[str]
    competencies_addressed: List[LeadershipCompetency]
    learning_path_update: Optional[Dict[str, Any]]
    recommended_exercises: List[Dict[str, str]]
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
                "Leadership is about inspiring others to achieve common goals through vision and influence.",
                {"source": "leadership_fundamentals.pdf", "competency": "INFLUENCE"},
            ),
            MockDocument(
                "Emotional intelligence involves self-awareness, self-regulation, and empathy in leadership.",
                {"source": "eq_guide.pdf", "competency": "EMOTIONAL_INTELLIGENCE"},
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

        if "assess" in prompt_lower or "evaluate" in prompt_lower:
            return "COMPETENCY: COMMUNICATION | LEVEL: DEVELOPING | SCORE: 0.65 | STRENGTHS: Clear articulation, active listening | IMPROVEMENTS: Executive presence, stakeholder management"
        elif "learning path" in prompt_lower or "recommend" in prompt_lower:
            return "Based on your assessment, focus on: 1) Executive Communication Workshop, 2) Stakeholder Influence Techniques, 3) Strategic Thinking Fundamentals"
        elif "exercise" in prompt_lower or "practice" in prompt_lower:
            return "EXERCISE: Prepare a 5-minute presentation on your team's quarterly goals. Focus on clarity, vision, and inspiring action."
        elif "score" in prompt_lower or "quality" in prompt_lower:
            return "SCORE: 0.75 | EXPLANATION: Response demonstrates good understanding with practical applications."
        else:
            return "Leadership development requires continuous learning and practice across multiple competencies."


# ========================================================================================
# COMPETENCY ASSESSMENT ENGINE
# ========================================================================================


class CompetencyAssessmentEngine:
    """Engine for assessing and tracking leadership competencies."""

    def __init__(self, config: LeadershipConfig):
        self.config = config
        self.assessment_store = Path(config.assessment_store_path)
        self.assessment_store.mkdir(parents=True, exist_ok=True)

    def assess_competency(
        self, user_response: str, competency: LeadershipCompetency, llm
    ) -> CompetencyAssessment:
        """Assess a specific competency based on user response."""

        assessment_prompt = f"""
        Assess the following response for {competency.value} competency:
        
        Response: {user_response}
        
        Evaluate on a scale of 1-5 (Novice to Expert):
        1. Novice: Limited understanding, requires significant guidance
        2. Developing: Basic understanding, can apply with support
        3. Proficient: Solid understanding, can apply independently
        4. Advanced: Strong understanding, can mentor others
        5. Expert: Exceptional understanding, can innovate and lead
        
        Provide assessment in format:
        COMPETENCY: {competency.value}
        LEVEL: [NOVICE/DEVELOPING/PROFICIENT/ADVANCED/EXPERT]
        SCORE: [0.0-1.0]
        STRENGTHS: [comma-separated list]
        IMPROVEMENTS: [comma-separated list]
        ACTIONS: [comma-separated recommended actions]
        """

        try:
            response = llm.invoke(assessment_prompt)
            # Fix: Convert response to string if needed
            if hasattr(response, "content"):
                response_text = str(response.content)
            else:
                response_text = str(response)
            assessment = self._parse_assessment(response_text, competency)
            return assessment
        except Exception as e:
            print(f"Error in competency assessment: {e}")
            return self._create_default_assessment(competency)

    def _parse_assessment(
        self, response: str, competency: LeadershipCompetency
    ) -> CompetencyAssessment:
        """Parse LLM assessment response."""

        lines = response.strip().split("\n")
        parsed = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.strip().upper()] = value.strip()

        # Extract level
        level_str = parsed.get("LEVEL", "DEVELOPING").upper()
        level_map = {
            "NOVICE": ProficiencyLevel.NOVICE,
            "DEVELOPING": ProficiencyLevel.DEVELOPING,
            "PROFICIENT": ProficiencyLevel.PROFICIENT,
            "ADVANCED": ProficiencyLevel.ADVANCED,
            "EXPERT": ProficiencyLevel.EXPERT,
        }
        current_level = level_map.get(level_str, ProficiencyLevel.DEVELOPING)

        # Extract score
        try:
            score = float(parsed.get("SCORE", "0.5"))
        except ValueError:
            score = 0.5

        # Extract lists
        strengths = [
            s.strip() for s in parsed.get("STRENGTHS", "").split(",") if s.strip()
        ]
        improvements = [
            i.strip() for i in parsed.get("IMPROVEMENTS", "").split(",") if i.strip()
        ]
        actions = [a.strip() for a in parsed.get("ACTIONS", "").split(",") if a.strip()]

        return CompetencyAssessment(
            competency=competency,
            current_level=current_level,
            target_level=ProficiencyLevel.ADVANCED,  # Default target for early talent
            score=score,
            last_assessed=datetime.now(),
            improvement_rate=0.0,  # Calculate based on history
            strengths=strengths,
            areas_for_improvement=improvements,
            recommended_actions=actions,
        )

    def _create_default_assessment(
        self, competency: LeadershipCompetency
    ) -> CompetencyAssessment:
        """Create default assessment when parsing fails."""

        return CompetencyAssessment(
            competency=competency,
            current_level=ProficiencyLevel.DEVELOPING,
            target_level=ProficiencyLevel.ADVANCED,
            score=0.5,
            last_assessed=datetime.now(),
            improvement_rate=0.0,
            strengths=["Showing interest in development"],
            areas_for_improvement=["Requires comprehensive assessment"],
            recommended_actions=["Complete initial assessment module"],
        )

    def calculate_improvement_rate(
        self, user_id: str, competency: LeadershipCompetency
    ) -> float:
        """Calculate improvement rate based on historical assessments."""

        history_file = self.assessment_store / f"{user_id}_history.json"

        if not history_file.exists():
            return 0.0

        try:
            with open(history_file, "r") as f:
                history = json.load(f)

            competency_history = history.get(competency.value, [])
            if len(competency_history) < 2:
                return 0.0

            # Calculate rate based on last 3 assessments
            recent = competency_history[-3:]
            if len(recent) >= 2:
                score_change = recent[-1]["score"] - recent[0]["score"]
                time_diff = (
                    datetime.fromisoformat(recent[-1]["timestamp"])
                    - datetime.fromisoformat(recent[0]["timestamp"])
                ).days

                if time_diff > 0:
                    return score_change / time_diff * 30  # Rate per month

            return 0.0

        except Exception as e:
            print(f"Error calculating improvement rate: {e}")
            return 0.0

    def save_assessment(self, user_id: str, assessment: CompetencyAssessment):
        """Save assessment to persistent storage."""

        history_file = self.assessment_store / f"{user_id}_history.json"

        # Load existing history
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
        else:
            history = {}

        # Add new assessment
        competency_key = assessment.competency.value
        if competency_key not in history:
            history[competency_key] = []

        history[competency_key].append(
            {
                "timestamp": assessment.last_assessed.isoformat(),
                "level": assessment.current_level.value,
                "score": assessment.score,
                "strengths": assessment.strengths,
                "improvements": assessment.areas_for_improvement,
            }
        )

        # Save updated history
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)


# ========================================================================================
# ADAPTIVE LEARNING PATH GENERATOR
# ========================================================================================


class AdaptiveLearningPathGenerator:
    """Generates and updates personalized learning paths."""

    def __init__(self, config: LeadershipConfig):
        self.config = config
        self.paths_store = Path("learning_paths")
        self.paths_store.mkdir(parents=True, exist_ok=True)

    def generate_initial_path(
        self,
        user_id: str,
        assessments: Dict[LeadershipCompetency, CompetencyAssessment],
        learning_style: str = "mixed",
        pace_preference: str = "standard",
    ) -> LearningPath:
        """Generate initial learning path based on assessments."""

        # Identify focus areas (lowest scoring competencies)
        sorted_competencies = sorted(assessments.items(), key=lambda x: x[1].score)

        # Focus on bottom 3 competencies initially
        focus_areas = [comp for comp, _ in sorted_competencies[:3]]

        # Generate module recommendations
        recommended_modules = self._generate_module_recommendations(
            focus_areas, learning_style, pace_preference
        )

        # Calculate overall progress
        if assessments:
            overall_progress = sum(a.score for a in assessments.values()) / len(
                assessments
            )
        else:
            overall_progress = 0

        # Estimate completion time
        estimated_completion = self._estimate_completion_time(
            assessments, pace_preference
        )

        return LearningPath(
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            competency_assessments=assessments,
            current_focus_areas=focus_areas,
            completed_modules=[],
            in_progress_modules=[],
            recommended_modules=recommended_modules,
            overall_progress=overall_progress,
            learning_style=learning_style,
            pace_preference=pace_preference,
            next_milestone=f"Complete {recommended_modules[0] if recommended_modules else 'initial assessment'}",
            estimated_completion=estimated_completion,
        )

    def _generate_module_recommendations(
        self,
        focus_areas: List[LeadershipCompetency],
        learning_style: str,
        pace_preference: str,
    ) -> List[str]:
        """Generate module recommendations based on focus areas and preferences."""

        modules = []

        for competency in focus_areas:
            # Base modules for each competency
            if competency == LeadershipCompetency.COMMUNICATION:
                modules.extend(
                    [
                        "Executive Communication Fundamentals",
                        "Active Listening Workshop",
                        "Presenting with Impact",
                    ]
                )
            elif competency == LeadershipCompetency.EMOTIONAL_INTELLIGENCE:
                modules.extend(
                    [
                        "Self-Awareness Assessment",
                        "Empathy in Leadership",
                        "Managing Emotions Under Pressure",
                    ]
                )
            elif competency == LeadershipCompetency.DECISION_MAKING:
                modules.extend(
                    [
                        "Data-Driven Decision Making",
                        "Risk Assessment Framework",
                        "Decisive Leadership in Uncertainty",
                    ]
                )
            elif competency == LeadershipCompetency.STRATEGIC_THINKING:
                modules.extend(
                    [
                        "Systems Thinking Introduction",
                        "Long-term Vision Development",
                        "Strategic Planning Essentials",
                    ]
                )
            elif competency == LeadershipCompetency.TEAM_BUILDING:
                modules.extend(
                    [
                        "Building High-Performance Teams",
                        "Delegation and Empowerment",
                        "Team Dynamics and Motivation",
                    ]
                )
            elif competency == LeadershipCompetency.ADAPTABILITY:
                modules.extend(
                    [
                        "Leading Through Change",
                        "Agile Leadership Principles",
                        "Resilience Building",
                    ]
                )
            elif competency == LeadershipCompetency.INFLUENCE:
                modules.extend(
                    [
                        "Persuasion and Influence Tactics",
                        "Building Stakeholder Relationships",
                        "Leading Without Authority",
                    ]
                )
            elif competency == LeadershipCompetency.ACCOUNTABILITY:
                modules.extend(
                    [
                        "Creating Accountability Culture",
                        "Performance Management Basics",
                        "Giving and Receiving Feedback",
                    ]
                )

        # Adjust based on learning style
        if learning_style == "visual":
            modules = [f"{m} (Visual)" for m in modules]
        elif learning_style == "kinesthetic":
            modules = [f"{m} (Interactive)" for m in modules]

        # Adjust quantity based on pace
        if pace_preference == "accelerated":
            return modules[:9]  # More modules
        elif pace_preference == "gradual":
            return modules[:3]  # Fewer modules
        else:
            return modules[:6]  # Standard pace

    def _estimate_completion_time(
        self,
        assessments: Dict[LeadershipCompetency, CompetencyAssessment],
        pace_preference: str,
    ) -> datetime:
        """Estimate completion time for reaching target proficiency."""

        # Calculate average gap to target
        total_gap = sum(
            (a.target_level.value - a.current_level.value) for a in assessments.values()
        )

        # Base estimation: 2 months per proficiency level
        months_needed = total_gap * 2

        # Adjust for pace preference
        if pace_preference == "accelerated":
            months_needed *= 0.7
        elif pace_preference == "gradual":
            months_needed *= 1.3

        return datetime.now() + timedelta(days=int(months_needed * 30))

    def update_learning_path(
        self,
        learning_path: LearningPath,
        completed_module: Optional[str] = None,
        new_assessment: Optional[CompetencyAssessment] = None,
        llm=None,
    ) -> LearningPath:
        """Update learning path based on progress and new assessments."""

        learning_path.updated_at = datetime.now()

        # Update completed modules
        if completed_module:
            if completed_module in learning_path.in_progress_modules:
                learning_path.in_progress_modules.remove(completed_module)
            if completed_module not in learning_path.completed_modules:
                learning_path.completed_modules.append(completed_module)

        # Update assessment if provided
        if new_assessment:
            learning_path.competency_assessments[new_assessment.competency] = (
                new_assessment
            )

        # Recalculate overall progress
        if learning_path.competency_assessments:
            learning_path.overall_progress = sum(
                a.score for a in learning_path.competency_assessments.values()
            ) / len(learning_path.competency_assessments)
        else:
            learning_path.overall_progress = 0.0

        # Adapt focus areas based on progress
        if self.config.enable_adaptive_learning:
            learning_path = self._adapt_focus_areas(learning_path)

        # Generate new recommendations
        learning_path.recommended_modules = self._generate_module_recommendations(
            learning_path.current_focus_areas,
            learning_path.learning_style,
            learning_path.pace_preference,
        )

        # Update next milestone
        if learning_path.recommended_modules:
            learning_path.next_milestone = (
                f"Complete {learning_path.recommended_modules[0]}"
            )

        return learning_path

    def _adapt_focus_areas(self, learning_path: LearningPath) -> LearningPath:
        """Adapt focus areas based on progress and improvement rates."""

        # Calculate improvement rates for each competency
        improvements = {}
        for comp, assessment in learning_path.competency_assessments.items():
            # Simple improvement calculation based on score
            improvements[comp] = assessment.score

        # Identify competencies that need more focus
        sorted_comps = sorted(improvements.items(), key=lambda x: x[1])

        # Update focus areas to bottom 3 performers
        learning_path.current_focus_areas = [comp for comp, _ in sorted_comps[:3]]

        # If someone is doing exceptionally well, add stretch goals
        high_performers = [comp for comp, score in sorted_comps if score > 0.8]
        if high_performers and len(learning_path.current_focus_areas) < 4:
            # Add advanced competency as stretch goal
            learning_path.current_focus_areas.append(high_performers[0])

        return learning_path

    def save_learning_path(self, learning_path: LearningPath):
        """Save learning path to persistent storage."""

        path_file = self.paths_store / f"{learning_path.user_id}_path.json"

        path_data = {
            "user_id": learning_path.user_id,
            "created_at": learning_path.created_at.isoformat(),
            "updated_at": learning_path.updated_at.isoformat(),
            "overall_progress": learning_path.overall_progress,
            "learning_style": learning_path.learning_style,
            "pace_preference": learning_path.pace_preference,
            "completed_modules": learning_path.completed_modules,
            "in_progress_modules": learning_path.in_progress_modules,
            "recommended_modules": learning_path.recommended_modules,
            "current_focus_areas": [c.value for c in learning_path.current_focus_areas],
            "next_milestone": learning_path.next_milestone,
            "estimated_completion": (
                learning_path.estimated_completion.isoformat()
                if learning_path.estimated_completion
                else None
            ),
        }

        with open(path_file, "w") as f:
            json.dump(path_data, f, indent=2)

    def load_learning_path(self, user_id: str) -> Optional[LearningPath]:
        """Load learning path from storage."""

        path_file = self.paths_store / f"{user_id}_path.json"

        if not path_file.exists():
            return None

        try:
            with open(path_file, "r") as f:
                data = json.load(f)

            # Note: This is simplified - you'd need to reconstruct full assessments
            return LearningPath(
                user_id=data["user_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                competency_assessments={},  # Would need to load separately
                current_focus_areas=[
                    LeadershipCompetency[c] for c in data["current_focus_areas"]
                ],
                completed_modules=data["completed_modules"],
                in_progress_modules=data["in_progress_modules"],
                recommended_modules=data["recommended_modules"],
                overall_progress=data["overall_progress"],
                learning_style=data["learning_style"],
                pace_preference=data["pace_preference"],
                next_milestone=data["next_milestone"],
                estimated_completion=(
                    datetime.fromisoformat(data["estimated_completion"])
                    if data["estimated_completion"]
                    else None
                ),
            )
        except Exception as e:
            print(f"Error loading learning path: {e}")
            return None


# ========================================================================================
# VECTOR STORE MANAGEMENT
# ========================================================================================


class VectorStoreManager:
    """Vector store management for document retrieval."""

    def __init__(self, config: LeadershipConfig, embeddings):
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
# LEADERSHIP DOCUMENT MANAGEMENT
# ========================================================================================


class LeadershipDocumentManager:
    """Document manager for leadership development content."""

    def __init__(self, config: LeadershipConfig):
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
        """Load leadership development documents."""
        documents = []
        docs_path = Path(self.config.documents_path)

        if not docs_path.exists():
            print(
                f"Documents directory not found. Creating sample leadership documents..."
            )
            self._create_sample_documents(docs_path)

        supported_extensions = {".txt", ".md", ".json"}

        try:
            for file_path in docs_path.rglob("*"):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if content.strip():
                            # Extract competency from filename or content
                            competency = self._extract_competency(
                                file_path.name, content
                            )

                            doc = (
                                Document(
                                    page_content=content,
                                    metadata={
                                        "source": str(file_path),
                                        "filename": file_path.name,
                                        "competency": competency,
                                        "doc_type": "leadership_content",
                                    },
                                )
                                if LANGCHAIN_AVAILABLE
                                else MockDocument(
                                    content,
                                    {
                                        "source": str(file_path),
                                        "filename": file_path.name,
                                        "competency": competency,
                                        "doc_type": "leadership_content",
                                    },
                                )
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        except Exception as e:
            print(f"Error in document loading: {e}")
            return self._get_sample_documents()

        print(f"Loaded {len(documents)} documents for leadership knowledge base.")
        return documents

    def _extract_competency(self, filename: str, content: str) -> str:
        """Extract competency from filename or content."""

        filename_lower = filename.lower()
        content_lower = content.lower()

        for competency in LeadershipCompetency:
            comp_name = competency.value.lower().replace("_", " ")
            if comp_name in filename_lower or comp_name in content_lower[:500]:
                return competency.value

        return "GENERAL"

    def _create_sample_documents(self, docs_path: Path):
        """Create comprehensive sample leadership documents."""

        docs_path.mkdir(parents=True, exist_ok=True)

        sample_content = {
            "communication_excellence.md": """
# Communication Excellence for Emerging Leaders

## Overview
Effective communication is the cornerstone of successful leadership. As an early-career professional transitioning into leadership roles, mastering communication skills will significantly impact your ability to inspire, influence, and guide teams.

## Key Components

### 1. Active Listening
- **Practice full attention**: Put away devices, maintain eye contact
- **Reflect and paraphrase**: "What I'm hearing is..."
- **Ask clarifying questions**: Ensure complete understanding
- **Acknowledge emotions**: Recognize feelings behind words

### 2. Clear Articulation
- **Structure your thoughts**: Use frameworks like PREP (Point, Reason, Example, Point)
- **Adapt to your audience**: Technical vs. non-technical, senior vs. peer level
- **Use concrete examples**: Abstract concepts need tangible illustrations
- **Eliminate filler words**: Practice pausing instead of "um" or "uh"

### 3. Executive Presence
- **Command the room**: Stand tall, project confidence
- **Vocal variety**: Vary pace, tone, and volume for emphasis
- **Strategic silence**: Use pauses for impact
- **Body language alignment**: Ensure non-verbals support your message

### 4. Written Communication
- **Email excellence**: Clear subject lines, bullet points, action items
- **Documentation**: Comprehensive yet concise project updates
- **Presentation decks**: Visual storytelling with data
- **Professional messaging**: Slack/Teams etiquette

## Practical Exercises

### Exercise 1: The Elevator Pitch
Prepare a 30-second summary of your current project that could be delivered to a senior executive in an elevator. Focus on impact and value.

### Exercise 2: Difficult Conversations Role-Play
Practice delivering constructive feedback to a peer. Structure: Situation, Behavior, Impact, Next Steps.

### Exercise 3: Stakeholder Communication Matrix
Create a matrix identifying key stakeholders and tailoring communication style for each.

## Common Pitfalls to Avoid
1. Information overload - Know when enough detail is enough
2. Assuming understanding - Always verify comprehension
3. Emotional reactions - Maintain professional composure
4. One-size-fits-all approach - Customize for your audience

## Reflection Questions
- How do others perceive my communication style?
- What triggers cause me to communicate less effectively?
- Which communication channel is most appropriate for this message?
- How can I ensure my message drives action?
            """,
            "emotional_intelligence_guide.md": """
# Emotional Intelligence in Leadership

## Understanding EQ
Emotional Intelligence (EQ) is your ability to recognize, understand, and manage both your own emotions and those of others. For emerging leaders, EQ often matters more than technical expertise.

## The Four Domains of EQ

### 1. Self-Awareness
**Definition**: Understanding your emotions, strengths, weaknesses, values, and impact on others.

**Development Strategies**:
- Daily reflection journaling
- Seek 360-degree feedback
- Identify emotional triggers
- Practice mindfulness meditation

**Questions for Reflection**:
- What emotions am I experiencing right now?
- How do my emotions influence my decisions?
- What are my emotional patterns under stress?

### 2. Self-Management
**Definition**: Controlling disruptive emotions and adapting to change.

**Key Competencies**:
- Emotional self-control
- Adaptability
- Achievement orientation
- Positive outlook

**Techniques**:
- The 6-second pause before reacting
- Reframing negative situations
- Stress management techniques
- Building resilience practices

### 3. Social Awareness
**Definition**: Understanding others' emotions and organizational dynamics.

**Skills to Develop**:
- Empathy and perspective-taking
- Organizational awareness
- Service orientation
- Reading non-verbal cues

**Practice Methods**:
- Active listening without judgment
- Observing team dynamics
- Cultural sensitivity training
- Customer/stakeholder interviews

### 4. Relationship Management
**Definition**: Influencing, coaching, and mentoring others; resolving conflict.

**Core Abilities**:
- Influence and persuasion
- Conflict management
- Team leadership
- Inspirational leadership

## Practical Applications

### Scenario 1: Team Member Frustration
Your team member seems frustrated during a meeting. How do you:
1. Recognize the emotion (Social Awareness)
2. Manage your response (Self-Management)
3. Address the situation (Relationship Management)

### Scenario 2: Project Setback
A critical project fails. Apply EQ to:
1. Process your disappointment (Self-Awareness)
2. Maintain team morale (Relationship Management)
3. Learn from failure (Self-Management)

## EQ Development Plan
1. **Week 1-2**: Self-awareness assessment and journaling
2. **Week 3-4**: Practice emotional regulation techniques
3. **Week 5-6**: Empathy exercises with team members
4. **Week 7-8**: Apply EQ in challenging conversations

## Measuring Your Progress
- Regular self-assessments
- Peer feedback on interpersonal skills
- Track successful conflict resolutions
- Monitor team engagement scores
            """,
            "strategic_thinking_fundamentals.md": """
# Strategic Thinking for Emerging Leaders

## What is Strategic Thinking?
Strategic thinking is the ability to see the big picture, anticipate future trends, and make decisions that align with long-term objectives while managing short-term demands.

## Core Components

### 1. Systems Thinking
**Understanding Interconnections**:
- Map relationships between departments, processes, and outcomes
- Identify ripple effects of decisions
- Recognize feedback loops and unintended consequences

**Tools**:
- Stakeholder mapping
- Process flow diagrams
- Cause-and-effect analysis
- SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)

### 2. Future-Oriented Perspective
**Anticipating Change**:
- Industry trend analysis
- Scenario planning
- Risk assessment
- Innovation opportunities

**Questions to Ask**:
- What will our industry look like in 5 years?
- What skills will become obsolete or critical?
- How might customer needs evolve?
- What disruptions could impact our business?

### 3. Data-Driven Decision Making
**Analytical Framework**:
- Gather relevant metrics
- Identify patterns and correlations
- Challenge assumptions with data
- Measure outcome effectiveness

**Key Metrics for Leaders**:
- Team productivity indicators
- Customer satisfaction scores
- Process efficiency metrics
- Innovation/improvement rates

### 4. Balancing Competing Priorities
**Strategic Trade-offs**:
- Short-term vs. long-term gains
- Quality vs. speed
- Innovation vs. stability
- Individual vs. team needs

## Strategic Thinking Exercises

### Exercise 1: The Pre-Mortem
Before launching a project, imagine it has failed. Work backwards to identify potential failure points and preventive measures.

### Exercise 2: Competitive Analysis
Study three competitors or parallel industries. What strategies are they employing? What can you learn or adapt?

### Exercise 3: Vision Development
Create a 3-year vision for your team/department. Define:
- Desired future state
- Key milestones
- Required capabilities
- Success metrics

## Frameworks for Strategic Analysis

### Porter's Five Forces
1. Competitive rivalry
2. Supplier power
3. Buyer power
4. Threat of substitution
5. Threat of new entry

### BCG Growth-Share Matrix
- Stars (High growth, High share)
- Cash Cows (Low growth, High share)
- Question Marks (High growth, Low share)
- Dogs (Low growth, Low share)

### OKRs (Objectives and Key Results)
- Set ambitious objectives
- Define measurable key results
- Align team efforts
- Track progress regularly

## Common Strategic Thinking Pitfalls
1. Analysis paralysis - Overthinking without action
2. Short-term focus - Missing long-term implications
3. Silo thinking - Ignoring cross-functional impacts
4. Confirmation bias - Seeking only supporting data

## Developing Your Strategic Mindset
1. Read widely outside your field
2. Question the status quo regularly
3. Network across industries
4. Practice systems thinking daily
5. Seek diverse perspectives
            """,
            "team_building_leadership.md": """
# Building and Leading High-Performance Teams

## Foundation of Team Leadership
Leading a team as an early-career professional requires balancing authority with collaboration, driving results while developing people.

## Stages of Team Development

### 1. Forming
**Your Role**: Set clear expectations and create psychological safety
- Define team charter and goals
- Establish communication norms
- Facilitate introductions and relationship building
- Clarify roles and responsibilities

### 2. Storming
**Your Role**: Navigate conflict constructively
- Address disagreements directly
- Encourage healthy debate
- Mediate personality clashes
- Reinforce team values

### 3. Norming
**Your Role**: Build cohesion and trust
- Celebrate early wins
- Establish team rituals
- Encourage peer support
- Delegate appropriately

### 4. Performing
**Your Role**: Optimize performance and growth
- Provide stretch opportunities
- Remove obstacles
- Champion team achievements
- Plan for succession

## Key Team Leadership Competencies

### Creating Psychological Safety
**What It Means**: Team members feel safe to take risks, make mistakes, and speak up

**How to Build It**:
- Admit your own mistakes
- Ask for feedback regularly
- Show curiosity, not judgment
- Celebrate learning from failures

### Effective Delegation
**The Delegation Framework**:
1. **Define**: Clear outcomes and success criteria
2. **Discuss**: Ensure understanding and buy-in
3. **Determine**: Resources and support needed
4. **Deliver**: Set checkpoints and deadlines
5. **Debrief**: Review results and lessons learned

**What to Delegate**:
- Tasks that develop team skills
- Recurring operational activities
- Projects with clear parameters
- Areas where team members have expertise

### Motivation and Engagement
**Understanding Individual Drivers**:
- Autonomy - Desire for self-direction
- Mastery - Need to improve skills
- Purpose - Connection to meaningful work
- Recognition - Appreciation for contributions
- Growth - Career advancement opportunities

**Motivation Strategies**:
- Personalized recognition approaches
- Skill development opportunities
- Meaningful project assignments
- Regular career conversations
- Team celebration rituals

### Performance Management
**Continuous Performance Approach**:
- Weekly 1:1 check-ins
- Monthly goal reviews
- Quarterly development discussions
- Annual comprehensive reviews

**Giving Effective Feedback**:
- SBI Model: Situation, Behavior, Impact
- Balance affirmative and constructive
- Make it timely and specific
- Focus on behaviors, not personality
- Collaborate on solutions

## Team Dynamics Optimization

### Diversity and Inclusion
- Value different perspectives
- Ensure equitable participation
- Address unconscious bias
- Create inclusive team norms
- Leverage diverse strengths

### Remote/Hybrid Team Leadership
- Over-communicate expectations
- Use asynchronous collaboration tools
- Schedule regular virtual coffee chats
- Ensure meeting equity
- Build virtual team culture

### Conflict Resolution
**The RESOLVE Framework**:
- **R**ecognize the conflict early
- **E**ngage parties involved
- **S**eek to understand all perspectives
- **O**utline common ground
- **L**ook for win-win solutions
- **V**erify agreement and commitment
- **E**valuate and follow up

## Team Building Activities

### Activity 1: Strengths Mapping
Have team members identify and share their top strengths. Create a team strengths matrix for better collaboration.

### Activity 2: Problem-Solving Challenge
Present a real business challenge. Break into sub-teams to develop solutions. Present and combine best ideas.

### Activity 3: Peer Coaching Circles
Establish monthly peer coaching sessions where team members help each other with challenges.

## Measuring Team Effectiveness
- Team health surveys
- Project delivery metrics
- Individual growth tracking
- Stakeholder feedback
- Innovation indicators

## Red Flags to Watch
1. Decreased communication
2. Missed deadlines becoming normal
3. Lack of healthy conflict
4. Individual focus over team goals
5. High turnover or disengagement
            """,
            "decision_making_framework.md": """
# Decision-Making Excellence for Leaders

## The Leadership Decision Challenge
As an emerging leader, your decisions impact not just your work but your team's success and organizational outcomes. Developing strong decision-making capabilities is crucial.

## Decision-Making Frameworks

### 1. The DECIDE Model
- **D**efine the problem clearly
- **E**stablish criteria for solutions
- **C**onsider alternatives
- **I**dentify best alternatives
- **D**evelop and implement action plan
- **E**valuate and monitor solution

### 2. The Cynefin Framework
**Simple/Obvious**: Best practices exist
- Sense → Categorize → Respond

**Complicated**: Good practices exist
- Sense → Analyze → Respond

**Complex**: Emergent practices needed
- Probe → Sense → Respond

**Chaotic**: Novel practices required
- Act → Sense → Respond

### 3. OODA Loop (Observe-Orient-Decide-Act)
- **Observe**: Gather information
- **Orient**: Analyze and synthesize
- **Decide**: Determine course of action
- **Act**: Implement decision

## Types of Decisions

### Strategic vs. Tactical
**Strategic Decisions**:
- Long-term impact
- Difficult to reverse
- Require extensive analysis
- Example: Hiring key team members

**Tactical Decisions**:
- Short-term impact
- Easily adjustable
- Quick turnaround needed
- Example: Weekly task prioritization

### Individual vs. Collaborative
**When to Decide Alone**:
- Emergency situations
- Confidential matters
- Clear expertise advantage
- Low-stakes decisions

**When to Involve Others**:
- High impact on team
- Need diverse perspectives
- Building buy-in critical
- Development opportunity

## Decision-Making Tools

### Decision Matrix
Create weighted criteria for evaluating options:
1. List all options
2. Define evaluation criteria
3. Weight criteria by importance
4. Score each option
5. Calculate weighted totals

### Cost-Benefit Analysis
- Quantify costs (time, money, resources)
- Estimate benefits (revenue, efficiency, morale)
- Consider opportunity costs
- Factor in risk probability

### SWOT Analysis
- **Strengths**: Internal advantages
- **Weaknesses**: Internal limitations
- **Opportunities**: External possibilities
- **Threats**: External challenges

### Pre-Mortem Analysis
Imagine decision has failed:
- What went wrong?
- What was overlooked?
- What assumptions were false?
- How can we prevent failure?

## Cognitive Biases to Avoid

### Confirmation Bias
Seeking information that confirms existing beliefs
**Mitigation**: Actively seek disconfirming evidence

### Anchoring Bias
Over-relying on first piece of information
**Mitigation**: Consider multiple reference points

### Sunk Cost Fallacy
Continuing due to past investment
**Mitigation**: Focus on future value, not past costs

### Groupthink
Conformity pressure in group decisions
**Mitigation**: Encourage devil's advocate role

### Availability Heuristic
Overweighting recent/memorable events
**Mitigation**: Use data, not just memory

## Decision-Making Under Pressure

### Time-Constrained Decisions
1. Set decision deadline
2. Gather minimum viable information
3. Use 80/20 rule
4. Make reversible decisions quickly
5. Document assumptions for review

### High-Stakes Decisions
1. Expand consultation circle
2. Scenario planning
3. Risk mitigation strategies
4. Clear success metrics
5. Contingency plans

### Uncertain Environment
1. Gather diverse perspectives
2. Use probabilistic thinking
3. Build in flexibility
4. Set review checkpoints
5. Prepare multiple scenarios

## Practical Exercises

### Exercise 1: Daily Decision Journal
Track your decisions for a week:
- Decision made
- Framework used
- Outcome achieved
- Lessons learned

### Exercise 2: Bias Audit
Review recent decisions for biases:
- What assumptions did you make?
- What information did you ignore?
- Would you decide differently now?

### Exercise 3: Scenario Planning
For your next big decision:
- Best case scenario
- Worst case scenario
- Most likely scenario
- Contingency for each

## After the Decision

### Implementation Excellence
- Clear communication of decision
- Assign ownership and timelines
- Resource allocation
- Progress monitoring
- Adjustment readiness

### Learning from Outcomes
- Was the decision effective?
- What information would have helped?
- How was the process?
- What will you do differently?

## Building Decision-Making Confidence
1. Start with low-stakes decisions
2. Use frameworks consistently
3. Track decision outcomes
4. Learn from both successes and failures
5. Seek feedback on decision process
6. Study others' decision-making
7. Practice under time pressure
            """,
            "influence_without_authority.md": """
# Leading Through Influence

## The Reality of Modern Leadership
In today's organizations, you'll often need to lead projects and initiatives without formal authority over team members. Influence becomes your primary leadership tool.

## Building Your Influence Foundation

### 1. Credibility
**Technical Competence**: Demonstrate expertise in your domain
**Reliability**: Consistently deliver on commitments
**Integrity**: Align actions with values
**Results**: Track record of success

### 2. Relationships
**Trust Building**: Invest time before you need help
**Reciprocity**: Give before you receive
**Genuine Interest**: Care about others' success
**Network Strategically**: Build broad and deep connections

### 3. Communication
**Listen First**: Understand before seeking to be understood
**Speak Their Language**: Adapt to audience preferences
**Tell Stories**: Make data memorable with narratives
**Be Concise**: Respect others' time

## Influence Tactics

### Rational Persuasion
Use logic, data, and reasoning
**When to Use**: Analytical audiences, data-driven cultures
**Example**: "Based on last quarter's data, this approach will increase efficiency by 23%"

### Inspirational Appeals
Connect to values and emotions
**When to Use**: Vision-setting, change initiatives
**Example**: "This project aligns with our mission to transform customer experience"

### Consultation
Involve others in planning
**When to Use**: Need buy-in, collaborative cultures
**Example**: "I'd love your input on how we approach this challenge"

### Coalition Building
Gain support from multiple stakeholders
**When to Use**: Large initiatives, political environments
**Example**: "Marketing and Sales are already on board with this approach"

### Personal Appeals
Leverage relationships and friendship
**When to Use**: Existing strong relationships
**Example**: "I could really use your expertise on this"

### Exchange
Offer something in return
**When to Use**: Transactional relationships
**Example**: "If you help with this, I can support your project next month"

## Stakeholder Mapping and Management

### Influence-Interest Grid
- **High Influence, High Interest**: Manage closely
- **High Influence, Low Interest**: Keep satisfied
- **Low Influence, High Interest**: Keep informed
- **Low Influence, Low Interest**: Monitor

### Understanding Stakeholder Motivations
- What are their goals?
- What are their pain points?
- What does success look like for them?
- What are their concerns?
- How do they prefer to communicate?

## Influence in Specific Situations

### Influencing Upward (Managing Up)
- Understand your manager's priorities
- Present solutions, not just problems
- Use executive summaries
- Time your requests strategically
- Build business cases

### Influencing Peers
- Find mutual benefits
- Build alliance networks
- Share credit generously
- Offer expertise and support
- Create win-win scenarios

### Influencing Across Functions
- Learn their language and priorities
- Understand their metrics
- Respect their expertise
- Find common ground
- Build bridges, not walls

## Building Long-term Influence

### Become a Connector
- Introduce people who should know each other
- Share resources and opportunities
- Build communities of practice
- Facilitate collaboration

### Develop Expertise
- Become the go-to person for specific knowledge
- Share insights generously
- Write and present on your expertise
- Mentor others in your area

### Create Value
- Solve problems others avoid
- Volunteer for challenging projects
- Improve processes
- Generate innovative ideas
- Deliver consistent results

## Common Influence Mistakes

1. **Pushing Too Hard**: Aggressive tactics create resistance
2. **Ignoring Politics**: Organizational dynamics matter
3. **One-Size-Fits-All**: Same approach for everyone
4. **Transactional Only**: Focusing only on immediate needs
5. **Neglecting Relationships**: Reaching out only when you need something

## Influence Action Plan

### Week 1-2: Assessment
- Map your current influence network
- Identify influence gaps
- Assess your influence style

### Week 3-4: Relationship Building
- Schedule coffee chats with key stakeholders
- Join cross-functional projects
- Attend networking events

### Week 5-6: Skill Development
- Practice different influence tactics
- Seek feedback on your approach
- Study effective influencers

### Week 7-8: Application
- Lead a cross-functional initiative
- Practice influencing without authority
- Track what works

## Measuring Your Influence
- Frequency of being consulted
- Invitations to key meetings
- Success rate of proposals
- Network growth
- Project participation requests
            """,
            "adaptive_leadership.md": """
# Adaptive Leadership in Changing Environments

## Understanding Adaptive Leadership
Adaptive leadership is about navigating change, uncertainty, and complex challenges that don't have clear solutions. It's essential for emerging leaders in today's dynamic workplace.

## Core Principles

### 1. Diagnose the System
**Observe patterns**: What behaviors and dynamics do you see?
**Identify adaptations needed**: What must change?
**Distinguish technical from adaptive**: What's a simple fix vs. requiring new learning?

### 2. Mobilize the System
**Create urgency**: Help others see the need for change
**Regulate distress**: Maintain productive discomfort
**Engage stakeholders**: Involve those affected by change

### 3. See the System
**Get on the balcony**: Step back to see the big picture
**Observe yourself**: Understand your role in the system
**Identify hidden dynamics**: What's not being said?

### 4. Intervene Skillfully
**Make interpretations**: Offer observations, not judgments
**Design effective interventions**: Small experiments, not big changes
**Hold steady**: Maintain presence during turbulence

## Types of Challenges

### Technical Challenges
- Clear problem definition
- Known solutions exist
- Expert knowledge applies
- Can be solved by authority

**Example**: Implementing new software system

### Adaptive Challenges
- Problem definition unclear
- Solutions require learning
- Multiple stakeholders involved
- Requires attitude/behavior change

**Example**: Changing team culture

### Mixed Challenges
- Contains both elements
- Requires dual approach
- Most common in reality

**Example**: Digital transformation

## Leading Through Change

### The Change Curve
1. **Denial**: "This won't affect us"
2. **Resistance**: "This won't work"
3. **Exploration**: "How might this work?"
4. **Commitment**: "Let's make this work"

### Your Role at Each Stage
**Denial**: Create awareness, share data
**Resistance**: Listen to concerns, acknowledge loss
**Exploration**: Facilitate learning, celebrate experiments
**Commitment**: Reinforce benefits, sustain momentum

## Building Resilience

### Personal Resilience
- **Physical**: Exercise, sleep, nutrition
- **Mental**: Mindfulness, learning mindset
- **Emotional**: Support network, self-compassion
- **Spiritual**: Purpose, values alignment

### Team Resilience
- **Psychological safety**: OK to fail and learn
- **Collective efficacy**: Belief in team capability
- **Shared purpose**: Clear why behind change
- **Support systems**: Peer support and resources

### Organizational Resilience
- **Redundancy**: Backup systems and plans
- **Diversity**: Multiple perspectives and approaches
- **Adaptability**: Rapid response capability
- **Learning culture**: Continuous improvement mindset

## Adaptive Leadership Tools

### Experimentation Mindset
- Run small pilots
- Learn fast, fail cheap
- Scale what works
- Kill what doesn't

### Scenario Planning
- Best case scenarios
- Worst case scenarios
- Most likely scenarios
- Black swan events

### After Action Reviews
- What was supposed to happen?
- What actually happened?
- Why were there differences?
- What can we learn?

## Managing Resistance

### Understanding Resistance
**Sources**:
- Loss of control
- Excess uncertainty
- Surprise elements
- Difference overload
- Loss of face
- Concerns about competence
- Ripple effects

### Strategies for Addressing Resistance
1. **Involve early and often**
2. **Communicate the why**
3. **Acknowledge losses**
4. **Provide support and training**
5. **Celebrate small wins**
6. **Model the change**
7. **Be patient but persistent**

## Innovation and Adaptation

### Creating Innovation Space
- Dedicated time for experimentation
- Resources for trying new things
- Permission to fail
- Recognition for attempts

### Fostering Innovative Thinking
- Question assumptions
- Encourage wild ideas
- Build on others' ideas
- Defer judgment
- Go for quantity

## Practical Applications

### Exercise 1: Change Readiness Assessment
Evaluate your team's readiness:
- Past change experiences
- Current capacity
- Risk tolerance
- Support systems

### Exercise 2: Adaptive Challenge Mapping
Identify a current challenge:
- What's technical vs. adaptive?
- Who needs to learn what?
- What experiments could you run?

### Exercise 3: Resilience Building Plan
Create a 30-day plan:
- Personal resilience practices
- Team resilience activities
- System redundancies

## Leading in Crisis

### Crisis Leadership Framework
1. **Assess**: Understand the situation
2. **Act**: Take decisive action
3. **Assure**: Provide confidence
4. **Adapt**: Adjust as needed
5. **Anchor**: Connect to values

### Communication in Crisis
- Frequent updates
- Transparent about unknowns
- Clear about decisions
- Empathetic to concerns
- Consistent messaging

## Developing Adaptive Capacity

### For Yourself
- Seek diverse experiences
- Practice comfort with ambiguity
- Build learning agility
- Develop multiple perspectives
- Strengthen emotional regulation

### For Your Team
- Rotate roles and responsibilities
- Cross-training initiatives
- Diverse team composition
- Regular reflection sessions
- Celebration of learning

## Measuring Adaptive Success
- Speed of response to change
- Quality of innovations
- Team engagement during change
- Stakeholder satisfaction
- Learning captured and applied
            """,
        }

        for filename, content in sample_content.items():
            file_path = docs_path / filename
            file_path.write_text(content.strip())

        print(f"Created {len(sample_content)} sample leadership documents.")

    def _get_sample_documents(self) -> List[Document]:
        """Generate sample leadership documents for fallback."""

        sample_docs = [
            {
                "content": """LEADERSHIP COMMUNICATION FUNDAMENTALS
                
                Effective leadership communication involves:
                1. Active listening - Give full attention, reflect, and clarify
                2. Clear articulation - Structure thoughts, adapt to audience
                3. Executive presence - Command attention through confidence
                4. Written excellence - Clear, concise, action-oriented messages
                
                Practice: Prepare elevator pitches, conduct difficult conversations, 
                create stakeholder matrices for targeted communication.""",
                "metadata": {
                    "source": "communication.md",
                    "competency": "COMMUNICATION",
                },
            },
            {
                "content": """EMOTIONAL INTELLIGENCE IN LEADERSHIP
                
                Four domains of EQ:
                1. Self-awareness - Understand your emotions and impact
                2. Self-management - Control emotions and adapt to change  
                3. Social awareness - Read others and organizational dynamics
                4. Relationship management - Influence and resolve conflicts
                
                Development: Daily reflection, feedback seeking, empathy practice.""",
                "metadata": {
                    "source": "emotional_intelligence.md",
                    "competency": "EMOTIONAL_INTELLIGENCE",
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
        """Chunk documents for better retrieval."""

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
                            "competency": doc.metadata.get("competency", "GENERAL"),
                        }
                    )
                chunked_docs.extend(chunks)
            except Exception as e:
                print(f"Error chunking document: {e}")
                chunked_docs.append(doc)

        print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")
        return chunked_docs


# ========================================================================================
# LEADERSHIP SELF-RAG PIPELINE
# ========================================================================================


class LeadershipSelfRAGPipeline:
    """Self-Reflective RAG pipeline for leadership development."""

    def __init__(
        self,
        llm,
        vector_manager,
        assessment_engine: CompetencyAssessmentEngine,
        path_generator: AdaptiveLearningPathGenerator,
        config: LeadershipConfig,
    ):
        self.llm = llm
        self.vector_manager = vector_manager
        self.assessment_engine = assessment_engine
        self.path_generator = path_generator
        self.config = config

        # Generation prompt for leadership guidance
        self.generation_prompt = (
            PromptTemplate(
                template="""You are an expert leadership development coach helping early-career professionals develop leadership skills.

Context Information:
{context}

User Query: {question}

Current Focus Areas: {focus_areas}

Instructions:
- Provide actionable leadership guidance based on the context
- Include specific techniques and frameworks
- Offer practical exercises when relevant
- Connect advice to the user's current development focus areas
- Be encouraging while maintaining high standards
- Format response clearly with examples

Leadership Development Response:""",
                input_variables=["context", "question", "focus_areas"],
            )
            if LANGCHAIN_AVAILABLE
            else None
        )

        # Evaluation prompt
        self.evaluation_prompt = (
            PromptTemplate(
                template="""Evaluate this leadership development response quality.

Original Question: {question}
Generated Response: {response}
Available Context: {context}

Evaluation Criteria:
1. Actionability: Does it provide specific, actionable advice?
2. Relevance: Does it address the leadership competencies involved?
3. Accuracy: Is the guidance based on established leadership principles?
4. Practicality: Can an early-career professional implement this?
5. Completeness: Does it fully address the question?

Provide a score from 0.0 to 1.0 and explanation.
Format: SCORE: [0.0-1.0] | EXPLANATION: [reasoning]""",
                input_variables=["question", "response", "context"],
            )
            if LANGCHAIN_AVAILABLE
            else None
        )

    def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        learning_path: Optional[LearningPath] = None,
    ) -> LeadershipRAGResponse:
        """Process leadership development query through RAG pipeline."""

        processing_steps = []
        start_time = datetime.now()

        try:
            # Step 1: Identify relevant competencies
            processing_steps.append("Identifying relevant leadership competencies...")
            competencies = self._identify_competencies(query)

            # Step 2: Retrieve relevant documents
            processing_steps.append("Retrieving leadership development content...")
            retrieved_docs = self._retrieve_with_competency_focus(query, competencies)

            if not retrieved_docs:
                return self._create_no_documents_response(processing_steps)

            # Step 3: Generate response with learning path context
            processing_steps.append("Generating personalized guidance...")
            context = self._format_context(retrieved_docs)
            focus_areas = self._get_focus_areas(learning_path)

            initial_response = self._generate_response(query, context, focus_areas)

            # Step 4: Self-evaluation and refinement
            confidence_score = 0.8
            final_response = initial_response

            if self.config.enable_self_correction:
                processing_steps.append("Evaluating response quality...")
                confidence_score = self._evaluate_response_quality(
                    query, initial_response, context
                )

                if confidence_score < self.config.relevance_threshold:
                    processing_steps.append(
                        "Refining response for better actionability..."
                    )
                    final_response = self._refine_response(
                        query, initial_response, context, focus_areas
                    )
                    confidence_score = min(confidence_score + 0.15, 1.0)

            # Step 5: Generate exercises and recommendations
            processing_steps.append("Creating practice exercises...")
            exercises = self._generate_exercises(query, competencies)

            # Step 6: Update learning path if applicable
            learning_path_update = None
            if learning_path and user_id:
                processing_steps.append("Updating learning path...")
                learning_path_update = self._create_learning_path_update(
                    competencies, confidence_score
                )

            # Extract sources and metadata
            sources = self._extract_sources(retrieved_docs)
            processing_time = (datetime.now() - start_time).total_seconds()

            processing_steps.append(
                f"Processing completed in {processing_time:.2f} seconds"
            )

            return LeadershipRAGResponse(
                answer=final_response,
                retrieved_docs=retrieved_docs,
                confidence_score=confidence_score,
                sources=sources,
                competencies_addressed=competencies,
                learning_path_update=learning_path_update,
                recommended_exercises=exercises,
                processing_steps=processing_steps,
                workflow_metadata={
                    "processing_time": processing_time,
                    "document_count": len(retrieved_docs),
                    "competency_count": len(competencies),
                    "self_correction_applied": confidence_score
                    < self.config.relevance_threshold,
                },
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_error_response(
                query, str(e), processing_steps, processing_time
            )

    def _identify_competencies(self, query: str) -> List[LeadershipCompetency]:
        """Identify relevant leadership competencies from query."""

        query_lower = query.lower()
        identified = []

        competency_keywords = {
            LeadershipCompetency.COMMUNICATION: [
                "communication",
                "presentation",
                "speaking",
                "writing",
                "listening",
            ],
            LeadershipCompetency.EMOTIONAL_INTELLIGENCE: [
                "emotional",
                "eq",
                "empathy",
                "self-awareness",
                "feelings",
            ],
            LeadershipCompetency.DECISION_MAKING: [
                "decision",
                "decide",
                "choice",
                "problem-solving",
                "analysis",
            ],
            LeadershipCompetency.STRATEGIC_THINKING: [
                "strategic",
                "strategy",
                "planning",
                "vision",
                "long-term",
            ],
            LeadershipCompetency.TEAM_BUILDING: [
                "team",
                "collaboration",
                "group",
                "delegate",
                "motivate",
            ],
            LeadershipCompetency.ADAPTABILITY: [
                "change",
                "adapt",
                "flexible",
                "agile",
                "resilience",
            ],
            LeadershipCompetency.INFLUENCE: [
                "influence",
                "persuade",
                "convince",
                "stakeholder",
                "negotiate",
            ],
            LeadershipCompetency.ACCOUNTABILITY: [
                "accountability",
                "responsibility",
                "ownership",
                "feedback",
                "performance",
            ],
        }

        for competency, keywords in competency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                identified.append(competency)

        # Default to general leadership if no specific competency identified
        if not identified:
            identified = [LeadershipCompetency.COMMUNICATION]  # Default competency

        return identified

    def _retrieve_with_competency_focus(
        self, query: str, competencies: List[LeadershipCompetency]
    ) -> List[Document]:
        """Retrieve documents with focus on identified competencies."""

        # Standard retrieval
        docs = self.vector_manager.similarity_search(query)

        # Filter or boost based on competencies if possible
        if docs and competencies:
            # Prioritize docs matching identified competencies
            prioritized = []
            others = []

            for doc in docs:
                doc_competency = doc.metadata.get("competency", "GENERAL")
                if any(comp.value == doc_competency for comp in competencies):
                    prioritized.append(doc)
                else:
                    others.append(doc)

            return prioritized + others

        return docs

    def _get_focus_areas(self, learning_path: Optional[LearningPath]) -> str:
        """Get current focus areas from learning path."""

        if not learning_path:
            return "General leadership development"

        focus_names = [
            comp.value.replace("_", " ").title()
            for comp in learning_path.current_focus_areas
        ]

        return (
            ", ".join(focus_names) if focus_names else "General leadership development"
        )

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context."""

        if not documents:
            return "No relevant context available."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown Source")
            competency = doc.metadata.get("competency", "General")
            content = doc.page_content.strip()

            context_parts.append(
                f"Document {i} ({source} - {competency}):\n{content}\n"
            )

        return "\n".join(context_parts)

    def _generate_response(self, query: str, context: str, focus_areas: str) -> str:
        """Generate leadership development response."""

        try:
            if self.generation_prompt and LANGCHAIN_AVAILABLE:
                prompt = self.generation_prompt.format(
                    question=query, context=context, focus_areas=focus_areas
                )
            else:
                prompt = f"""
                As a leadership development coach, provide actionable guidance:
                
                Question: {query}
                Focus Areas: {focus_areas}
                Available Information: {context}
                
                Response:
                """

            response = self.llm.invoke(prompt)

            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating your leadership development guidance."

    def _evaluate_response_quality(
        self, query: str, response: str, context: str
    ) -> float:
        """Evaluate leadership response quality."""

        try:
            if self.evaluation_prompt and LANGCHAIN_AVAILABLE:
                eval_prompt = self.evaluation_prompt.format(
                    question=query, response=response, context=context
                )
            else:
                eval_prompt = f"""
                Rate this leadership advice quality (0.0-1.0):
                Question: {query}
                Response: {response}
                Score:
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

            return 0.7  # Default confidence

        except Exception as e:
            print(f"Error evaluating response: {e}")
            return 0.6

    def _refine_response(
        self, query: str, initial_response: str, context: str, focus_areas: str
    ) -> str:
        """Refine response for better actionability."""

        refine_prompt = f"""
        Improve this leadership development response to be more actionable and practical:
        
        Question: {query}
        Initial Response: {initial_response}
        Focus Areas: {focus_areas}
        Context: {context}
        
        Provide specific steps, techniques, and examples.
        Improved Response:"""

        try:
            refined = self.llm.invoke(refine_prompt)
            if hasattr(refined, "content"):
                return refined.content.strip()
            return str(refined).strip()
        except Exception as e:
            print(f"Error refining response: {e}")
            return initial_response

    def _generate_exercises(
        self, query: str, competencies: List[LeadershipCompetency]
    ) -> List[Dict[str, str]]:
        """Generate practical exercises for skill development."""

        exercises = []

        for competency in competencies[:2]:  # Limit to 2 exercises
            if competency == LeadershipCompetency.COMMUNICATION:
                exercises.append(
                    {
                        "title": "Executive Summary Challenge",
                        "description": "Write a one-page executive summary of your current project, focusing on impact and key decisions needed.",
                        "competency": competency.value,
                        "duration": "30 minutes",
                        "difficulty": "intermediate",
                    }
                )
            elif competency == LeadershipCompetency.EMOTIONAL_INTELLIGENCE:
                exercises.append(
                    {
                        "title": "Emotion Mapping Exercise",
                        "description": "Track your emotional responses throughout the day and identify triggers and patterns.",
                        "competency": competency.value,
                        "duration": "1 day",
                        "difficulty": "beginner",
                    }
                )
            elif competency == LeadershipCompetency.DECISION_MAKING:
                exercises.append(
                    {
                        "title": "Decision Journal",
                        "description": "Document 3 decisions this week using the DECIDE framework, then review outcomes.",
                        "competency": competency.value,
                        "duration": "1 week",
                        "difficulty": "intermediate",
                    }
                )
            elif competency == LeadershipCompetency.TEAM_BUILDING:
                exercises.append(
                    {
                        "title": "Team Strengths Mapping",
                        "description": "Create a matrix of your team members' strengths and plan how to leverage them.",
                        "competency": competency.value,
                        "duration": "2 hours",
                        "difficulty": "intermediate",
                    }
                )
            else:
                exercises.append(
                    {
                        "title": f'{competency.value.replace("_", " ").title()} Practice',
                        "description": f"Apply concepts from today's learning in a real work situation.",
                        "competency": competency.value,
                        "duration": "1 week",
                        "difficulty": "intermediate",
                    }
                )

        return exercises

    def _create_learning_path_update(
        self, competencies: List[LeadershipCompetency], confidence: float
    ) -> Dict[str, Any]:
        """Create update information for learning path."""

        return {
            "competencies_practiced": [c.value for c in competencies],
            "confidence_score": confidence,
            "timestamp": datetime.now().isoformat(),
            "recommendation": (
                "Continue with current focus"
                if confidence > 0.7
                else "Review fundamentals"
            ),
        }

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """Extract source information from documents."""

        sources = []
        for doc in documents:
            source_info = doc.metadata.get(
                "filename", doc.metadata.get("source", "Unknown")
            )
            sources.append(source_info)
        return list(set(sources))

    def _create_no_documents_response(
        self, processing_steps: List[str]
    ) -> LeadershipRAGResponse:
        """Create response when no documents found."""

        processing_steps.append("No relevant leadership content found")

        return LeadershipRAGResponse(
            answer="I don't have specific leadership development content for that topic. Please try rephrasing your question or contact your HR team for additional resources.",
            retrieved_docs=[],
            confidence_score=0.0,
            sources=[],
            competencies_addressed=[],
            learning_path_update=None,
            recommended_exercises=[],
            processing_steps=processing_steps,
            workflow_metadata={"no_documents_found": True},
        )

    def _create_error_response(
        self,
        query: str,
        error: str,
        processing_steps: List[str],
        processing_time: float,
    ) -> LeadershipRAGResponse:
        """Create error response."""

        processing_steps.append(f"Error encountered: {error}")

        return LeadershipRAGResponse(
            answer="I apologize, but I encountered a technical issue processing your leadership development query. Please try again or contact support.",
            retrieved_docs=[],
            confidence_score=0.0,
            sources=[],
            competencies_addressed=[],
            learning_path_update=None,
            recommended_exercises=[],
            processing_steps=processing_steps,
            workflow_metadata={
                "error_occurred": True,
                "error_message": error,
                "processing_time": processing_time,
            },
        )


# ========================================================================================
# MAIN LEADERSHIP DEVELOPMENT WORKFLOW
# ========================================================================================


class LeadershipDevelopmentWorkflow:
    """Main orchestrator for leadership development queries and learning paths."""

    def __init__(self, llm, embeddings, config: LeadershipConfig = None):
        self.config = config or LeadershipConfig()
        self.llm = llm
        self.embeddings = embeddings
        self.initialization_time = datetime.now()

        try:
            print("Initializing leadership development system...")

            # Initialize components
            self.doc_manager = LeadershipDocumentManager(self.config)
            self.vector_manager = VectorStoreManager(self.config, embeddings)
            self.assessment_engine = CompetencyAssessmentEngine(self.config)
            self.path_generator = AdaptiveLearningPathGenerator(self.config)

            # Load and process documents
            documents = self.doc_manager.load_documents()
            chunked_docs = self.doc_manager.chunk_documents(documents)
            self.vector_store = self.vector_manager.get_or_create_vector_store(
                chunked_docs
            )

            # Initialize RAG pipeline
            self.rag_pipeline = LeadershipSelfRAGPipeline(
                llm,
                self.vector_manager,
                self.assessment_engine,
                self.path_generator,
                self.config,
            )

            self.initialization_successful = True
            print("Leadership development workflow initialized successfully.")

        except Exception as e:
            print(f"Error initializing leadership workflow: {e}")
            self.initialization_successful = False
            self.initialization_error = str(e)

    def handle_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle leadership development query with optional user context."""

        if not self.initialization_successful:
            return self._create_initialization_error_response()

        print(f"Processing leadership query: {query[:100]}...")

        try:
            # Load user's learning path if available
            learning_path = None
            if user_id:
                learning_path = self.path_generator.load_learning_path(user_id)

            # Process through RAG pipeline
            rag_response = self.rag_pipeline.process_query(
                query, user_id, learning_path
            )

            # Format response for main system
            response = {
                "answer": rag_response.answer,
                "confidence": rag_response.confidence_score,
                "sources": rag_response.sources,
                "retrieved_documents_count": len(rag_response.retrieved_docs),
                "competencies_addressed": [
                    c.value for c in rag_response.competencies_addressed
                ],
                "recommended_exercises": rag_response.recommended_exercises,
                "processing_steps": rag_response.processing_steps,
                "workflow": "leadership_development",
                "timestamp": datetime.now().isoformat(),
                "workflow_metadata": rag_response.workflow_metadata,
                "retrieved_docs": rag_response.retrieved_docs,  # For DeepEval
            }

            # Add learning path info if available
            if learning_path:
                response["learning_path_info"] = {
                    "overall_progress": learning_path.overall_progress,
                    "current_focus": [
                        c.value for c in learning_path.current_focus_areas
                    ],
                    "next_milestone": learning_path.next_milestone,
                }

            if rag_response.learning_path_update and user_id:
                # Update and save learning path
                if learning_path:
                    updated_path = self.path_generator.update_learning_path(
                        learning_path, new_assessment=None
                    )
                    self.path_generator.save_learning_path(updated_path)

            print(
                f"Response generated with {rag_response.confidence_score:.2f} confidence"
            )
            return response

        except Exception as e:
            print(f"Error in leadership workflow: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your leadership development question. Please try again or contact HR.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_documents_count": 0,
                "processing_steps": [f"Error: {str(e)}"],
                "workflow": "leadership_error",
                "timestamp": datetime.now().isoformat(),
                "error_details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            }

    def conduct_assessment(
        self, user_id: str, responses: Dict[LeadershipCompetency, str]
    ) -> LearningPath:
        """Conduct comprehensive leadership assessment and generate learning path."""

        assessments = {}

        for competency, response in responses.items():
            assessment = self.assessment_engine.assess_competency(
                response, competency, self.llm
            )
            assessments[competency] = assessment
            self.assessment_engine.save_assessment(user_id, assessment)

        # Generate or update learning path
        existing_path = self.path_generator.load_learning_path(user_id)

        if existing_path:
            # Update existing path with new assessments
            for comp, assessment in assessments.items():
                existing_path.competency_assessments[comp] = assessment
            learning_path = self.path_generator.update_learning_path(
                existing_path, llm=self.llm
            )
        else:
            # Create new learning path
            learning_path = self.path_generator.generate_initial_path(
                user_id, assessments
            )

        self.path_generator.save_learning_path(learning_path)
        return learning_path

    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get user's leadership development progress."""

        learning_path = self.path_generator.load_learning_path(user_id)

        if not learning_path:
            return {
                "status": "not_started",
                "message": "No learning path found. Complete initial assessment to begin.",
            }

        progress_data = {
            "overall_progress": learning_path.overall_progress,
            "current_focus_areas": [c.value for c in learning_path.current_focus_areas],
            "completed_modules": learning_path.completed_modules,
            "in_progress_modules": learning_path.in_progress_modules,
            "recommended_modules": learning_path.recommended_modules[:3],
            "next_milestone": learning_path.next_milestone,
            "learning_style": learning_path.learning_style,
            "pace_preference": learning_path.pace_preference,
            "estimated_completion": (
                learning_path.estimated_completion.isoformat()
                if learning_path.estimated_completion
                else None
            ),
        }

        # Add competency scores if available
        if (
            hasattr(learning_path, "competency_assessments")
            and learning_path.competency_assessments
        ):
            progress_data["competency_scores"] = {
                comp.value: assessment.score
                for comp, assessment in learning_path.competency_assessments.items()
            }

        return progress_data

    def _create_initialization_error_response(self) -> Dict[str, Any]:
        """Create error response for initialization failures."""

        return {
            "answer": "The leadership development system is currently unavailable. Please contact HR for assistance.",
            "confidence": 0.0,
            "sources": [],
            "retrieved_documents_count": 0,
            "processing_steps": ["Initialization failed"],
            "workflow": "leadership_init_error",
            "timestamp": datetime.now().isoformat(),
            "error_details": {
                "initialization_error": getattr(
                    self, "initialization_error", "Unknown error"
                )
            },
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration."""

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
                        "assessment_store_path": self.config.assessment_store_path,
                        "chunk_size": self.config.chunk_size,
                        "top_k_retrieval": self.config.top_k_retrieval,
                        "adaptive_learning_enabled": self.config.enable_adaptive_learning,
                        "self_correction_enabled": self.config.enable_self_correction,
                        "assessment_frequency_days": self.config.assessment_frequency_days,
                    }
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


def create_sample_leadership_docs(docs_path: str = "leadership_docs"):
    """Create sample leadership development documents."""

    docs_dir = Path(docs_path)
    docs_dir.mkdir(exist_ok=True)

    doc_manager = LeadershipDocumentManager(LeadershipConfig(documents_path=docs_path))
    doc_manager._create_sample_documents(docs_dir)


def run_sample_assessment(llm) -> Dict[LeadershipCompetency, str]:
    """Generate sample assessment responses for testing."""

    return {
        LeadershipCompetency.COMMUNICATION: "I focus on clear articulation and active listening in team meetings.",
        LeadershipCompetency.EMOTIONAL_INTELLIGENCE: "I try to understand team members' perspectives and manage my reactions.",
        LeadershipCompetency.DECISION_MAKING: "I gather data and consult stakeholders before making important decisions.",
        LeadershipCompetency.STRATEGIC_THINKING: "I work to understand how my projects fit into larger organizational goals.",
        LeadershipCompetency.TEAM_BUILDING: "I delegate tasks based on team members' strengths and provide support.",
    }


# ========================================================================================
# MAIN EXECUTION (FOR TESTING)
# ========================================================================================


if __name__ == "__main__":
    print("Leadership Development Component Demo")
    print("=" * 60)

    # Create sample documents
    create_sample_leadership_docs()

    # Initialize mock components
    llm = MockLLM()

    class MockEmbeddings:
        def __init__(self):
            self.model_name = "mock-embeddings"

    embeddings = MockEmbeddings()

    # Initialize workflow
    config = LeadershipConfig()
    workflow = LeadershipDevelopmentWorkflow(llm, embeddings, config)

    # Test queries
    test_queries = [
        "How can I improve my communication skills as a new team lead?",
        "What are effective strategies for building emotional intelligence?",
        "How do I make better decisions under pressure?",
        "Tips for developing strategic thinking capabilities?",
        "How to build trust with my team members?",
    ]

    print("\nTesting Leadership Development Queries:")
    print("-" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = workflow.handle_query(query, user_id="test_user_001")
        print(f"Answer: {response['answer'][:300]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Competencies: {', '.join(response.get('competencies_addressed', []))}")

        if response.get("recommended_exercises"):
            print(f"Exercises: {response['recommended_exercises'][0]['title']}")

    # Test assessment and learning path
    print("\n" + "=" * 60)
    print("Testing Assessment and Learning Path Generation:")
    print("-" * 60)

    sample_responses = run_sample_assessment(llm)
    learning_path = workflow.conduct_assessment("test_user_001", sample_responses)

    print(f"Overall Progress: {learning_path.overall_progress:.2%}")
    print(
        f"Focus Areas: {', '.join([c.value for c in learning_path.current_focus_areas])}"
    )
    print(f"Recommended Modules: {', '.join(learning_path.recommended_modules[:3])}")
    print(f"Next Milestone: {learning_path.next_milestone}")

    # Display system status
    print("\n" + "=" * 60)
    print("System Status:")
    status = workflow.get_system_status()
    print(
        f"Initialization: {'Success' if status['initialization_successful'] else 'Failed'}"
    )
    print(f"LangChain Available: {status['langchain_available']}")
    print(
        f"Adaptive Learning: {status.get('config', {}).get('adaptive_learning_enabled', False)}"
    )
