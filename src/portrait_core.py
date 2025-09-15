import os
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import statistics
from collections import defaultdict

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
    from onboarding_component import OnboardingWorkflow, OnboardingConfig
    from re_upskilling_component import (
        LeadershipDevelopmentWorkflow,
        LeadershipConfig,
        LeadershipCompetency,
        ProficiencyLevel,
        CompetencyAssessment,
    )

    COMPONENTS_AVAILABLE = True
except ImportError:
    print("Component modules not available - using standalone mode")
    COMPONENTS_AVAILABLE = False

# ========================================================================================
# CONFIGURATION AND DATA STRUCTURES
# ========================================================================================


class EmployeeType(Enum):
    """Employee classification types."""
    
    EARLY_TALENT = "EARLY_TALENT"  # 0-1 years
    DEVELOPING_PROFESSIONAL = "DEVELOPING_PROFESSIONAL"  # 1-3 years
    EXPERIENCED_PROFESSIONAL = "EXPERIENCED_PROFESSIONAL"  # 3+ years
    HIGH_POTENTIAL = "HIGH_POTENTIAL"  # Identified high performers
    AT_RISK = "AT_RISK"  # Performance or retention risk


class SkillCategory(Enum):
    """Skill categorization for SAP context."""
    
    # Technical Skills
    SAP_FUNCTIONAL = "SAP_FUNCTIONAL"  # SAP module expertise
    SAP_TECHNICAL = "SAP_TECHNICAL"   # ABAP, Development
    DATA_ANALYTICS = "DATA_ANALYTICS"  # Analytics, BI
    CLOUD_TECHNOLOGIES = "CLOUD_TECHNOLOGIES"  # Cloud, Integration
    
    # Business Skills
    BUSINESS_PROCESS = "BUSINESS_PROCESS"  # Process understanding
    PROJECT_MANAGEMENT = "PROJECT_MANAGEMENT"  # PM methodologies
    CONSULTING_SKILLS = "CONSULTING_SKILLS"  # Client interaction
    INDUSTRY_KNOWLEDGE = "INDUSTRY_KNOWLEDGE"  # Domain expertise
    
    # Leadership & Soft Skills
    COMMUNICATION = "COMMUNICATION"
    LEADERSHIP = "LEADERSHIP"
    COLLABORATION = "COLLABORATION"
    PROBLEM_SOLVING = "PROBLEM_SOLVING"
    ADAPTABILITY = "ADAPTABILITY"


class PerformanceIndicator(Enum):
    """Performance measurement indicators."""
    
    EXCEEDS_EXPECTATIONS = "EXCEEDS_EXPECTATIONS"
    MEETS_EXPECTATIONS = "MEETS_EXPECTATIONS"
    APPROACHING_EXPECTATIONS = "APPROACHING_EXPECTATIONS"
    BELOW_EXPECTATIONS = "BELOW_EXPECTATIONS"
    NEW_HIRE = "NEW_HIRE"


class CareerTrack(Enum):
    """Career progression tracks within SAP context."""
    
    TECHNICAL_EXPERT = "TECHNICAL_EXPERT"
    FUNCTIONAL_CONSULTANT = "FUNCTIONAL_CONSULTANT"
    TECHNICAL_CONSULTANT = "TECHNICAL_CONSULTANT"
    PROJECT_MANAGER = "PROJECT_MANAGER"
    TEAM_LEADER = "TEAM_LEADER"
    SOLUTION_ARCHITECT = "SOLUTION_ARCHITECT"
    BUSINESS_ANALYST = "BUSINESS_ANALYST"


@dataclass
class PortraitConfig:
    """Configuration for employee portrait system."""
    
    data_path: str = "employee_data"
    vector_store_path: str = "vector_stores/portrait_faiss"
    portrait_store_path: str = "portraits"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    confidence_threshold: float = 0.7
    enable_predictive_analytics: bool = True
    portrait_refresh_days: int = 30
    risk_threshold: float = 0.3
    high_potential_threshold: float = 0.8


@dataclass
class SkillAssessment:
    """Individual skill assessment."""
    
    skill: SkillCategory
    current_level: float  # 0.0 to 1.0
    target_level: float  # 0.0 to 1.0
    last_assessed: datetime
    assessment_source: str  # "self", "manager", "peer", "system"
    confidence: float  # Confidence in assessment accuracy
    evidence: List[str] = field(default_factory=list)
    growth_trajectory: float = 0.0  # Rate of improvement
    certifications: List[str] = field(default_factory=list)


@dataclass
class PerformanceData:
    """Performance tracking data."""
    
    period: str  # "Q1 2024", "H1 2024", etc.
    overall_rating: PerformanceIndicator
    goals_achieved: int
    goals_total: int
    manager_feedback: str
    peer_ratings: List[float] = field(default_factory=list)
    project_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    learning_hours: float = 0.0
    innovation_contributions: int = 0
    client_satisfaction: Optional[float] = None


@dataclass
class CareerProgression:
    """Career development tracking."""
    
    current_role: str
    current_level: str
    career_track: CareerTrack
    tenure_months: int
    promotion_readiness: float  # 0.0 to 1.0
    next_role_target: Optional[str] = None
    development_areas: List[SkillCategory] = field(default_factory=list)
    mentor_assigned: bool = False
    succession_planning: bool = False
    mobility_preferences: List[str] = field(default_factory=list)


@dataclass
class RiskIndicators:
    """Employee risk assessment indicators."""
    
    flight_risk_score: float  # 0.0 to 1.0 (higher = more risk)
    performance_risk_score: float  # 0.0 to 1.0
    engagement_score: float  # 0.0 to 1.0
    last_survey_date: Optional[datetime] = None
    red_flags: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    recommended_interventions: List[str] = field(default_factory=list)


@dataclass
class EmployeePortrait:
    """Comprehensive employee portrait."""
    
    employee_id: str
    name: str
    email: str
    hire_date: datetime
    current_role: str
    department: str
    manager_id: Optional[str] = None
    
    # Classification
    employee_type: EmployeeType = EmployeeType.EARLY_TALENT
    career_progression: Optional[CareerProgression] = None
    
    # Assessments and Performance
    skill_assessments: Dict[SkillCategory, SkillAssessment] = field(default_factory=dict)
    performance_history: List[PerformanceData] = field(default_factory=list)
    leadership_assessments: Dict[str, Any] = field(default_factory=dict)
    
    # Analytics and Insights
    overall_potential_score: float = 0.5
    growth_velocity: float = 0.0
    risk_indicators: Optional[RiskIndicators] = None
    
    # Personalization
    learning_style: str = "mixed"
    preferred_communication: str = "email"
    career_aspirations: List[str] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    portrait_version: str = "1.0"
    data_sources: List[str] = field(default_factory=list)
    insights_generated: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PortraitInsight:
    """AI-generated insight about employee."""
    
    insight_type: str  # "strength", "risk", "opportunity", "recommendation"
    confidence: float
    title: str
    description: str
    supporting_evidence: List[str]
    recommended_actions: List[str]
    priority: str  # "high", "medium", "low"
    timeline: str  # "immediate", "short_term", "long_term"
    stakeholders: List[str]  # Who should be involved
    created_at: datetime = field(default_factory=datetime.now)


# ========================================================================================
# MOCK IMPLEMENTATIONS
# ========================================================================================


class MockDocument:
    """Mock document class for fallback."""
    
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockVectorStore:
    """Mock vector store for fallback."""
    
    def __init__(self, documents: List):
        self.documents = documents
    
    def similarity_search(self, query: str, k: int = 5) -> List[MockDocument]:
        return self.documents[:k]
    
    def save_local(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_local(cls, path: str, embeddings):
        mock_docs = [
            MockDocument("Employee performance data and skill assessments", 
                        {"source": "hr_system", "type": "performance"}),
            MockDocument("Leadership development progress and competency growth",
                        {"source": "learning_system", "type": "development"}),
        ]
        return cls(mock_docs)


class MockLLM:
    """Mock LLM for fallback."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        
        if "risk" in prompt_lower:
            return "RISK_SCORE: 0.2 | FACTORS: High engagement, strong performance, good manager relationship"
        elif "potential" in prompt_lower:
            return "POTENTIAL_SCORE: 0.75 | REASONING: Strong technical skills, leadership potential, high growth rate"
        elif "insight" in prompt_lower:
            return "INSIGHT: Employee shows strong technical capabilities with opportunities for leadership development"
        elif "recommendation" in prompt_lower:
            return "RECOMMEND: Leadership training program, mentorship assignment, stretch project assignment"
        else:
            return "Employee portrait analysis shows positive development trajectory with areas for growth"


# ========================================================================================
# DATA COLLECTION AND INTEGRATION
# ========================================================================================


class EmployeeDataCollector:
    """Collects employee data from various sources."""
    
    def __init__(self, config: PortraitConfig):
        self.config = config
        self.data_sources = {
            "hr_system": self._collect_hr_data,
            "learning_system": self._collect_learning_data,
            "performance_system": self._collect_performance_data,
            "survey_system": self._collect_survey_data,
            "project_system": self._collect_project_data,
        }
    
    def collect_employee_data(self, employee_id: str) -> Dict[str, Any]:
        """Collect comprehensive employee data from all sources."""
        
        collected_data = {
            "employee_id": employee_id,
            "collection_timestamp": datetime.now().isoformat(),
            "data_sources": [],
        }
        
        for source_name, collector_func in self.data_sources.items():
            try:
                source_data = collector_func(employee_id)
                collected_data[source_name] = source_data
                collected_data["data_sources"].append(source_name)
            except Exception as e:
                print(f"Error collecting from {source_name}: {e}")
                collected_data[source_name] = {"error": str(e)}
        
        return collected_data
    
    def _collect_hr_data(self, employee_id: str) -> Dict[str, Any]:
        """Collect basic HR information."""
        
        # In real implementation, this would connect to HR systems
        return {
            "personal_info": {
                "employee_id": employee_id,
                "name": f"Employee_{employee_id}",
                "email": f"employee_{employee_id}@sap.com",
                "hire_date": "2023-01-15",
                "current_role": "Associate Consultant",
                "department": "SAP Consulting",
                "manager_id": "MGR_001",
                "location": "Walldorf, Germany",
            },
            "employment_history": [
                {
                    "role": "Associate Consultant",
                    "start_date": "2023-01-15",
                    "department": "SAP Consulting",
                    "promotion_date": None,
                }
            ],
        }
    
    def _collect_learning_data(self, employee_id: str) -> Dict[str, Any]:
        """Collect learning and development data."""
        
        return {
            "completed_courses": [
                {"course": "SAP S/4HANA Fundamentals", "completion_date": "2023-03-15", "score": 85},
                {"course": "Leadership Essentials", "completion_date": "2023-06-20", "score": 78},
                {"course": "Agile Project Management", "completion_date": "2023-09-10", "score": 92},
            ],
            "in_progress_courses": [
                {"course": "Advanced ABAP Programming", "progress": 65, "expected_completion": "2024-02-01"},
            ],
            "certifications": [
                {"certification": "SAP Certified Application Associate", "date": "2023-08-15"},
            ],
            "learning_hours_ytd": 120,
            "learning_budget_used": 2500,
            "learning_budget_total": 3000,
        }
    
    def _collect_performance_data(self, employee_id: str) -> Dict[str, Any]:
        """Collect performance review data."""
        
        return {
            "current_rating": "MEETS_EXPECTATIONS",
            "performance_history": [
                {
                    "period": "H1 2023",
                    "overall_rating": "MEETS_EXPECTATIONS",
                    "goals_achieved": 7,
                    "goals_total": 8,
                    "manager_feedback": "Strong technical performer with good client interaction skills.",
                    "development_areas": ["Leadership", "Strategic Thinking"],
                }
            ],
            "360_feedback": {
                "manager_rating": 4.2,
                "peer_ratings": [4.0, 4.1, 3.9, 4.3],
                "direct_report_ratings": [],
                "client_ratings": [4.5, 4.2],
            },
        }
    
    def _collect_survey_data(self, employee_id: str) -> Dict[str, Any]:
        """Collect survey and engagement data."""
        
        return {
            "engagement_score": 78,
            "satisfaction_score": 82,
            "retention_indicators": {
                "intent_to_stay": 85,
                "career_satisfaction": 75,
                "manager_relationship": 88,
                "workload_balance": 72,
            },
            "survey_date": "2023-10-15",
        }
    
    def _collect_project_data(self, employee_id: str) -> Dict[str, Any]:
        """Collect project participation and outcomes."""
        
        return {
            "current_projects": [
                {
                    "project": "S/4HANA Migration - Manufacturing Client",
                    "role": "Junior Consultant",
                    "utilization": 80,
                    "client_satisfaction": 4.3,
                    "start_date": "2023-09-01",
                }
            ],
            "completed_projects": [
                {
                    "project": "SuccessFactors Implementation",
                    "role": "Associate Consultant",
                    "outcome": "Successful",
                    "client_satisfaction": 4.1,
                    "duration_months": 6,
                }
            ],
            "project_performance_avg": 4.2,
            "billable_hours_ytd": 1580,
            "target_billable_hours": 1600,
        }


# ========================================================================================
# SKILL ASSESSMENT ENGINE
# ========================================================================================


class SkillAssessmentEngine:
    """Advanced skill assessment and gap analysis."""
    
    def __init__(self, config: PortraitConfig):
        self.config = config
        self.skill_frameworks = self._initialize_skill_frameworks()
    
    def _initialize_skill_frameworks(self) -> Dict[SkillCategory, Dict]:
        """Initialize skill assessment frameworks for SAP context."""
        
        return {
            SkillCategory.SAP_FUNCTIONAL: {
                "levels": {
                    "beginner": {"threshold": 0.2, "description": "Basic module understanding"},
                    "intermediate": {"threshold": 0.5, "description": "Can configure and customize"},
                    "advanced": {"threshold": 0.7, "description": "Expert configuration and integration"},
                    "expert": {"threshold": 0.9, "description": "Thought leader and innovator"},
                },
                "assessment_criteria": [
                    "Module configuration knowledge",
                    "Business process understanding", 
                    "Integration capabilities",
                    "Troubleshooting skills",
                ],
            },
            SkillCategory.LEADERSHIP: {
                "levels": {
                    "emerging": {"threshold": 0.3, "description": "Shows leadership potential"},
                    "developing": {"threshold": 0.5, "description": "Leads small teams/projects"},
                    "competent": {"threshold": 0.7, "description": "Effective team leadership"},
                    "advanced": {"threshold": 0.9, "description": "Strategic leadership"},
                },
                "assessment_criteria": [
                    "Team management",
                    "Decision making",
                    "Communication",
                    "Strategic thinking",
                ],
            },
            # Add more skill categories as needed
        }
    
    def assess_skill_portfolio(self, employee_data: Dict[str, Any], llm) -> Dict[SkillCategory, SkillAssessment]:
        """Assess complete skill portfolio for an employee."""
        
        assessments = {}
        
        for skill_category in SkillCategory:
            assessment = self._assess_individual_skill(skill_category, employee_data, llm)
            assessments[skill_category] = assessment
        
        return assessments
    
    def _assess_individual_skill(self, skill: SkillCategory, employee_data: Dict[str, Any], llm) -> SkillAssessment:
        """Assess individual skill based on available data."""
        
        # Extract relevant data for skill assessment
        skill_evidence = self._extract_skill_evidence(skill, employee_data)
        
        # Generate assessment using LLM
        assessment_prompt = f"""
        Assess {skill.value} skill level based on the following evidence:
        
        Employee Data: {json.dumps(skill_evidence, indent=2)}
        
        Provide assessment in format:
        SKILL: {skill.value}
        CURRENT_LEVEL: [0.0-1.0]
        TARGET_LEVEL: [0.0-1.0] 
        CONFIDENCE: [0.0-1.0]
        EVIDENCE: [comma-separated evidence points]
        GROWTH_TRAJECTORY: [0.0-1.0 representing rate of improvement]
        CERTIFICATIONS: [relevant certifications]
        """
        
        try:
            response = llm.invoke(assessment_prompt)
            return self._parse_skill_assessment(skill, response, employee_data)
        except Exception as e:
            print(f"Error assessing skill {skill.value}: {e}")
            return self._create_default_skill_assessment(skill)
    
    def _extract_skill_evidence(self, skill: SkillCategory, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant evidence for specific skill assessment."""
        
        evidence = {
            "skill_category": skill.value,
            "relevant_courses": [],
            "project_experience": [],
            "performance_indicators": [],
            "certifications": [],
        }
        
        # Extract learning data
        learning_data = employee_data.get("learning_system", {})
        for course in learning_data.get("completed_courses", []):
            if self._is_course_relevant(skill, course["course"]):
                evidence["relevant_courses"].append(course)
        
        # Extract project data  
        project_data = employee_data.get("project_system", {})
        for project in project_data.get("completed_projects", []):
            if self._is_project_relevant(skill, project):
                evidence["project_experience"].append(project)
        
        # Extract performance data
        performance_data = employee_data.get("performance_system", {})
        evidence["performance_indicators"] = performance_data.get("360_feedback", {})
        
        # Extract certifications
        evidence["certifications"] = learning_data.get("certifications", [])
        
        return evidence
    
    def _is_course_relevant(self, skill: SkillCategory, course_name: str) -> bool:
        """Check if course is relevant to skill category."""
        
        course_lower = course_name.lower()
        
        skill_keywords = {
            SkillCategory.SAP_FUNCTIONAL: ["sap", "s/4hana", "module", "functional"],
            SkillCategory.SAP_TECHNICAL: ["abap", "development", "technical", "programming"],
            SkillCategory.LEADERSHIP: ["leadership", "management", "team", "communication"],
            SkillCategory.PROJECT_MANAGEMENT: ["project", "agile", "scrum", "management"],
            # Add more mappings
        }
        
        keywords = skill_keywords.get(skill, [])
        return any(keyword in course_lower for keyword in keywords)
    
    def _is_project_relevant(self, skill: SkillCategory, project: Dict[str, Any]) -> bool:
        """Check if project experience is relevant to skill."""
        
        role = project.get("role", "").lower()
        project_name = project.get("project", "").lower()
        
        if skill == SkillCategory.LEADERSHIP:
            return "lead" in role or "manager" in role
        elif skill == SkillCategory.SAP_FUNCTIONAL:
            return "sap" in project_name or "functional" in role
        elif skill == SkillCategory.PROJECT_MANAGEMENT:
            return "project" in role or "pm" in role
        
        return False
    
    def _parse_skill_assessment(self, skill: SkillCategory, response: str, employee_data: Dict) -> SkillAssessment:
        """Parse LLM response into SkillAssessment object."""
        
        lines = response.strip().split("\n")
        parsed = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed[key.strip().upper()] = value.strip()
        
        # Extract values with defaults
        current_level = float(parsed.get("CURRENT_LEVEL", "0.3"))
        target_level = float(parsed.get("TARGET_LEVEL", "0.7"))
        confidence = float(parsed.get("CONFIDENCE", "0.6"))
        growth_trajectory = float(parsed.get("GROWTH_TRAJECTORY", "0.1"))
        
        evidence = [e.strip() for e in parsed.get("EVIDENCE", "").split(",") if e.strip()]
        certifications = [c.strip() for c in parsed.get("CERTIFICATIONS", "").split(",") if c.strip()]
        
        return SkillAssessment(
            skill=skill,
            current_level=current_level,
            target_level=target_level,
            last_assessed=datetime.now(),
            assessment_source="ai_system",
            confidence=confidence,
            evidence=evidence,
            growth_trajectory=growth_trajectory,
            certifications=certifications,
        )
    
    def _create_default_skill_assessment(self, skill: SkillCategory) -> SkillAssessment:
        """Create default skill assessment when parsing fails."""
        
        return SkillAssessment(
            skill=skill,
            current_level=0.3,
            target_level=0.7,
            last_assessed=datetime.now(),
            assessment_source="default_system",
            confidence=0.5,
            evidence=["Requires assessment"],
            growth_trajectory=0.1,
            certifications=[],
        )
    
    def identify_skill_gaps(self, assessments: Dict[SkillCategory, SkillAssessment]) -> List[Dict[str, Any]]:
        """Identify critical skill gaps and development priorities."""
        
        gaps = []
        
        for skill, assessment in assessments.items():
            gap_size = assessment.target_level - assessment.current_level
            
            if gap_size > 0.2:  # Significant gap
                priority = "high" if gap_size > 0.4 else "medium"
                
                gaps.append({
                    "skill": skill.value,
                    "current_level": assessment.current_level,
                    "target_level": assessment.target_level,
                    "gap_size": gap_size,
                    "priority": priority,
                    "growth_trajectory": assessment.growth_trajectory,
                    "estimated_months_to_target": gap_size / max(assessment.growth_trajectory, 0.01),
                })
        
        # Sort by priority and gap size
        gaps.sort(key=lambda x: (x["priority"] == "high", x["gap_size"]), reverse=True)
        
        return gaps


# ========================================================================================
# PREDICTIVE ANALYTICS ENGINE
# ========================================================================================


class PredictiveAnalyticsEngine:
    """Advanced analytics for employee potential and risk prediction."""
    
    def __init__(self, config: PortraitConfig):
        self.config = config
        self.risk_factors = self._initialize_risk_factors()
        self.potential_indicators = self._initialize_potential_indicators()
    
    def _initialize_risk_factors(self) -> Dict[str, Dict]:
        """Initialize risk assessment factors."""
        
        return {
            "performance": {
                "weight": 0.3,
                "indicators": {
                    "below_expectations": 0.8,
                    "approaching_expectations": 0.4,
                    "meets_expectations": 0.1,
                    "exceeds_expectations": 0.05,
                }
            },
            "engagement": {
                "weight": 0.25,
                "threshold": 60,  # Below this score increases risk
                "max_risk": 0.7,
            },
            "tenure": {
                "weight": 0.15,
                "high_risk_periods": [(6, 18), (24, 36)],  # Months
            },
            "manager_relationship": {
                "weight": 0.2,
                "threshold": 70,
                "max_risk": 0.6,
            },
            "workload": {
                "weight": 0.1,
                "threshold": 120,  # % utilization
                "max_risk": 0.5,
            }
        }
    
    def _initialize_potential_indicators(self) -> Dict[str, Dict]:
        """Initialize high potential indicators."""
        
        return {
            "performance_trajectory": {
                "weight": 0.25,
                "factors": ["consistent_high_performance", "improvement_rate", "goal_achievement"]
            },
            "learning_agility": {
                "weight": 0.2,
                "factors": ["course_completion_rate", "diverse_skill_development", "quick_skill_acquisition"]
            },
            "leadership_behaviors": {
                "weight": 0.2,
                "factors": ["peer_influence", "initiative_taking", "team_collaboration"]
            },
            "adaptability": {
                "weight": 0.15,
                "factors": ["change_management", "problem_solving", "innovation"]
            },
            "client_impact": {
                "weight": 0.2,
                "factors": ["client_satisfaction", "value_delivery", "relationship_building"]
            }
        }
    
    def calculate_flight_risk(self, employee_data: Dict[str, Any], portrait: EmployeePortrait) -> float:
        """Calculate employee flight risk score."""
        
        risk_score = 0.0
        
        # Performance-based risk
        performance_data = employee_data.get("performance_system", {})
        current_rating = performance_data.get("current_rating", "MEETS_EXPECTATIONS")
        performance_risk = self.risk_factors["performance"]["indicators"].get(
            current_rating.lower(), 0.1
        )
        risk_score += performance_risk * self.risk_factors["performance"]["weight"]
        
        # Engagement-based risk
        survey_data = employee_data.get("survey_system", {})
        engagement_score = survey_data.get("engagement_score", 75)
        engagement_threshold = self.risk_factors["engagement"]["threshold"]
        
        if engagement_score < engagement_threshold:
            engagement_risk = min(
                (engagement_threshold - engagement_score) / engagement_threshold,
                self.risk_factors["engagement"]["max_risk"]
            )
        else:
            engagement_risk = 0.0
        
        risk_score += engagement_risk * self.risk_factors["engagement"]["weight"]
        
        # Tenure-based risk
        hire_date = datetime.fromisoformat(employee_data["hr_system"]["personal_info"]["hire_date"])
        tenure_months = (datetime.now() - hire_date).days / 30.44
        
        tenure_risk = 0.0
        for start, end in self.risk_factors["tenure"]["high_risk_periods"]:
            if start <= tenure_months <= end:
                tenure_risk = 0.3  # Moderate risk during these periods
                break
        
        risk_score += tenure_risk * self.risk_factors["tenure"]["weight"]
        
        # Manager relationship risk
        manager_score = survey_data.get("retention_indicators", {}).get("manager_relationship", 80)
        manager_threshold = self.risk_factors["manager_relationship"]["threshold"]
        
        if manager_score < manager_threshold:
            manager_risk = min(
                (manager_threshold - manager_score) / manager_threshold,
                self.risk_factors["manager_relationship"]["max_risk"]
            )
        else:
            manager_risk = 0.0
        
        risk_score += manager_risk * self.risk_factors["manager_relationship"]["weight"]
        
        return min(risk_score, 1.0)
    
    def calculate_potential_score(self, employee_data: Dict[str, Any], skill_assessments: Dict) -> float:
        """Calculate high potential score."""
        
        potential_score = 0.0
        
        # Performance trajectory
        performance_data = employee_data.get("performance_system", {})
        performance_history = performance_data.get("performance_history", [])
        
        if len(performance_history) >= 2:
            # Calculate improvement trend
            recent_ratings = [self._rating_to_numeric(period["overall_rating"]) for period in performance_history[-2:]]
            performance_trajectory = (recent_ratings[-1] - recent_ratings[0]) / len(recent_ratings)
        else:
            performance_trajectory = 0.0
        
        trajectory_score = min(max(performance_trajectory + 0.5, 0.0), 1.0)
        potential_score += trajectory_score * self.potential_indicators["performance_trajectory"]["weight"]
        
        # Learning agility
        learning_data = employee_data.get("learning_system", {})
        completed_courses = len(learning_data.get("completed_courses", []))
        learning_hours = learning_data.get("learning_hours_ytd", 0)
        
        learning_agility_score = min((completed_courses * 0.2) + (learning_hours / 200), 1.0)
        potential_score += learning_agility_score * self.potential_indicators["learning_agility"]["weight"]
        
        # Leadership behaviors (from skill assessments)
        leadership_assessment = skill_assessments.get(SkillCategory.LEADERSHIP)
        if leadership_assessment:
            leadership_score = leadership_assessment.current_level
        else:
            leadership_score = 0.3
        
        potential_score += leadership_score * self.potential_indicators["leadership_behaviors"]["weight"]
        
        # Adaptability (from skill assessments)
        adaptability_assessment = skill_assessments.get(SkillCategory.ADAPTABILITY)
        if adaptability_assessment:
            adaptability_score = adaptability_assessment.current_level
        else:
            adaptability_score = 0.3
        
        potential_score += adaptability_score * self.potential_indicators["adaptability"]["weight"]
        
        # Client impact
        project_data = employee_data.get("project_system", {})
        avg_client_satisfaction = project_data.get("project_performance_avg", 3.0)
        client_impact_score = min(avg_client_satisfaction / 5.0, 1.0)
        
        potential_score += client_impact_score * self.potential_indicators["client_impact"]["weight"]
        
        return min(potential_score, 1.0)
    
    def _rating_to_numeric(self, rating: str) -> float:
        """Convert performance rating to numeric value."""
        
        rating_map = {
            "EXCEEDS_EXPECTATIONS": 1.0,
            "MEETS_EXPECTATIONS": 0.7,
            "APPROACHING_EXPECTATIONS": 0.4,
            "BELOW_EXPECTATIONS": 0.1,
            "NEW_HIRE": 0.5,
        }
        
        return rating_map.get(rating, 0.5)
    
    def predict_career_progression(self, portrait: EmployeePortrait) -> Dict[str, Any]:
        """Predict career progression timeline and opportunities."""
        
        # Calculate readiness for next level
        current_potential = portrait.overall_potential_score
        skill_readiness = self._calculate_skill_readiness(portrait.skill_assessments)
        performance_readiness = self._calculate_performance_readiness(portrait.performance_history)
        
        overall_readiness = (current_potential * 0.4 + skill_readiness * 0.4 + performance_readiness * 0.2)
        
        # Predict timeline based on readiness and growth velocity
        if overall_readiness >= 0.8:
            predicted_months = 6
            confidence = 0.8
        elif overall_readiness >= 0.6:
            predicted_months = 12
            confidence = 0.7
        elif overall_readiness >= 0.4:
            predicted_months = 18
            confidence = 0.6
        else:
            predicted_months = 24
            confidence = 0.4
        
        # Adjust based on growth velocity
        if portrait.growth_velocity > 0.1:
            predicted_months = int(predicted_months * 0.8)
        elif portrait.growth_velocity < 0.05:
            predicted_months = int(predicted_months * 1.3)
        
        return {
            "readiness_score": overall_readiness,
            "predicted_promotion_months": predicted_months,
            "confidence": confidence,
            "key_development_areas": self._identify_development_priorities(portrait),
            "recommended_experiences": self._recommend_experiences(portrait),
        }
    
    def _calculate_skill_readiness(self, skill_assessments: Dict) -> float:
        """Calculate readiness based on skill development."""
        
        if not skill_assessments:
            return 0.3
        
        # Focus on key skills for progression
        key_skills = [SkillCategory.LEADERSHIP, SkillCategory.COMMUNICATION, 
                     SkillCategory.SAP_FUNCTIONAL, SkillCategory.PROJECT_MANAGEMENT]
        
        key_skill_scores = []
        for skill in key_skills:
            if skill in skill_assessments:
                key_skill_scores.append(skill_assessments[skill].current_level)
        
        if key_skill_scores:
            return statistics.mean(key_skill_scores)
        else:
            return 0.3
    
    def _calculate_performance_readiness(self, performance_history: List) -> float:
        """Calculate readiness based on performance history."""
        
        if not performance_history:
            return 0.3
        
        recent_performance = performance_history[-1] if performance_history else None
        if not recent_performance:
            return 0.3
        
        rating = recent_performance.overall_rating
        goals_ratio = recent_performance.goals_achieved / max(recent_performance.goals_total, 1)
        
        rating_score = self._rating_to_numeric(rating.value if hasattr(rating, 'value') else rating)
        
        return (rating_score * 0.7) + (goals_ratio * 0.3)
    
    def _identify_development_priorities(self, portrait: EmployeePortrait) -> List[str]:
        """Identify top development priorities for career progression."""
        
        priorities = []
        
        # Analyze skill gaps
        for skill, assessment in portrait.skill_assessments.items():
            gap = assessment.target_level - assessment.current_level
            if gap > 0.2:
                priorities.append(f"Develop {skill.value.replace('_', ' ').title()}")
        
        # Add leadership development for high potential employees
        if portrait.overall_potential_score > 0.7:
            priorities.append("Leadership Development Program")
            priorities.append("Mentorship Assignment")
        
        return priorities[:5]  # Top 5 priorities
    
    def _recommend_experiences(self, portrait: EmployeePortrait) -> List[str]:
        """Recommend stretch experiences for development."""
        
        experiences = []
        
        # Based on employee type and potential
        if portrait.employee_type == EmployeeType.HIGH_POTENTIAL:
            experiences.extend([
                "Lead cross-functional project",
                "International assignment opportunity",
                "Executive shadowing program",
                "Strategic initiative participation",
            ])
        elif portrait.employee_type == EmployeeType.EARLY_TALENT:
            experiences.extend([
                "Mentorship program participation",
                "Job rotation opportunity",
                "Client-facing project assignment",
                "Professional certification pursuit",
            ])
        
        # Based on skill assessments
        if SkillCategory.LEADERSHIP in portrait.skill_assessments:
            leadership_level = portrait.skill_assessments[SkillCategory.LEADERSHIP].current_level
            if leadership_level < 0.6:
                experiences.append("Team lead opportunity")
        
        return experiences[:4]  # Top 4 recommendations


# ========================================================================================
# PORTRAIT GENERATION ENGINE
# ========================================================================================


class PortraitGenerationEngine:
    """Core engine for generating comprehensive employee portraits."""
    
    def __init__(self, config: PortraitConfig, llm, data_collector: EmployeeDataCollector, 
                 skill_engine: SkillAssessmentEngine, analytics_engine: PredictiveAnalyticsEngine):
        self.config = config
        self.llm = llm
        self.data_collector = data_collector
        self.skill_engine = skill_engine
        self.analytics_engine = analytics_engine
        self.portrait_store = Path(config.portrait_store_path)
        self.portrait_store.mkdir(parents=True, exist_ok=True)
    
    def generate_employee_portrait(self, employee_id: str, force_refresh: bool = False) -> EmployeePortrait:
        """Generate comprehensive employee portrait."""
        
        # Check if recent portrait exists
        if not force_refresh:
            existing_portrait = self._load_existing_portrait(employee_id)
            if existing_portrait and self._is_portrait_recent(existing_portrait):
                print(f"Using existing portrait for {employee_id}")
                return existing_portrait
        
        print(f"Generating new portrait for {employee_id}")
        
        # Collect comprehensive data
        employee_data = self.data_collector.collect_employee_data(employee_id)
        
        # Create base portrait from HR data
        hr_data = employee_data.get("hr_system", {})
        personal_info = hr_data.get("personal_info", {})
        
        portrait = EmployeePortrait(
            employee_id=employee_id,
            name=personal_info.get("name", f"Employee_{employee_id}"),
            email=personal_info.get("email", ""),
            hire_date=datetime.fromisoformat(personal_info.get("hire_date", "2023-01-01")),
            current_role=personal_info.get("current_role", ""),
            department=personal_info.get("department", ""),
            manager_id=personal_info.get("manager_id"),
            data_sources=employee_data.get("data_sources", []),
        )
        
        # Assess skills
        portrait.skill_assessments = self.skill_engine.assess_skill_portfolio(employee_data, self.llm)
        
        # Process performance data
        portrait.performance_history = self._process_performance_data(employee_data)
        
        # Calculate analytics
        portrait.overall_potential_score = self.analytics_engine.calculate_potential_score(
            employee_data, portrait.skill_assessments
        )
        
        portrait.growth_velocity = self._calculate_growth_velocity(employee_data, portrait)
        
        # Risk assessment
        flight_risk = self.analytics_engine.calculate_flight_risk(employee_data, portrait)
        portrait.risk_indicators = self._create_risk_indicators(employee_data, flight_risk)
        
        # Classify employee type
        portrait.employee_type = self._classify_employee_type(portrait)
        
        # Career progression analysis
        portrait.career_progression = self._analyze_career_progression(employee_data, portrait)
        
        # Generate insights
        portrait.insights_generated = self._generate_insights(portrait, employee_data)
        
        # Save portrait
        self._save_portrait(portrait)
        
        return portrait
    
    def _load_existing_portrait(self, employee_id: str) -> Optional[EmployeePortrait]:
        """Load existing portrait from storage."""
        
        portrait_file = self.portrait_store / f"{employee_id}_portrait.json"
        
        if not portrait_file.exists():
            return None
        
        try:
            with open(portrait_file, "r") as f:
                data = json.load(f)
            
            # Reconstruct portrait object (simplified version)
            portrait = EmployeePortrait(
                employee_id=data["employee_id"],
                name=data["name"],
                email=data["email"],
                hire_date=datetime.fromisoformat(data["hire_date"]),
                current_role=data["current_role"],
                department=data["department"],
                manager_id=data.get("manager_id"),
                overall_potential_score=data.get("overall_potential_score", 0.5),
                growth_velocity=data.get("growth_velocity", 0.0),
                last_updated=datetime.fromisoformat(data["last_updated"]),
            )
            
            return portrait
            
        except Exception as e:
            print(f"Error loading portrait for {employee_id}: {e}")
            return None
    
    def _is_portrait_recent(self, portrait: EmployeePortrait) -> bool:
        """Check if portrait is recent enough to use."""
        
        days_since_update = (datetime.now() - portrait.last_updated).days
        return days_since_update < self.config.portrait_refresh_days
    
    def _process_performance_data(self, employee_data: Dict) -> List[PerformanceData]:
        """Process and structure performance data."""
        
        performance_history = []
        performance_system = employee_data.get("performance_system", {})
        
        for period_data in performance_system.get("performance_history", []):
            perf_data = PerformanceData(
                period=period_data.get("period", "Unknown"),
                overall_rating=PerformanceIndicator.MEETS_EXPECTATIONS,  # Default
                goals_achieved=period_data.get("goals_achieved", 0),
                goals_total=period_data.get("goals_total", 1),
                manager_feedback=period_data.get("manager_feedback", ""),
                peer_ratings=performance_system.get("360_feedback", {}).get("peer_ratings", []),
            )
            performance_history.append(perf_data)
        
        return performance_history
    
    def _calculate_growth_velocity(self, employee_data: Dict, portrait: EmployeePortrait) -> float:
        """Calculate employee's growth velocity."""
        
        # Calculate based on skill improvements over time
        skill_improvements = []
        for skill, assessment in portrait.skill_assessments.items():
            skill_improvements.append(assessment.growth_trajectory)
        
        # Calculate based on learning activity
        learning_data = employee_data.get("learning_system", {})
        learning_hours = learning_data.get("learning_hours_ytd", 0)
        courses_completed = len(learning_data.get("completed_courses", []))
        
        learning_velocity = min((courses_completed * 0.1) + (learning_hours / 1000), 1.0)
        
        # Combine factors
        if skill_improvements:
            skill_velocity = statistics.mean(skill_improvements)
            return (skill_velocity * 0.6) + (learning_velocity * 0.4)
        else:
            return learning_velocity
    
    def _create_risk_indicators(self, employee_data: Dict, flight_risk: float) -> RiskIndicators:
        """Create comprehensive risk indicators."""
        
        survey_data = employee_data.get("survey_system", {})
        engagement_score = survey_data.get("engagement_score", 75) / 100
        
        # Identify red flags
        red_flags = []
        if flight_risk > 0.6:
            red_flags.append("High flight risk score")
        if engagement_score < 0.6:
            red_flags.append("Low engagement score")
        
        # Identify protective factors
        protective_factors = []
        if engagement_score > 0.8:
            protective_factors.append("High engagement")
        
        manager_relationship = survey_data.get("retention_indicators", {}).get("manager_relationship", 80)
        if manager_relationship > 85:
            protective_factors.append("Strong manager relationship")
        
        # Recommend interventions
        interventions = []
        if flight_risk > 0.4:
            interventions.append("Schedule retention conversation")
        if engagement_score < 0.7:
            interventions.append("Engagement improvement plan")
        
        return RiskIndicators(
            flight_risk_score=flight_risk,
            performance_risk_score=0.0,  # Calculate based on performance trends
            engagement_score=engagement_score,
            last_survey_date=datetime.now() - timedelta(days=30),
            red_flags=red_flags,
            protective_factors=protective_factors,
            recommended_interventions=interventions,
        )
    
    def _classify_employee_type(self, portrait: EmployeePortrait) -> EmployeeType:
        """Classify employee based on various factors."""
        
        # Calculate tenure in months
        tenure_months = (datetime.now() - portrait.hire_date).days / 30.44
        
        # Check for high potential
        if portrait.overall_potential_score > self.config.high_potential_threshold:
            return EmployeeType.HIGH_POTENTIAL
        
        # Check for risk
        if portrait.risk_indicators and portrait.risk_indicators.flight_risk_score > self.config.risk_threshold:
            return EmployeeType.AT_RISK
        
        # Classify by experience level
        if tenure_months < 12:
            return EmployeeType.EARLY_TALENT
        elif tenure_months < 36:
            return EmployeeType.DEVELOPING_PROFESSIONAL
        else:
            return EmployeeType.EXPERIENCED_PROFESSIONAL
    
    def _analyze_career_progression(self, employee_data: Dict, portrait: EmployeePortrait) -> CareerProgression:
        """Analyze career progression potential."""
        
        hr_data = employee_data.get("hr_system", {})
        personal_info = hr_data.get("personal_info", {})
        
        # Determine career track based on role and skills
        current_role = personal_info.get("current_role", "").lower()
        
        if "consultant" in current_role or "advisory" in current_role:
            career_track = CareerTrack.FUNCTIONAL_CONSULTANT
        elif "developer" in current_role or "technical" in current_role:
            career_track = CareerTrack.TECHNICAL_EXPERT
        elif "manager" in current_role or "lead" in current_role:
            career_track = CareerTrack.TEAM_LEADER
        else:
            career_track = CareerTrack.FUNCTIONAL_CONSULTANT  # Default
        
        # Calculate promotion readiness
        promotion_prediction = self.analytics_engine.predict_career_progression(portrait)
        
        tenure_months = (datetime.now() - portrait.hire_date).days / 30.44
        
        return CareerProgression(
            current_role=portrait.current_role,
            current_level="Associate" if tenure_months < 24 else "Consultant",
            career_track=career_track,
            tenure_months=int(tenure_months),
            promotion_readiness=promotion_prediction["readiness_score"],
            next_role_target=self._suggest_next_role(career_track, portrait),
            development_areas=self._extract_development_areas(portrait),
            mentor_assigned=False,  # Would be determined from HR system
            succession_planning=portrait.employee_type == EmployeeType.HIGH_POTENTIAL,
        )
    
    def _suggest_next_role(self, career_track: CareerTrack, portrait: EmployeePortrait) -> str:
        """Suggest next role based on career track and readiness."""
        
        role_progressions = {
            CareerTrack.FUNCTIONAL_CONSULTANT: [
                "Senior Consultant", "Principal Consultant", "Director"
            ],
            CareerTrack.TECHNICAL_EXPERT: [
                "Senior Developer", "Technical Lead", "Architect"
            ],
            CareerTrack.TEAM_LEADER: [
                "Team Manager", "Department Manager", "VP"
            ],
        }
        
        progression_path = role_progressions.get(career_track, ["Senior Role"])
        
        if portrait.overall_potential_score > 0.8:
            return progression_path[0] if progression_path else "Senior Role"
        else:
            return f"Continue development toward {progression_path[0] if progression_path else 'Senior Role'}"
    
    def _extract_development_areas(self, portrait: EmployeePortrait) -> List[SkillCategory]:
        """Extract key development areas from skill assessments."""
        
        development_areas = []
        
        for skill, assessment in portrait.skill_assessments.items():
            gap = assessment.target_level - assessment.current_level
            if gap > 0.2:  # Significant gap
                development_areas.append(skill)
        
        # Sort by gap size
        development_areas.sort(
            key=lambda s: portrait.skill_assessments[s].target_level - portrait.skill_assessments[s].current_level,
            reverse=True
        )
        
        return development_areas[:3]  # Top 3 development areas
    
    def _generate_insights(self, portrait: EmployeePortrait, employee_data: Dict) -> List[Dict[str, Any]]:
        """Generate AI-powered insights about the employee."""
        
        insights = []
        
        # Performance insights
        if portrait.overall_potential_score > 0.8:
            insights.append({
                "type": "opportunity",
                "title": "High Potential Identified",
                "description": f"Employee shows strong potential with score of {portrait.overall_potential_score:.2f}",
                "priority": "high",
                "actions": ["Consider for accelerated development", "Assign stretch projects"],
            })
        
        # Risk insights
        if portrait.risk_indicators and portrait.risk_indicators.flight_risk_score > 0.4:
            insights.append({
                "type": "risk",
                "title": "Retention Risk Detected",
                "description": f"Flight risk score of {portrait.risk_indicators.flight_risk_score:.2f} indicates retention concerns",
                "priority": "high",
                "actions": ["Schedule retention conversation", "Review career development plan"],
            })
        
        # Development insights
        skill_gaps = self.skill_engine.identify_skill_gaps(portrait.skill_assessments)
        if skill_gaps:
            top_gap = skill_gaps[0]
            insights.append({
                "type": "development",
                "title": f"Development Opportunity in {top_gap['skill']}",
                "description": f"Significant gap identified with {top_gap['gap_size']:.2f} point difference",
                "priority": top_gap["priority"],
                "actions": [f"Enroll in {top_gap['skill']} training", "Assign relevant projects"],
            })
        
        return insights
    
    def _save_portrait(self, portrait: EmployeePortrait):
        """Save portrait to persistent storage."""
        
        portrait_file = self.portrait_store / f"{portrait.employee_id}_portrait.json"
        
        # Convert to serializable format
        portrait_data = {
            "employee_id": portrait.employee_id,
            "name": portrait.name,
            "email": portrait.email,
            "hire_date": portrait.hire_date.isoformat(),
            "current_role": portrait.current_role,
            "department": portrait.department,
            "manager_id": portrait.manager_id,
            "employee_type": portrait.employee_type.value,
            "overall_potential_score": portrait.overall_potential_score,
            "growth_velocity": portrait.growth_velocity,
            "last_updated": portrait.last_updated.isoformat(),
            "portrait_version": portrait.portrait_version,
            "data_sources": portrait.data_sources,
            "insights_generated": portrait.insights_generated,
        }
        
        with open(portrait_file, "w") as f:
            json.dump(portrait_data, f, indent=2)


# ========================================================================================
# PORTRAIT RAG PIPELINE
# ========================================================================================


class PortraitRAGPipeline:
    """RAG pipeline for portrait-based insights and recommendations."""
    
    def __init__(self, llm, vector_manager, config: PortraitConfig):
        self.llm = llm
        self.vector_manager = vector_manager
        self.config = config
        
        # Portrait analysis prompt
        self.portrait_prompt = (
            PromptTemplate(
                template="""You are an expert HR analyst specializing in employee development and talent management.

Employee Portrait Data:
{portrait_data}

Context Information:
{context}

HR Query: {question}

Instructions:
- Provide actionable insights based on the employee portrait
- Include specific recommendations for HR actions
- Consider development opportunities and risk factors
- Format response professionally for HR stakeholders
- Include confidence levels and supporting evidence

HR Analysis:""",
                input_variables=["portrait_data", "context", "question"],
            )
            if LANGCHAIN_AVAILABLE
            else None
        )
    
    def analyze_employee_portrait(self, employee_id: str, query: str, portrait: EmployeePortrait) -> Dict[str, Any]:
        """Analyze employee portrait and provide insights."""
        
        try:
            # Format portrait data for analysis
            portrait_summary = self._format_portrait_summary(portrait)
            
            # Retrieve relevant context
            context_docs = self.vector_manager.similarity_search(
                f"{query} {employee_id} {portrait.current_role}"
            )
            context = self._format_context(context_docs)
            
            # Generate analysis
            if self.portrait_prompt and LANGCHAIN_AVAILABLE:
                prompt = self.portrait_prompt.format(
                    portrait_data=portrait_summary,
                    context=context,
                    question=query
                )
            else:
                prompt = f"""
                Analyze this employee portrait and answer the question:
                
                Employee: {portrait.name} ({employee_id})
                Role: {portrait.current_role}
                Potential Score: {portrait.overall_potential_score:.2f}
                Employee Type: {portrait.employee_type.value}
                
                Question: {query}
                
                Analysis:
                """
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, "content"):
                analysis = response.content.strip()
            else:
                analysis = str(response).strip()
            
            return {
                "employee_id": employee_id,
                "analysis": analysis,
                "portrait_summary": portrait_summary,
                "confidence": 0.8,
                "recommendations": self._extract_recommendations(analysis),
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            print(f"Error analyzing portrait: {e}")
            return {
                "employee_id": employee_id,
                "analysis": "Unable to generate analysis due to technical issues.",
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
            }
    
    def _format_portrait_summary(self, portrait: EmployeePortrait) -> str:
        """Format portrait data for analysis."""
        
        summary_parts = [
            f"Employee: {portrait.name} ({portrait.employee_id})",
            f"Role: {portrait.current_role} in {portrait.department}",
            f"Hire Date: {portrait.hire_date.strftime('%Y-%m-%d')}",
            f"Employee Type: {portrait.employee_type.value}",
            f"Potential Score: {portrait.overall_potential_score:.2f}",
            f"Growth Velocity: {portrait.growth_velocity:.2f}",
        ]
        
        # Add skill summary
        if portrait.skill_assessments:
            summary_parts.append("\nTop Skills:")
            sorted_skills = sorted(
                portrait.skill_assessments.items(),
                key=lambda x: x[1].current_level,
                reverse=True
            )
            for skill, assessment in sorted_skills[:5]:
                summary_parts.append(f"  {skill.value}: {assessment.current_level:.2f}")
        
        # Add risk indicators
        if portrait.risk_indicators:
            summary_parts.append(f"\nFlight Risk: {portrait.risk_indicators.flight_risk_score:.2f}")
            summary_parts.append(f"Engagement: {portrait.risk_indicators.engagement_score:.2f}")
            
            if portrait.risk_indicators.red_flags:
                summary_parts.append(f"Risk Factors: {', '.join(portrait.risk_indicators.red_flags)}")
        
        # Add recent insights
        if portrait.insights_generated:
            summary_parts.append("\nRecent Insights:")
            for insight in portrait.insights_generated[:3]:
                summary_parts.append(f"  {insight.get('title', 'Insight')}: {insight.get('description', '')}")
        
        return "\n".join(summary_parts)
    
    def _format_context(self, documents: List) -> str:
        """Format retrieved documents into context."""
        
        if not documents:
            return "No additional context available."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"Context {i} ({source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract actionable recommendations from analysis."""
        
        recommendations = []
        
        # Simple extraction based on common recommendation patterns
        lines = analysis.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should', 'consider']):
                recommendations.append(line.strip())
        
        # If no specific recommendations found, provide general ones
        if not recommendations:
            recommendations = [
                "Continue monitoring employee development",
                "Schedule regular career conversations",
                "Provide relevant learning opportunities",
            ]
        
        return recommendations[:5]  # Top 5 recommendations


# ========================================================================================
# MAIN EMPLOYEE PORTRAIT WORKFLOW
# ========================================================================================


class EmployeePortraitWorkflow:
    """Main orchestrator for employee portrait system."""
    
    def __init__(self, llm, embeddings, config: PortraitConfig = None):
        self.config = config or PortraitConfig()
        self.llm = llm
        self.embeddings = embeddings
        self.initialization_time = datetime.now()
        
        try:
            print("Initializing Employee Portrait System...")
            
            # Initialize components
            self.data_collector = EmployeeDataCollector(self.config)
            self.skill_engine = SkillAssessmentEngine(self.config)
            self.analytics_engine = PredictiveAnalyticsEngine(self.config)
            
            # Initialize vector store (for storing portrait data and insights)
            self.vector_manager = self._initialize_vector_store()
            
            # Initialize portrait generation engine
            self.portrait_engine = PortraitGenerationEngine(
                self.config, llm, self.data_collector, 
                self.skill_engine, self.analytics_engine
            )
            
            # Initialize RAG pipeline
            self.rag_pipeline = PortraitRAGPipeline(llm, self.vector_manager, self.config)
            
            self.initialization_successful = True
            print("Employee Portrait System initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing portrait system: {e}")
            self.initialization_successful = False
            self.initialization_error = str(e)
    
    def _initialize_vector_store(self):
        """Initialize vector store for portrait insights."""
        
        # Create sample documents about portrait analysis
        sample_docs = self._create_sample_docs()
        
        # Initialize vector store manager
        from re_upskilling_component import VectorStoreManager
        vector_manager = VectorStoreManager(self.config, self.embeddings)
        vector_store = vector_manager.get_or_create_vector_store(sample_docs)
        
        return vector_manager
    
    def _create_sample_docs(self) -> List:
        """Create sample documents for portrait analysis."""
        
        sample_content = [
            {
                "content": """
                Employee Portrait Analysis Guidelines
                
                High Potential Indicators:
                - Consistent performance above expectations
                - Rapid skill development and learning agility
                - Strong leadership behaviors and influence
                - Positive client feedback and business impact
                - Adaptability to change and challenges
                
                Development Recommendations:
                - Stretch assignments and challenging projects
                - Leadership development programs
                - Cross-functional experience
                - Mentorship and coaching
                - Executive visibility opportunities
                """,
                "metadata": {"source": "hr_guidelines", "type": "analysis"},
            },
            {
                "content": """
                Retention Risk Factors and Interventions
                
                Common Risk Indicators:
                - Declining engagement scores
                - Performance plateaus or declines
                - Limited career progression opportunities
                - Work-life balance challenges
                - Weak manager relationships
                
                Intervention Strategies:
                - Career development conversations
                - Role enrichment and job crafting
                - Manager coaching and support
                - Work arrangement flexibility
                - Recognition and rewards programs
                """,
                "metadata": {"source": "retention_guide", "type": "risk_management"},
            },
        ]
        
        return [
            (
                Document(page_content=doc["content"], metadata=doc["metadata"])
                if LANGCHAIN_AVAILABLE
                else MockDocument(doc["content"], doc["metadata"])
            )
            for doc in sample_content
        ]
    
    def handle_query(self, query: str, employee_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle portrait-related queries."""
        
        if not self.initialization_successful:
            return self._create_initialization_error_response()
        
        print(f"Processing portrait query: {query[:100]}...")
        
        try:
            if employee_id:
                # Employee-specific query
                portrait = self.get_employee_portrait(employee_id)
                
                response_data = self.rag_pipeline.analyze_employee_portrait(
                    employee_id, query, portrait
                )
                
                return {
                    "answer": response_data["analysis"],
                    "confidence": response_data["confidence"],
                    "sources": ["Employee Portrait System"],
                    "employee_id": employee_id,
                    "portrait_summary": response_data.get("portrait_summary", ""),
                    "recommendations": response_data.get("recommendations", []),
                    "workflow": "employee_portrait",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # General portrait system query
                return self._handle_general_query(query)
                
        except Exception as e:
            print(f"Error in portrait workflow: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your portrait query.",
                "confidence": 0.0,
                "sources": [],
                "workflow": "portrait_error",
                "error_details": {"error_message": str(e)},
                "timestamp": datetime.now().isoformat(),
            }
    
    def get_employee_portrait(self, employee_id: str, force_refresh: bool = False) -> EmployeePortrait:
        """Get comprehensive employee portrait."""
        
        return self.portrait_engine.generate_employee_portrait(employee_id, force_refresh)
    
    def get_team_insights(self, manager_id: str) -> Dict[str, Any]:
        """Get insights for all direct reports of a manager."""
        
        # In real implementation, would query HR system for direct reports
        # For demo, using sample employee IDs
        sample_team = ["EMP_001", "EMP_002", "EMP_003"]
        
        team_insights = {
            "manager_id": manager_id,
            "team_size": len(sample_team),
            "high_potential_employees": [],
            "at_risk_employees": [],
            "development_priorities": {},
            "team_analytics": {},
        }
        
        for employee_id in sample_team:
            try:
                portrait = self.get_employee_portrait(employee_id)
                
                if portrait.employee_type == EmployeeType.HIGH_POTENTIAL:
                    team_insights["high_potential_employees"].append({
                        "id": employee_id,
                        "name": portrait.name,
                        "potential_score": portrait.overall_potential_score,
                    })
                
                if portrait.employee_type == EmployeeType.AT_RISK:
                    team_insights["at_risk_employees"].append({
                        "id": employee_id,
                        "name": portrait.name,
                        "risk_score": portrait.risk_indicators.flight_risk_score if portrait.risk_indicators else 0.0,
                    })
                
            except Exception as e:
                print(f"Error processing team member {employee_id}: {e}")
        
        return team_insights
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries about the portrait system."""
        
        general_prompt = f"""
        You are an expert HR analyst. Answer this question about employee portrait analysis:
        
        Question: {query}
        
        Provide a comprehensive answer covering:
        - Key concepts and methodologies
        - Best practices and recommendations  
        - Implementation considerations
        
        Answer:
        """
        
        try:
            response = self.llm.invoke(general_prompt)
            
            if hasattr(response, "content"):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
            
            return {
                "answer": answer,
                "confidence": 0.7,
                "sources": ["HR Best Practices", "Portrait Analytics"],
                "workflow": "portrait_general",
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            return {
                "answer": "I apologize, but I couldn't process your general portrait query.",
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    def _create_initialization_error_response(self) -> Dict[str, Any]:
        """Create error response for initialization failures."""
        
        return {
            "answer": "The Employee Portrait System is currently unavailable.",
            "confidence": 0.0,
            "sources": [],
            "workflow": "portrait_init_error",
            "timestamp": datetime.now().isoformat(),
            "error_details": {
                "initialization_error": getattr(self, "initialization_error", "Unknown error")
            },
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration."""
        
        return {
            "initialization_successful": self.initialization_successful,
            "initialization_time": self.initialization_time.isoformat(),
            "config": {
                "data_path": self.config.data_path,
                "portrait_store_path": self.config.portrait_store_path,
                "confidence_threshold": self.config.confidence_threshold,
                "portrait_refresh_days": self.config.portrait_refresh_days,
                "predictive_analytics_enabled": self.config.enable_predictive_analytics,
            },
            "components_available": COMPONENTS_AVAILABLE,
            "langchain_available": LANGCHAIN_AVAILABLE,
        }


# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================


def create_sample_employee_data(employee_id: str = "EMP_001") -> Dict[str, Any]:
    """Create sample employee data for testing."""
    
    collector = EmployeeDataCollector(PortraitConfig())
    return collector.collect_employee_data(employee_id)


def run_portrait_demo():
    """Run a demonstration of the employee portrait system."""
    
    print("Employee Portrait System Demo")
    print("=" * 60)
    
    # Initialize mock components
    llm = MockLLM()
    
    class MockEmbeddings:
        def __init__(self):
            self.model_name = "mock-embeddings"
    
    embeddings = MockEmbeddings()
    
    # Initialize workflow
    config = PortraitConfig()
    workflow = EmployeePortraitWorkflow(llm, embeddings, config)
    
    # Test employee portrait generation
    employee_id = "EMP_001"
    portrait = workflow.get_employee_portrait(employee_id)
    
    print(f"\nEmployee Portrait Generated:")
    print(f"  Name: {portrait.name}")
    print(f"  Role: {portrait.current_role}")
    print(f"  Type: {portrait.employee_type.value}")
    print(f"  Potential Score: {portrait.overall_potential_score:.2f}")
    print(f"  Growth Velocity: {portrait.growth_velocity:.2f}")
    
    # Test queries
    test_queries = [
        f"What development opportunities should we provide for {employee_id}?",
        f"Is {employee_id} ready for promotion?",
        f"What are the retention risks for {employee_id}?",
        "How do we identify high potential employees?",
    ]
    
    print(f"\nTesting Portrait Queries:")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        if "EMP_001" in query:
            response = workflow.handle_query(query, employee_id="EMP_001")
        else:
            response = workflow.handle_query(query)
        
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Confidence: {response['confidence']:.2f}")
        
        if "recommendations" in response:
            print(f"Recommendations: {response['recommendations'][:2]}")
    
    # Test team insights
    print(f"\n{'='*60}")
    print("Testing Team Insights:")
    print("-" * 60)
    
    team_insights = workflow.get_team_insights("MGR_001")
    print(f"Team Size: {team_insights['team_size']}")
    print(f"High Potential: {len(team_insights['high_potential_employees'])}")
    print(f"At Risk: {len(team_insights['at_risk_employees'])}")
    
    # Display system status
    print(f"\n{'='*60}")
    print("System Status:")
    status = workflow.get_system_status()
    print(f"Initialization: {'Success' if status['initialization_successful'] else 'Failed'}")
    print(f"Components Available: {status['components_available']}")
    print(f"Predictive Analytics: {status['config']['predictive_analytics_enabled']}")


# ========================================================================================
# MAIN EXECUTION
# ========================================================================================


if __name__ == "__main__":
    run_portrait_demo()