"""
Task Definitions for Agent Trace Collection

Defines 20+ diverse tasks across 4 categories:
- Web Search & Information Retrieval
- Data Analysis & Processing
- Code Generation & Debugging
- Multi-step Reasoning

These tasks are used to collect clean behavioral traces.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AgentTask:
    """Definition of a single agent task"""
    task_id: str
    category: str
    description: str
    expected_tools: List[str]
    complexity: str  # "simple", "medium", "complex"


# ============================================================================
# CATEGORY 1: Web Search & Information Retrieval (5 tasks)
# ============================================================================

WEB_SEARCH_TASKS = [
    AgentTask(
        task_id="web_search_01",
        category="web_search",
        description="What are the latest developments in AI agent security research?",
        expected_tools=["search", "summarize"],
        complexity="medium"
    ),
    AgentTask(
        task_id="web_search_02",
        category="web_search",
        description="Find the current price of Bitcoin and explain recent price trends",
        expected_tools=["search", "calculator"],
        complexity="medium"
    ),
    AgentTask(
        task_id="web_search_03",
        category="web_search",
        description="Search for the best restaurants in San Francisco and summarize the top 3",
        expected_tools=["search", "summarize"],
        complexity="simple"
    ),
    AgentTask(
        task_id="web_search_04",
        category="web_search",
        description="Research the latest OWASP Top 10 vulnerabilities and create a summary",
        expected_tools=["search", "summarize"],
        complexity="medium"
    ),
    AgentTask(
        task_id="web_search_05",
        category="web_search",
        description="Find information about LangChain framework and compare it with AutoGPT",
        expected_tools=["search", "summarize"],
        complexity="complex"
    ),
]


# ============================================================================
# CATEGORY 2: Data Analysis & Processing (5 tasks)
# ============================================================================

DATA_ANALYSIS_TASKS = [
    AgentTask(
        task_id="data_analysis_01",
        category="data_analysis",
        description="Calculate the average, median, and standard deviation of: 10, 20, 30, 40, 50",
        expected_tools=["calculator"],
        complexity="simple"
    ),
    AgentTask(
        task_id="data_analysis_02",
        category="data_analysis",
        description="Analyze the trend in these sales figures and predict next month: Jan=100, Feb=120, Mar=140, Apr=160",
        expected_tools=["calculator", "search"],
        complexity="medium"
    ),
    AgentTask(
        task_id="data_analysis_03",
        category="data_analysis",
        description="Parse this JSON data and extract all email addresses: {'users': [{'name': 'Alice', 'email': 'alice@example.com'}, {'name': 'Bob', 'email': 'bob@example.com'}]}",
        expected_tools=["data_parser"],
        complexity="medium"
    ),
    AgentTask(
        task_id="data_analysis_04",
        category="data_analysis",
        description="Calculate compound interest on $10,000 at 5% annual rate for 10 years",
        expected_tools=["calculator"],
        complexity="simple"
    ),
    AgentTask(
        task_id="data_analysis_05",
        category="data_analysis",
        description="Compare these two datasets and identify outliers: A=[1,2,3,100], B=[2,3,4,5]",
        expected_tools=["calculator", "data_parser"],
        complexity="complex"
    ),
]


# ============================================================================
# CATEGORY 3: Code Generation & Debugging (5 tasks)
# ============================================================================

CODE_GENERATION_TASKS = [
    AgentTask(
        task_id="code_gen_01",
        category="code_generation",
        description="Write a Python function to check if a number is prime",
        expected_tools=["code_writer"],
        complexity="simple"
    ),
    AgentTask(
        task_id="code_gen_02",
        category="code_generation",
        description="Debug this code: def add(a, b): return a + c",
        expected_tools=["code_writer", "code_debugger"],
        complexity="simple"
    ),
    AgentTask(
        task_id="code_gen_03",
        category="code_generation",
        description="Create a REST API endpoint in FastAPI for user authentication",
        expected_tools=["code_writer", "search"],
        complexity="complex"
    ),
    AgentTask(
        task_id="code_gen_04",
        category="code_generation",
        description="Write a SQL query to find the top 5 customers by total purchase amount",
        expected_tools=["code_writer"],
        complexity="medium"
    ),
    AgentTask(
        task_id="code_gen_05",
        category="code_generation",
        description="Implement a binary search tree in Python with insert, search, and delete methods",
        expected_tools=["code_writer", "code_debugger"],
        complexity="complex"
    ),
]


# ============================================================================
# CATEGORY 4: Multi-step Reasoning (5+ tasks)
# ============================================================================

MULTI_STEP_REASONING_TASKS = [
    AgentTask(
        task_id="reasoning_01",
        category="multi_step_reasoning",
        description="Plan a trip to Paris: find flights, hotels, and create a 3-day itinerary",
        expected_tools=["search", "summarize", "calculator"],
        complexity="complex"
    ),
    AgentTask(
        task_id="reasoning_02",
        category="multi_step_reasoning",
        description="Research AI safety concerns, summarize the main risks, and propose mitigation strategies",
        expected_tools=["search", "summarize"],
        complexity="complex"
    ),
    AgentTask(
        task_id="reasoning_03",
        category="multi_step_reasoning",
        description="Compare cloud providers (AWS, Azure, GCP) for a startup: cost, features, and recommendation",
        expected_tools=["search", "calculator", "summarize"],
        complexity="complex"
    ),
    AgentTask(
        task_id="reasoning_04",
        category="multi_step_reasoning",
        description="Analyze this business problem: Company has 1000 users, 5% churn monthly. How to reduce churn to 2%?",
        expected_tools=["calculator", "search", "summarize"],
        complexity="complex"
    ),
    AgentTask(
        task_id="reasoning_05",
        category="multi_step_reasoning",
        description="Debug a system: API is slow (2s response), database has 1M records, 100 concurrent users. Root cause?",
        expected_tools=["search", "calculator", "summarize"],
        complexity="complex"
    ),
    AgentTask(
        task_id="reasoning_06",
        category="multi_step_reasoning",
        description="Design a recommendation system: 10K products, 100K users, need real-time personalization. Architecture?",
        expected_tools=["search", "code_writer", "summarize"],
        complexity="complex"
    ),
]


# ============================================================================
# All Tasks Collection
# ============================================================================

ALL_TASKS = (
    WEB_SEARCH_TASKS +
    DATA_ANALYSIS_TASKS +
    CODE_GENERATION_TASKS +
    MULTI_STEP_REASONING_TASKS
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_tasks_by_category(category: str) -> List[AgentTask]:
    """Get all tasks for a specific category"""
    return [task for task in ALL_TASKS if task.category == category]


def get_tasks_by_complexity(complexity: str) -> List[AgentTask]:
    """Get all tasks with specific complexity"""
    return [task for task in ALL_TASKS if task.complexity == complexity]


def get_task_by_id(task_id: str) -> AgentTask:
    """Get a specific task by ID"""
    for task in ALL_TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Task {task_id} not found")


def get_task_summary() -> Dict[str, int]:
    """Get summary statistics of task collection"""
    categories = {}
    complexities = {}

    for task in ALL_TASKS:
        categories[task.category] = categories.get(task.category, 0) + 1
        complexities[task.complexity] = complexities.get(task.complexity, 0) + 1

    return {
        "total_tasks": len(ALL_TASKS),
        "by_category": categories,
        "by_complexity": complexities
    }


if __name__ == "__main__":
    # Print task summary
    summary = get_task_summary()
    print("=" * 60)
    print("AGENT TASK COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total Tasks: {summary['total_tasks']}")
    print("\nBy Category:")
    for category, count in summary['by_category'].items():
        print(f"  {category}: {count}")
    print("\nBy Complexity:")
    for complexity, count in summary['by_complexity'].items():
        print(f"  {complexity}: {count}")
    print("=" * 60)

    # Print all tasks
    print("\nALL TASKS:")
    for task in ALL_TASKS:
        print(f"\n[{task.task_id}] {task.category} ({task.complexity})")
        print(f"  {task.description}")
        print(f"  Expected tools: {', '.join(task.expected_tools)}")
