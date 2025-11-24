"""
LangChain Agent Wrapper with Instrumentation

Wraps LangChain agents to log execution traces for behavioral anomaly detection.
Captures: tool calls, parameters, timestamps, data flow, control flow.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


@dataclass
class TraceStep:
    """Single step in an agent execution trace"""
    step_id: int
    timestamp: str
    tool: str
    params: Dict[str, Any]
    data_in: Dict[str, Any]
    data_out: Dict[str, Any]
    duration_ms: float
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    """Complete execution trace for an agent task"""
    trace_id: str
    task_description: str
    timestamp: str
    steps: List[TraceStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class InstrumentationCallback(BaseCallbackHandler):
    """Callback handler to capture agent execution details"""

    def __init__(self):
        self.current_trace: Optional[ExecutionTrace] = None
        self.current_step: Optional[Dict[str, Any]] = None
        self.step_counter = 0

    def on_tool_start(self, tool_name: str, input_str: str, **kwargs) -> None:
        """Called when a tool starts executing"""
        self.step_counter += 1
        self.current_step = {
            "step_id": self.step_counter,
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "params": {"input": input_str},
            "data_in": {},
            "start_time": datetime.utcnow()
        }

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing"""
        if self.current_step:
            end_time = datetime.utcnow()
            duration = (end_time - self.current_step["start_time"]).total_seconds() * 1000

            step = TraceStep(
                step_id=self.current_step["step_id"],
                timestamp=self.current_step["timestamp"],
                tool=self.current_step["tool"],
                params=self.current_step["params"],
                data_in=self.current_step.get("data_in", {}),
                data_out={"output": output},
                duration_ms=duration
            )

            if self.current_trace:
                self.current_trace.steps.append(step)

            self.current_step = None

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error"""
        if self.current_step and self.current_trace:
            end_time = datetime.utcnow()
            duration = (end_time - self.current_step["start_time"]).total_seconds() * 1000

            step = TraceStep(
                step_id=self.current_step["step_id"],
                timestamp=self.current_step["timestamp"],
                tool=self.current_step["tool"],
                params=self.current_step["params"],
                data_in=self.current_step.get("data_in", {}),
                data_out={},
                duration_ms=duration,
                error=str(error)
            )

            self.current_trace.steps.append(step)
            self.current_trace.success = False
            self.current_trace.error_message = str(error)

        self.current_step = None


class InstrumentedAgent:
    """
    LangChain agent wrapper with execution tracing.

    Logs all tool calls, parameters, timestamps, and data flow for
    behavioral anomaly detection.
    """

    def __init__(
        self,
        tools: List[BaseTool],
        llm: Optional[Any] = None,
        output_dir: Path = Path("data/clean_traces")
    ):
        """
        Initialize instrumented agent.

        Args:
            tools: List of LangChain tools available to the agent
            llm: Language model to use (defaults to GPT-3.5)
            output_dir: Directory to save execution traces
        """
        self.tools = tools
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create ReAct prompt template
        self.prompt = PromptTemplate.from_template(
            """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""
        )

        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def execute(self, task_description: str) -> ExecutionTrace:
        """
        Execute a task and record full execution trace.

        Args:
            task_description: Natural language description of the task

        Returns:
            ExecutionTrace with all steps, timestamps, and data flow
        """
        # Create new trace
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            task_description=task_description,
            timestamp=datetime.utcnow().isoformat()
        )

        # Create callback handler
        callback = InstrumentationCallback()
        callback.current_trace = trace

        # Execute task with instrumentation
        start_time = datetime.utcnow()

        try:
            result = self.agent_executor.invoke(
                {"input": task_description},
                config={"callbacks": [callback]}
            )
            trace.total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        except Exception as e:
            trace.success = False
            trace.error_message = str(e)
            trace.total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Save trace to disk
        self.save_trace(trace)

        return trace

    def save_trace(self, trace: ExecutionTrace) -> None:
        """Save execution trace to JSON file"""
        trace_file = self.output_dir / f"trace_{trace.trace_id}.json"

        with open(trace_file, "w") as f:
            json.dump(asdict(trace), f, indent=2)

    @staticmethod
    def load_trace(trace_file: Path) -> ExecutionTrace:
        """Load execution trace from JSON file"""
        with open(trace_file, "r") as f:
            data = json.load(f)

        steps = [TraceStep(**step) for step in data["steps"]]
        data["steps"] = steps

        return ExecutionTrace(**data)


if __name__ == "__main__":
    # Example usage
    from langchain.tools import Tool

    def mock_search(query: str) -> str:
        """Mock search tool for testing"""
        return f"Results for: {query}"

    def mock_calculator(expression: str) -> str:
        """Mock calculator tool for testing"""
        try:
            result = eval(expression)
            return str(result)
        except:
            return "Invalid expression"

    # Create tools
    tools = [
        Tool(
            name="Search",
            func=mock_search,
            description="Search for information on the web"
        ),
        Tool(
            name="Calculator",
            func=mock_calculator,
            description="Perform mathematical calculations"
        )
    ]

    # Create instrumented agent
    agent = InstrumentedAgent(tools=tools, output_dir=Path("data/clean_traces/test"))

    # Execute test task
    trace = agent.execute("What is 2 + 2?")

    print(f"✅ Trace saved: {trace.trace_id}")
    print(f"Steps: {len(trace.steps)}")
    print(f"Success: {trace.success}")
