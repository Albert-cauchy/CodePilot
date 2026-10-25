from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from codepilot.agent.budget import DecisionBudget
from codepilot.agent.prompts import build_system_prompt
from codepilot.agent.state import AgentState
from codepilot.sandbox.base import BaseSandbox
from codepilot.trajectory import TrajectoryLogger
from codepilot.tools.delegate import create_delegate_tool
from codepilot.tools.edit import create_edit_tool
from codepilot.tools.finish import finish
from codepilot.tools.read import create_read_tool
from codepilot.tools.shell import create_shell_tool
from codepilot.tools.think import create_think_tool
from codepilot.tools.write import create_write_tool


class AgentCore:
    """Run a main graph or an isolated child graph under shared soft accounting."""

    def __init__(
        self,
        model,
        sandbox: BaseSandbox,
        max_model_calls: int = 500,
        *,
        logger=None,
        trajectory_dir=None,
        max_parallel_subagents: int = 4,
        budget=None,
        is_subagent: bool = False,
        agent_id: str = "main",
        run_id=None,
        parent_tool_call_id=None,
    ):
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if max_parallel_subagents < 1:
            raise ValueError("max_parallel_subagents must be positive")
        self.model = model
        self.sandbox = sandbox
        self.budget = budget or DecisionBudget(max_model_calls)
        self.is_subagent = is_subagent
        self.agent_id = agent_id
        self.run_id = run_id or uuid4().hex
        self.parent_tool_call_id = parent_tool_call_id
        self.max_parallel_subagents = max_parallel_subagents
        self.trajectory_dir = (
            Path(trajectory_dir)
            if trajectory_dir is not None
            else (
                Path(logger.path).parent
                if logger is not None and hasattr(logger, "path")
                else Path("trajectory") / self.run_id
            )
        )
        self.logger = logger or TrajectoryLogger(self.trajectory_dir / "events.jsonl")
        if hasattr(self.logger, "context"):
            self.logger.context.update(
                run_id=self.run_id,
                agent_id=self.agent_id,
                parent_tool_call_id=self.parent_tool_call_id,
            )
        self.verification: list[dict[str, Any]] = []
        self.model_step = 0
        self.normal_responses = 0
        self.shell = create_shell_tool(sandbox)
        self.tools = [
            create_read_tool(sandbox),
            create_edit_tool(sandbox),
            create_write_tool(sandbox),
            self.shell,
            finish,
            create_think_tool(self.trajectory_dir),
        ]
        if not is_subagent:
            self.tools.append(create_delegate_tool(self.delegate_batch))
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.max_model_calls = max_model_calls
        self.system_prompt = build_system_prompt(self.tools, is_subagent=is_subagent)
        self.model_with_tools = model.bind_tools(self.tools, parallel_tool_calls=True)
        self.graph = self.build_graph()

    def log(self, event_type, **payload):
        if self.logger is not None:
            self.logger.log(
                event_type,
                run_id=self.run_id,
                agent_id=self.agent_id,
                parent_tool_call_id=self.parent_tool_call_id,
                **payload,
            )

    def invoke_model(self, messages, *, final=False):
        self.model_step += 1
        role = "final_analysis" if final else ("subagent" if self.is_subagent else "main")
        self.log(
            "model_input",
            role=role,
            step=self.model_step,
            messages=[message_to_dict(message) for message in messages],
        )
        try:
            response = (self.model if final else self.model_with_tools).invoke(messages)
        except Exception as exc:
            self.log(
                "model_error",
                role=role,
                step=self.model_step,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        total = self.budget.record(role)
        self.normal_responses += 1
        self.log(
            "model_response",
            role=role,
            step=self.model_step,
            message=message_to_dict(response),
            total_model_calls=total,
        )
        return response

    def agent_node(self, state: AgentState):
        # Active child graphs ignore the shared threshold. Only the main graph
        # transitions to one tool-free final analysis before its next decision.
        if not self.is_subagent and self.budget.exhausted:
            return Command(goto="final_analysis")

        try:
            response = self.invoke_model(
                [SystemMessage(content=self.system_prompt), *state["messages"]]
            )
        except Exception as exc:  # Classify context limits; propagate service errors.
            if getattr(exc, "code", None) == "context_length_exceeded":
                return Command(
                    goto=END,
                    update={"termination_reason": "context_limit"},
                )
            raise

        # Every normally returned response consumes one decision, including
        # plain text, rejected batches, and finish. Preserve the full history.
        return Command(
            goto="tools_node",
            update={
                "messages": [response],
                "model_iterations": state["model_iterations"] + 1,
            },
        )

    def final_analysis_node(self, state: AgentState):
        messages = [
            SystemMessage(
                content=(
                    "The shared soft decision budget has been reached. All delegated work "
                    "has returned and valid changes have been applied. Analyze the results "
                    "once: summarize changes, actual validation and unresolved issues. "
                    "No tools are available; this response ends the run regardless of completion."
                )
            ),
            *state["messages"],
        ]
        try:
            response = self.invoke_model(messages, final=True)
        except Exception as exc:
            if getattr(exc, "code", None) == "context_length_exceeded":
                return Command(goto=END, update={"termination_reason": "context_limit"})
            raise
        return Command(
            goto=END,
            update={
                "messages": [response],
                "summary": response.content,
                "model_iterations": state["model_iterations"] + 1,
                "termination_reason": "model_call_limit",
            },
        )

    def delegate_batch(self, tasks, tool_call_id):
        from .delegation import run_delegation_batch

        return run_delegation_batch(self, tasks, tool_call_id)

    def observation(self, call, content, *, failed=False, skipped=False):
        # Return a message for this call ID; the node publishes it through
        # Command.update. The caller supplies the result or error explanation.
        # Request errors include rejection before execution. Skips are later
        # calls canceled after an earlier sequential failure.
        tool_call_id = call["id"]
        name = call["name"]
        if skipped:
            log_status = "skipped"
            content = f"Skipped: not executed. {content}"
        elif failed:
            log_status = "error"
        else:
            log_status = "success"

        message_status = "error" if (failed or skipped) else "success"

        tool_result = ToolMessage(
            tool_call_id=tool_call_id, name=name, content=content, status=message_status
        )

        self.log(
            "tool_result",
            tool_call_id=tool_call_id,
            name=name,
            content=content,
            status=log_status,
        )
        return tool_result

    def invoke_tool(self, call):
        # Convert tool failures into observations so the model can recover.
        # A nonzero shell exit code remains a completed command result;
        # the shell tool raises when it cannot execute or times out.
        name = call["name"]
        args = call["args"]
        if name not in self.tools_by_name:
            return self.observation(call=call, content=f"Error:unknown tool '{name}'", failed=True)
        tool = self.tools_by_name[name]
        try:
            result = tool.invoke({**call, "type": "tool_call"})
            if isinstance(result, ToolMessage):
                result = result.content
        except (TypeError, ValueError, KeyError, ValidationError) as e:
            return self.observation(
                call, f"Error: invalid arguments for '{name}': {e}", failed=True
            )
        except TimeoutError as e:
            return self.observation(call, f"Error: tool '{name}' timed out: {e}", failed=True)
        except Exception as e:  # noqa: BLE001 - Report tool execution failures to the model.
            return self.observation(call, f"Error: tool '{name}' failed: {e}", failed=True)

        if name == "shell":
            self.verification.append({"command": args.get("command"), "result": result})
        return self.observation(call, result, failed=False)

    def execute_sequential(self, calls):
        # Preserve request order and stop execution after the first failure.
        # Remaining calls still receive messages, but do not count as attempts.
        skip_flag = False
        observations = []
        attempted_count = 0
        for call in calls:
            if skip_flag:
                result = self.observation(
                    call, "An earlier tool call in this batch failed.", skipped=True
                )
            else:
                attempted_count += 1
                result = self.invoke_tool(call)
                if result.status == "error":
                    skip_flag = True
            observations.append(result)

        return (observations, attempted_count)

    def execute_parallel(self, calls):
        # tools_node admits only batches whose tools all allow parallel execution.
        # Wait for every read; executor.map preserves the original request order.
        if not calls:
            return [], 0

        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            observations = list(executor.map(self.invoke_tool, calls))

        return observations, len(calls)

    def tools_node(self, state: AgentState):
        response = cast(AIMessage, state["messages"][-1])
        calls = response.tool_calls
        invalid_calls = response.invalid_tool_calls
        all_calls = [*calls, *invalid_calls]
        # Record all requests before validation or execution, including rejected
        # batches and calls that a sequential failure may later skip.
        for call in all_calls:
            self.log(
                "tool_call",
                tool_call_id=call["id"],
                name=call["name"],
                args=call["args"],
            )

        # A JSON parsing failure rejects every request before finish validation
        # or scheduling. Parsed calls come first, followed by invalid calls;
        # their original interleaving is unavailable in these separate lists.
        if invalid_calls:
            content = (
                "Batch rejected: one or more tool calls contain invalid JSON "
                "arguments. No tools were executed. Retry with valid JSON."
            )
            observations = [
                self.observation(
                    call,
                    content,
                    failed=True,
                    skipped=False,
                )
                for call in all_calls
            ]
            return Command(
                goto="agent_node",
                update={"messages": observations},
            )

        # Plain text ends the run without declaring voluntary completion.
        if not calls:
            return Command(
                goto=END,
                update={
                    "termination_reason": "no_tool_call",
                    "summary": state["messages"][-1].content,
                },
            )

        if len(calls) > 1 and any(
            call["name"] in {"finish", "think", "delegate"} for call in calls
        ):
            # Reject the entire batch before execution. Each call receives an
            # error rather than a sequential skip; no tool attempts are counted.
            content = "Batch rejected: finish, think and delegate must each be called alone. No tools were executed."
            observations = [
                self.observation(
                    call,
                    content,
                    failed=True,
                    skipped=False,
                )
                for call in calls
            ]

            return Command(
                goto="agent_node",
                update={"messages": observations},
            )

        if len(calls) == 1 and calls[0]["name"] == "finish":
            # Only a successful sole finish declares voluntary completion.
            # Finish is not a repository operation and does not enter its count.
            result = self.invoke_tool(calls[0])
            update: dict[str, Any] = {"messages": [result]}

            if result.status == "success":
                update.update(
                    summary=result.content,
                    unresolved_issues=calls[0]["args"].get("unresolved_issues") or [],
                    termination_reason="voluntary_completion",
                )
                return Command(goto=END, update=update)

            return Command(goto="agent_node", update=update)

        # Scheduling follows registered metadata, never model-supplied arguments.
        # Unknown tools use sequential execution and receive error observations.
        parallel_allowed = all(
            call["name"] in self.tools_by_name
            and (self.tools_by_name[call["name"]].metadata or {}).get("executionMode") == "parallel"
            for call in calls
        )
        if parallel_allowed:
            observations, attempted_count = self.execute_parallel(calls)
        else:
            observations, attempted_count = self.execute_sequential(calls)

        return Command(
            goto="agent_node",
            update={
                "messages": observations,
                "tool_call_iterations": state["tool_call_iterations"] + attempted_count,
            },
        )

    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("agent_node", self.agent_node)
        builder.add_node("tools_node", self.tools_node)
        builder.add_node("final_analysis", self.final_analysis_node)
        builder.add_edge(START, "agent_node")
        return builder.compile()

    def run(self, problem_statement: str):
        # Measure agent graph execution; sandbox setup and patch collection are
        # outside this interval. Each graph records its actual model interactions.
        initial_state: AgentState = {
            "messages": [HumanMessage(content=problem_statement)],
            "tool_call_iterations": 0,
            "model_iterations": 0,
        }
        config = {"recursion_limit": 2**31 - 1}
        # Child graphs have no local decision cap; do not impose one indirectly
        # through LangGraph's default recursion limit.
        started = perf_counter()
        final_state = self.graph.invoke(initial_state, config=config)
        counts = self.budget.snapshot()
        final_state["own_model_iterations"] = final_state["model_iterations"]
        final_state["decision_counts"] = counts
        final_state["final_analysis_calls"] = counts["final_analysis"]
        if not self.is_subagent:
            final_state["model_iterations"] = counts["total"]
        self.log(
            "termination",
            termination_reason=final_state["termination_reason"],
            model_iterations=final_state["model_iterations"],
            tool_call_iterations=final_state["tool_call_iterations"],
            elapsed_seconds=perf_counter() - started,
            decision_counts=counts,
            model_call_threshold=self.max_model_calls,
        )
        return final_state
