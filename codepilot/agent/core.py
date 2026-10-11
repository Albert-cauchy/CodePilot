from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from codepilot.agent.prompts import build_system_prompt
from codepilot.agent.state import AgentState
from codepilot.sandbox.base import BaseSandbox
from tools.edit import create_edit_tool
from tools.finish import finish
from tools.read import create_read_tool
from tools.shell import create_shell_tool
from tools.write import create_write_tool

TOOL_CONFIGURATIONS = ("two", "five")


class AgentCore:
    """Run model decisions and tool batches under a fixed decision budget."""

    def __init__(
        self,
        model,
        sandbox: BaseSandbox,
        max_model_calls: int = 500,
        *,
        tool_configuration: str = "two",
        logger=None,
    ):
        if tool_configuration not in TOOL_CONFIGURATIONS:
            raise ValueError(f"Unknown tool configuration: {tool_configuration}")
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        self.shell = create_shell_tool(sandbox)
        self.tools = [self.shell, finish]
        if tool_configuration == "five":
            self.tools = [
                create_read_tool(sandbox),
                create_edit_tool(sandbox),
                create_write_tool(sandbox),
                self.shell,
                finish,
            ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.max_model_calls = max_model_calls
        self.logger = logger
        self.system_prompt = build_system_prompt(self.tools)
        self.model_with_tools = model.bind_tools(self.tools, parallel_tool_calls=True)
        self.graph = self.build_graph()

    def log(self, event_type, **payload):
        if self.logger is not None:
            self.logger.log(event_type, **payload)

    def agent_node(self, state: AgentState):
        # Check the budget before invocation so the final allowed response still
        # reaches tools_node and receives normal batch processing.
        if state["model_iterations"] >= self.max_model_calls:
            return Command(goto=END, update={"termination_reason": "model_call_limit"})

        try:
            response = self.model_with_tools.invoke(
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
            return self.observation(
                call=call, content=f"Error:unknown tool '{name}'", failed=True
            )
        tool = self.tools_by_name[name]
        try:
            result = tool.invoke(args)
        except (TypeError, ValueError, KeyError, ValidationError) as e:
            return self.observation(
                call, f"Error: invalid arguments for '{name}': {e}", failed=True
            )
        except TimeoutError as e:
            return self.observation(
                call, f"Error: tool '{name}' timed out: {e}", failed=True
            )
        except Exception as e:  # noqa: BLE001 - Report tool execution failures to the model.
            return self.observation(
                call, f"Error: tool '{name}' failed: {e}", failed=True
            )

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
        calls = state["messages"][-1].tool_calls
        invalid_calls = state["messages"][-1].invalid_tool_calls
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
                update={"termination_reason": "no_tool_call"},
            )

        if len(calls) > 1 and any(call["name"] == "finish" for call in calls):
            # Reject the entire batch before execution. Each call receives an
            # error rather than a sequential skip; no tool attempts are counted.
            content = (
                "Batch rejected: finish must be called alone. No tools were executed."
            )
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
            update = {"messages": [result]}

            if result.status == "success":
                update.update(
                    summary=result.content,
                    termination_reason="voluntary_completion",
                )
                return Command(goto=END, update=update)

            return Command(goto="agent_node", update=update)

        # Scheduling follows registered metadata, never model-supplied arguments.
        # Unknown tools use sequential execution and receive error observations.
        parallel_allowed = all(
            call["name"] in self.tools_by_name
            and (self.tools_by_name[call["name"]].metadata or {}).get("executionMode")
            == "parallel"
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
        builder.add_edge(START, "agent_node")
        return builder.compile()

    def run(self, problem_statement: str):
        # Measure agent graph execution; sandbox setup and patch collection are
        # outside this interval. LoggedModel records model inputs and outputs.
        initial_state: AgentState = {
            "messages": [HumanMessage(content=problem_statement)],
            "tool_call_iterations": 0,
            "model_iterations": 0,
        }
        config = {"recursion_limit": 2 * self.max_model_calls + 2}
        # Allow both nodes per decision plus the final budget check and graph exit.
        started = perf_counter()
        final_state = self.graph.invoke(initial_state, config=config)
        self.log(
            "termination",
            termination_reason=final_state["termination_reason"],
            model_iterations=final_state["model_iterations"],
            tool_call_iterations=final_state["tool_call_iterations"],
            elapsed_seconds=perf_counter() - started,
        )
        return final_state
