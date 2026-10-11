# Agent loop behavior and acceptance checks

The agent-loop exercises are complete. `codepilot/agent/core.py` implements
decision budgeting, observations, tool dispatch, batch validation, sequential
and parallel scheduling, termination, and trajectory events. State changes use
`Command.update`; the benchmark evaluator remains outside the agent core.

`max_model_calls` is the model decision allowance, while `tool_call_iterations`
counts attempted repository operations without imposing a hard limit. Follow
`docs/experiment-protocol.md` and the relevant ADRs when changing these rules.
Code comments and model-facing messages are in English. Paired analysis of saved
development runs is described in `development-analysis-exercises.md`.

Helper contracts are deliberately small: `observation` and `invoke_tool` return
a `ToolMessage`; both execution helpers return `(observations, attempted_count)`.
You may reorganize the helpers if the observable behavior remains the same.

## Trajectory logging

`tools_node` records every requested invocation as `tool_call` before validation
or execution, with `tool_call_id`, `name`, and `args`. Valid batches use request
order; JSON parsing failures use valid-list order followed by invalid-list order. Rejected
and subsequently skipped requests are included. `observation` records
`tool_result` with the same ID and name, content, and a status of `success`,
`error`, or `skipped`. Parallel result events reflect completion order; the
messages returned to the model retain request order.

After the graph returns normally, `run` records one `termination` event with
`termination_reason`, `model_iterations`, `tool_call_iterations`, and
`elapsed_seconds`. Elapsed time measures graph execution using a monotonic
clock and excludes sandbox setup and patch collection. Model and infrastructure
exceptions propagate to the runner and do not produce a normal termination
event. Model inputs, responses, and errors are recorded by `LoggedModel`; patch
collection and evaluation results remain the responsibility of the runner and
evaluation interface. Logging does not add fields to `AgentState`.

## Tool observations and batch rejection

`call` is one dictionary from the model response's `tool_calls` or
`invalid_tool_calls`. Its `id`, `name`, and `args` identify the requested
invocation; invalid calls retain an unparsed argument string and a parsing error.
`observation` creates and
returns a new `ToolMessage` and records its result; it does not mutate the state.
The node collects these messages and returns them in `Command.update["messages"]`,
where the state's `add_messages` reducer merges them into the trajectory.

The loop supplies the observation flags and content:

| Situation | Content comes from | Flags |
| --- | --- | --- |
| Successful tool execution | The return value of `tool.invoke(call["args"])` | Both false |
| Tool execution or argument failure | An explanation constructed by `invoke_tool` from the caught error | `failed=True` |
| Any JSON parsing failure in the batch | A batch-rejection explanation constructed by `tools_node` | `failed=True`, `skipped=False` for every valid and invalid call |
| Invalid mixed finish batch | A batch-rejection explanation constructed by `tools_node` | `failed=True`, `skipped=False` for every call |
| Remaining calls after a sequential failure | A cancellation explanation constructed by `execute_sequential` | `skipped=True` |

`failed` describes a request error, so it also applies when validation prevents
execution. It does not establish that the tool was attempted. `skipped` describes
a later call canceled after an earlier sequential tool failed. Both produce
`ToolMessage.status="error"`; the trajectory log distinguishes errors and skips.

For `[write(id=a), finish(id=b), read(id=c)]`, reject all three before running
any tool. Generate the explanation inside `tools_node`, for example:

```text
Batch rejected: finish must be the only tool call in a response. No tools were executed. Retry with finish alone or a batch without finish.
```

Pass this explanation to `observation` for each call, with `failed=True` and
`skipped=False`. Return three messages in the order `a`, `b`, `c`; each has its
own tool call ID and name, with error status. Finish is rejected too, so the
batch cannot declare voluntary completion.

If `AIMessage.invalid_tool_calls` is nonempty, reject the entire batch before
checking finish or scheduling. Every ID in `tool_calls` and `invalid_tool_calls`
receives an error observation, including otherwise valid requests. Execute no
tools, count no attempted repository operations, and return to `agent_node`.
Malformed finish calls are parsing failures and cannot declare completion.
Results follow valid-list order and then invalid-list order; the parsed lists
do not retain their original interleaving. Only a response with both lists empty
ends with `no_tool_call`.

Return those messages through `Command(goto="agent_node", update={"messages":
observations})`. Do not write a rejection explanation into `termination_reason`,
set a summary, or increment attempted tool statistics. The model decision was
already counted in `agent_node`; that node checks the budget before another
model invocation.

## Budget acceptance checks


- With one model decision available, a response requesting two valid tools
  processes both under normal scheduling, records one decision, and permits no
  second model invocation.
- Rejected batches consume decisions. Repeated mixed finish batches therefore
  stop at the configured decision limit without modifying the repository.
- A batch containing any JSON parsing failure returns errors for every ID and
  attempts no tools. At the last decision it ends with `model_call_limit`; with
  remaining decisions, the model can correct the request.
- A successful sole finish on the last decision still records voluntary
  completion. A plain-text response records no tool call and ends the run.
- A write, edit, and finish trajectory requires three model decisions when
  each operation arrives in a separate response.
- A final sequential batch still skips calls after a tool failure, and a final
  all-read batch still waits for all reads. A decision limit does not override
  those scheduling rules.
- Budget exhaustion returns a final state so the runner can collect and
  independently evaluate the current patch.

The acceptance tests in `tests/test_tool_scheduling.py` use the decision budget.
For a focused scheduling check:

```bash
python -m pytest tests/test_tool_scheduling.py -k sequential_failure -q
```

For the agent integration and logging checks:

```bash
python -m pytest tests/test_tool_scheduling.py -q
python -m pytest tests/test_agent_logging.py -q
python -m pytest tests -q
```

Also inspect a real trajectory after a small development run. Confirm it contains
the model inputs/outputs, every requested tool call and result, termination reason,
elapsed time, and final patch. The existing tests do not cover every logging field
or real model-service error formats, so passing tests alone does not replace
integration checks.

Only then use `scripts.run_dev` to compare the two configurations. See
`docs/development-evaluation.md` for commands and output locations.
