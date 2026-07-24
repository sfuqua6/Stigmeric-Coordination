# Using the swarm from Odysseus (or any MCP client)

`mcp_server.py` (repo root) exposes the pipeline as MCP tools:

| Tool | What it does |
|---|---|
| `run_swarm(prompt, task_type, minutes_budget, baseline)` | Full pipeline run via `run_swarm.py` subprocess; returns the clean `answer.txt` + a `run_id` footer |
| `get_swarm_diagnostics(run_id)` | `diagnostics.md` + summary.json health numbers for a past run |
| `list_swarm_runs(limit)` | Recent runs, newest first |

## One-time setup

1. **Start the MCP server on the host** (not inside Docker — the swarm needs
   this machine's Python env, GPU, and API keys):

   ```powershell
   # from the repo root; set GROQ_API_KEY first if you want the Groq backend
   python mcp_server.py            # serves http://0.0.0.0:8756/mcp
   ```

2. **Register it in Odysseus** (Settings → MCP management, admin-gated):
   add a remote/HTTP MCP server with URL

   ```
   http://host.docker.internal:8756/mcp
   ```

   (`host.docker.internal` is how the Odysseus container reaches services on
   your host — same pattern its docs use for host Ollama.)

3. In a chat, enable tools/agent mode and ask something like *"use the swarm
   to analyze whether remote work helps innovation"*. The agent decides when
   to call `run_swarm`; it does **not** run on every message.

## Things to know

- **It is not automatic.** Odysseus won't launch the swarm on open; the MCP
  server must be running on the host, and the agent calls the tool when the
  conversation warrants it (or when you explicitly ask).
- **A call takes minutes.** `minutes_budget` (default 8) sets
  `SWARM_MAX_TIME_S` for the subprocess; synthesis runs after that budget.
  The server hard-kills a run 10 min past the budget.
- **VRAM contention.** If Odysseus's Ollama and the swarm's local model share
  one GPU, they will fight. Recommended: run the swarm on Groq
  (`GROQ_API_KEY` in the environment you start `mcp_server.py` from) so the
  GPU stays with Odysseus.
- **Mock mode is labeled.** With `MOCK_LLM=1` the answer footer says
  `MOCK RUN — plumbing only`; never treat those as real answers (P0.1).
- `--stdio` serves stdio instead of HTTP, for local clients (Claude Code /
  Claude Desktop). `--host` / `--port` override the HTTP bind.
- Runs land in `outputs/mcp_<task>_<timestamp>/` (or `outputs_mock/`), same
  artifact set as CLI runs, plus `mcp_launch.log` (subprocess stdout/stderr).

Dependency: `pip install mcp` (the official Python SDK; pulls fastapi-free
`uvicorn`/`sse-starlette` extras it needs).
