# Notebook Agent CLI

A chat-style CLI that can generate, edit, run, and log Jupyter notebooks with LLM assistance. Supports Mistral (official SDK), OpenAI-compatible endpoints, and local servers (Ollama / LM Studio). Logs and executed notebooks are kept in `.notebook_agent/` in the working directory.

## Quickstart

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set your provider (defaults to Mistral):

```bash
# Mistral (default)
export MISTRAL_API_KEY=...
# Optional overrides
export AGENT_PROVIDER=mistral
export AGENT_MODEL=mistral-large-latest
export AGENT_BASE_URL=https://api.mistral.ai/v1

# OpenAI-compatible
export AGENT_PROVIDER=openai
export OPENAI_API_KEY=...
export AGENT_MODEL=gpt-4o-mini

# Local (no key needed)
export AGENT_PROVIDER=ollama   # or lmstudio
export AGENT_MODEL=llama3.2:latest  # or your local model name
```

Run the CLI in the directory where you want notebooks created:

```bash
.venv/bin/python agent.py chat
```

## Slash commands

- `/help` show commands
- `/mode [author|autonomy]` interaction style
- `/new <prompt>` generate a notebook
- `/auto <experiment>` generate + execute
- `/run <path>` execute an existing notebook
- `/read <path>` summarize notebook
- `/edit <path> <instruction>` edit via LLM (backup saved)
- `/logs` list run logs
- `/log <path>` show log
- `/tail <path>` tail last lines of a log
- `/list` list notebooks in cwd
- `/timeout [secs]` view/set default run timeout
- `/last` show last run summary
- `/env` show provider/model/base URL (keys masked)
- `/provider <name>` switch LLM provider (mistral|ollama|lmstudio|openai)
- `/model <name>` set model id
- `/baseurl <url>` set API base URL
- `/setkey` prompt for API key and save to `.env`
- `/sysinfo` system snapshot
- `/exit` quit

Tab completion works for slash commands if `prompt_toolkit` is available (installed via requirements).

## Behavior notes

- Executions use `nbclient` + `ipykernel` with the notebook’s directory as working dir.
- Shell/pip lines in notebooks trigger a warning before run.
- After each run, the agent posts a short suggestion based on success/failure and output tail.
- Artifacts: `.notebook_agent/runs/` (executed notebooks), `.notebook_agent/logs/` (stdout/error logs), optional `.env` for stored keys.

## One-off commands

```bash
.venv/bin/python agent.py sysinfo            # show RAM/disk/GPU
.venv/bin/python agent.py run nb.ipynb       # execute once (non-chat)
.venv/bin/python agent.py new "my prompt"    # generate notebook once
```

## Troubleshooting

- Missing key: use `/setkey` in chat or set env vars above; keys are masked in `/env`.
- Kernel not found: `ipykernel` is installed via requirements; ensure the venv python matches your notebooks.
- Local providers: ensure Ollama/LM Studio HTTP server is running (defaults: 11434 or 1234). Update `AGENT_BASE_URL` if different.
