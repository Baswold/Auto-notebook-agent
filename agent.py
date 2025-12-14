import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
import psutil
import typer
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from mistralai import Mistral
except Exception:
    Mistral = None  # type: ignore

try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.completion import WordCompleter
except Exception:
    pt_prompt = None  # type: ignore
    WordCompleter = None  # type: ignore

console = Console()
app = typer.Typer(help="Notebook agent CLI with LLM-assisted authoring and execution.")
ENV_FILE = Path(".env")
BANNER = r"""
┌───────────────────────────────────────────────────────────────┐
│   JUPYTER NOTEBOOK AGENT — build/run/edit/log notebooks fast   │
└───────────────────────────────────────────────────────────────┘
"""

# Constants
MAX_CELL_PREVIEW = 12
MAX_TEXT_LENGTH = 240
OUTPUT_PREVIEW_LINES = 6
CHAT_HISTORY_LIMIT = 10
LOG_DISPLAY_LIMIT = 20
TAIL_LINES = 40
DEFAULT_TIMEOUT = 600
GPU_DETECT_TIMEOUT = 3


@dataclass
class AgentSettings:
    """Configuration settings for the LLM agent.

    Loads settings from environment variables with sensible defaults.
    Supports multiple providers: mistral, openai, ollama, lmstudio.
    """
    provider: str = field(default_factory=lambda: os.getenv("AGENT_PROVIDER", "mistral"))
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", os.getenv("MISTRAL_MODEL", "mistral-large-latest")))
    api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    base_url: str = field(default_factory=lambda: os.getenv("AGENT_BASE_URL", os.getenv("MISTRAL_BASE_URL", "")))
    temperature: float = 0.1

    def __post_init__(self) -> None:
        # normalize provider and fill sensible defaults
        self.provider = self.provider.lower()
        if not self.base_url:
            if self.provider == "mistral":
                self.base_url = "https://api.mistral.ai/v1"
            elif self.provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"
        # fallback model hints per provider if user did not override
        if not self.model:
            if self.provider in {"ollama", "lmstudio"}:
                self.model = "qwen2.5-coder:7b"  # adjust to your local model name
            elif self.provider == "openai":
                self.model = "gpt-4o-mini"
            else:
                self.model = "mistral-large-latest"


class LLMClient:
    """Client wrapper for various LLM providers.

    Supports Mistral, OpenAI, Ollama, and LM Studio providers.
    Handles API initialization and chat completions.
    """
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        provider = settings.provider
        self.provider = provider

        if provider == "mistral":
            if Mistral is None:
                raise RuntimeError(
                    "mistralai package is missing. Install dependencies with: "
                    "pip install -r requirements.txt"
                )
            if not settings.api_key:
                raise RuntimeError(
                    "MISTRAL_API_KEY is required for Mistral provider. "
                    "Set it via environment variable or use /setkey command."
                )
            self.client = Mistral(api_key=settings.api_key, server_url=settings.base_url)
        else:
            if OpenAI is None:
                raise RuntimeError(
                    "openai package is missing. Install dependencies with: "
                    "pip install -r requirements.txt"
                )
            if provider == "openai" and not settings.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required for OpenAI provider. "
                    "Set it via environment variable or use /setkey command."
                )
            # local providers may not require a key; supply a dummy
            api_key = settings.api_key or "not-needed"
            self.client = OpenAI(api_key=api_key, base_url=settings.base_url)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Send a chat completion request to the LLM provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional provider-specific parameters

        Returns:
            The assistant's response content as a string
        """
        if self.provider == "mistral":
            response = self.client.chat.complete(  # type: ignore[attr-defined]
                model=self.settings.model,
                messages=messages,
                temperature=self.settings.temperature,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        else:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=self.settings.temperature,
                **kwargs,
            )
            return response.choices[0].message.content or ""


class NotebookManager:
    """Manages notebook execution, logging, and file operations.

    Handles notebook reading, execution via nbclient, and log management.
    Stores execution artifacts in .notebook_agent directory.
    """
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.state_dir = self.workdir / ".notebook_agent"
        self.run_log_dir = self.state_dir / "runs"
        self.raw_log_dir = self.state_dir / "logs"
        self.state_dir.mkdir(exist_ok=True)
        self.run_log_dir.mkdir(parents=True, exist_ok=True)
        self.raw_log_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    def read(self, notebook_path: Path) -> str:
        """Read and summarize a notebook's structure and content."""
        nb = nbformat.read(notebook_path, as_version=4)
        summary_lines = [f"# Cells: {len(nb.cells)}", ""]
        for idx, cell in enumerate(nb.cells[:MAX_CELL_PREVIEW]):
            label = f"{idx+1:02d} {cell.cell_type.upper()}"
            body = cell.source.strip()
            if len(body) > MAX_TEXT_LENGTH:
                body = body[:MAX_TEXT_LENGTH] + "... [truncated]"
            summary_lines.append(f"{label}\n{body}\n")
        if len(nb.cells) > MAX_CELL_PREVIEW:
            summary_lines.append(f"... (+{len(nb.cells) - MAX_CELL_PREVIEW} more cells)")
        return "\n".join(summary_lines)

    def run(self, notebook_path: Path, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a Jupyter notebook and capture outputs and logs.

        Args:
            notebook_path: Path to the notebook file to execute
            timeout: Optional timeout in seconds (defaults to DEFAULT_TIMEOUT)

        Returns:
            Dict containing run_id, status, log_path, executed_path, error, and output_preview
        """
        nb = nbformat.read(notebook_path, as_version=4)
        run_id = f"{self._timestamp()}_{notebook_path.stem}"
        run_log = self.raw_log_dir / f"{run_id}.log"
        executed_path = self.run_log_dir / f"{run_id}.ipynb"

        console.print(f"[bold green]Executing[/bold green] {notebook_path} (run id {run_id})")
        stdout_lines: List[str] = []
        try:
            kernel_name = nb.metadata.get("kernelspec", {}).get("name") or "python3"
            client = NotebookClient(
                nb,
                timeout=timeout or DEFAULT_TIMEOUT,
                kernel_name=kernel_name,
                resources={"metadata": {"path": str(notebook_path.parent)}},
            )
            client.execute()
            nbformat.write(client.nb, executed_path)
            # collect outputs for a quick summary
            for cell in client.nb.cells:
                if cell.cell_type != "code":
                    continue
                for output in cell.get("outputs", []):
                    text = ""
                    if "text" in output:
                        text = output["text"]
                    elif "data" in output and "text/plain" in output["data"]:
                        text = output["data"]["text/plain"]
                    if text:
                        stdout_lines.append(text if isinstance(text, str) else "\n".join(text))
            status = "success"
            error = ""
        except CellExecutionError as err:
            nbformat.write(nb, executed_path)
            status = "failed"
            error = f"Cell execution failed: {err}"
            stdout_lines.append(error)
        except (OSError, IOError) as err:
            nbformat.write(nb, executed_path)
            status = "failed"
            error = f"File I/O error: {err}"
            stdout_lines.append(error)
        except Exception as err:  # noqa: BLE001
            nbformat.write(nb, executed_path)
            status = "failed"
            error = f"Unexpected error: {err}"
            stdout_lines.append(error)

        run_log.write_text("\n".join(stdout_lines), encoding="utf-8")
        return {
            "run_id": run_id,
            "status": status,
            "log_path": run_log,
            "executed_path": executed_path,
            "error": error,
            "output_preview": stdout_lines[-OUTPUT_PREVIEW_LINES:],
        }

    def list_logs(self) -> List[Path]:
        return sorted(self.raw_log_dir.glob("*.log"))

    def read_log(self, log_path: Path) -> str:
        return log_path.read_text(encoding="utf-8")

    def write_notebook(self, content: nbformat.NotebookNode, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(content, target)


class NotebookGenerator:
    """Generates and edits Jupyter notebooks using LLM assistance.

    Uses prompt-based generation to create or modify notebook content.
    Handles JSON extraction from LLM responses and provides fallback handling.
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON content from LLM response.

        Tries to find JSON in markdown code fences first, then validates raw text.
        """
        fenced = re.findall(r"```json(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced[0]
        try:
            json.loads(text)
            return text
        except (json.JSONDecodeError, ValueError):
            return None

    def generate(self, prompt: str) -> nbformat.NotebookNode:
        """Generate a new notebook from a text prompt.

        Args:
            prompt: Description of the notebook to generate

        Returns:
            A NotebookNode object representing the generated notebook
        """
        system = (
            "You are an autonomous notebook builder. "
            "Return a valid Jupyter nbformat v4 JSON. "
            "Keep outputs empty. Use Python. Include a concise markdown header and code cells. "
            "Do not add explanations outside JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages)
        extracted = self._extract_json(raw) or raw
        try:
            return nbformat.reads(extracted, as_version=4)
        except (json.JSONDecodeError, nbformat.ValidationError) as err:
            # fallback: simple notebook with content as markdown
            console.print(f"[yellow]Warning: Could not parse LLM response as notebook ({err}). Using fallback.[/yellow]")
            nb = nbformat.v4.new_notebook()
            nb.cells.append(nbformat.v4.new_markdown_cell(f"Autogenerated notebook placeholder.\n\n{raw}"))
            return nb

    def edit(self, notebook_text: str, instruction: str) -> nbformat.NotebookNode:
        """Edit an existing notebook based on an instruction.

        Args:
            notebook_text: JSON string of the notebook to edit
            instruction: Description of changes to make

        Returns:
            Updated NotebookNode object
        """
        system = (
            "You are a notebook editor. Update the provided Jupyter notebook JSON according to the instruction. "
            "Return the full updated nbformat v4 JSON. Keep outputs empty."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Instruction: {instruction}\n\nNotebook JSON:\n{notebook_text}"},
        ]
        raw = self.llm.chat(messages)
        extracted = self._extract_json(raw) or raw
        return nbformat.reads(extracted, as_version=4)


def format_sysinfo() -> str:
    info: List[str] = []
    info.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    info.append(f"Python: {sys.version.split()[0]}")
    try:
        vm = psutil.virtual_memory()
        info.append(f"RAM: {vm.total/1e9:0.1f} GB total, {vm.available/1e9:0.1f} GB available")
    except Exception:
        info.append("RAM: unavailable (psutil error)")
    try:
        disk = shutil.disk_usage(".")
        info.append(f"Disk: {disk.free/1e9:0.1f} GB free of {disk.total/1e9:0.1f} GB")
    except Exception:
        info.append("Disk: unavailable")
    gpu_info = detect_gpu()
    if gpu_info:
        info.append(f"GPU: {gpu_info}")
    else:
        info.append("GPU: none detected")
    return "\n".join(info)


def detect_gpu() -> Optional[str]:
    """Attempt to detect GPU information using system commands.

    Returns:
        GPU information string if detected, None otherwise
    """
    cmds = [
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        ["system_profiler", "SPDisplaysDataType"],
    ]
    for cmd in cmds:
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=GPU_DETECT_TIMEOUT
            )
            cleaned = " ".join(out.strip().split())
            if cleaned:
                return cleaned[:MAX_TEXT_LENGTH]
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


class ChatLoop:
    """Interactive chat interface for notebook management.

    Provides slash commands for notebook generation, execution, editing,
    and log management. Maintains conversation history for LLM context.
    """
    def __init__(self, settings: AgentSettings, llm: Optional[LLMClient], manager: NotebookManager):
        self.settings = settings
        self.llm = llm
        self.manager = manager
        self.mode = "author"  # author | autonomy
        self.history: List[Dict[str, str]] = []
        self.default_timeout = DEFAULT_TIMEOUT
        self.last_run: Optional[Dict[str, Any]] = None
        self.slash_commands = [
            "/help",
            "/mode",
            "/new",
            "/auto",
            "/run",
            "/read",
            "/edit",
            "/logs",
            "/log",
            "/tail",
            "/sysinfo",
            "/provider",
            "/model",
            "/baseurl",
            "/env",
            "/list",
            "/timeout",
            "/last",
            "/setkey",
            "/exit",
        ]

    def help_text(self) -> str:
        return (
            "Slash commands:\n"
            "  /mode [author|autonomy]   switch interaction style\n"
            "  /new <prompt>             generate a notebook from a prompt\n"
            "  /auto <experiment>        generate + execute notebook\n"
            "  /run <path>               execute an existing notebook\n"
            "  /read <path>              summarize a notebook\n"
            "  /edit <path> <instruction>edit notebook via LLM\n"
            "  /logs                     list recent run logs\n"
            "  /log <path>               read a specific log file\n"
            "  /tail <path>              print last lines of a log file\n"
            "  /list                     list notebooks in cwd\n"
            "  /timeout [seconds]        view/set default execution timeout\n"
            "  /last                     show last run summary\n"
            "  /env                      show provider/model/base URL (keys masked)\n"
            "  /sysinfo                  show system resources\n"
            "  /provider <name>          switch LLM provider (mistral|ollama|lmstudio|openai)\n"
            "  /model <name>             set LLM model id\n"
            "  /baseurl <url>            set API base URL\n"
            "  /setkey                   prompt for API key (for mistral/openai) and save to .env\n"
            "  /exit                     quit\n"
        )

    def run(self) -> None:
        console.print(f"[cyan]{BANNER}[/cyan]")
        console.print("[bold cyan]Notebook agent[/bold cyan]. Type a prompt or /help for commands.")
        while True:
            try:
                text = self._input()
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye.")
                return

            if not text:
                continue
            if text.startswith("/"):
                if text in ("/help", "/h"):
                    console.print(self.help_text())
                elif text.startswith("/mode"):
                    self._handle_mode(text)
                elif text.startswith("/new "):
                    self._handle_new(text[len("/new ") :].strip())
                elif text.startswith("/auto "):
                    self._handle_auto(text[len("/auto ") :].strip())
                elif text.startswith("/run "):
                    self._handle_run(text[len("/run ") :].strip())
                elif text.startswith("/read "):
                    self._handle_read(text[len("/read ") :].strip())
                elif text.startswith("/edit "):
                    parts = text.split(" ", 2)
                    if len(parts) < 3:
                        console.print("Usage: /edit <path> <instruction>")
                        continue
                    self._handle_edit(parts[1], parts[2])
                elif text.startswith("/logs"):
                    self._handle_logs()
                elif text.startswith("/log "):
                    self._handle_log_read(text[len("/log ") :].strip())
                elif text.startswith("/tail "):
                    self._handle_tail(text[len("/tail ") :].strip())
                elif text.startswith("/list"):
                    self._handle_list()
                elif text.startswith("/timeout"):
                    self._handle_timeout(text)
                elif text.startswith("/last"):
                    self._handle_last()
                elif text.startswith("/env"):
                    self._handle_env()
                elif text.startswith("/sysinfo"):
                    console.print(format_sysinfo())
                elif text.startswith("/provider"):
                    self._handle_provider(text)
                elif text.startswith("/model"):
                    self._handle_model(text)
                elif text.startswith("/baseurl"):
                    self._handle_baseurl(text)
                elif text.startswith("/setkey"):
                    self._handle_setkey()
                elif text.startswith("/exit"):
                    console.print("bye.")
                    return
                else:
                    console.print("Unknown command. /help to list.")
                continue

            # regular chat
            if not self.llm:
                console.print("LLM client is unavailable. Set MISTRAL_API_KEY and restart.")
                continue
            self._chat(text)

    def _handle_mode(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            console.print(f"Current mode: {self.mode}")
            return
        choice = parts[1].lower()
        if choice not in {"author", "autonomy"}:
            console.print("Mode must be author|autonomy")
            return
        self.mode = choice
        console.print(f"Mode set to {self.mode}")

    def _handle_new(self, prompt: str) -> None:
        if not self.llm:
            console.print("LLM client is unavailable. Set MISTRAL_API_KEY and restart.")
            return
        generator = NotebookGenerator(self.llm)
        nb = generator.generate(prompt)
        target = Path(f"{slugify(prompt) or 'notebook'}_{int(time.time())}.ipynb")
        self.manager.write_notebook(nb, target)
        console.print(f"Notebook written to {target}")

    def _handle_auto(self, prompt: str) -> None:
        if not self.llm:
            console.print("LLM client is unavailable. Set MISTRAL_API_KEY and restart.")
            return
        generator = NotebookGenerator(self.llm)
        nb = generator.generate(prompt)
        target = Path(f"{slugify(prompt) or 'experiment'}_{int(time.time())}.ipynb")
        self.manager.write_notebook(nb, target)
        self._warn_shell(nb=nb)
        result = self.manager.run(target, timeout=self.default_timeout)
        self._print_run_result(result)
        self._notify_agent(result, prompt)

    def _handle_run(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            console.print(f"Notebook not found: {path}")
            return
        self._warn_shell(path=path)
        result = self.manager.run(path, timeout=self.default_timeout)
        self._print_run_result(result)
        self._notify_agent(result, f"Executed notebook {path}")

    def _handle_read(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            console.print(f"Notebook not found: {path}")
            return
        console.print(self.manager.read(path))

    def _handle_edit(self, path_text: str, instruction: str) -> None:
        if not self.llm:
            console.print("LLM client is unavailable. Set MISTRAL_API_KEY and restart.")
            return
        path = Path(path_text)
        if not path.exists():
            console.print(f"Notebook not found: {path}")
            return
        raw_text = path.read_text(encoding="utf-8")
        generator = NotebookGenerator(self.llm)
        nb = generator.edit(raw_text, instruction)
        backup = path.with_suffix(".bak.ipynb")
        shutil.copy(path, backup)
        self.manager.write_notebook(nb, path)
        console.print(f"Updated {path} (backup at {backup})")

    def _handle_logs(self) -> None:
        logs = self.manager.list_logs()
        if not logs:
            console.print("No logs yet.")
            return
        table = Table("log", "modified")
        for log in logs[-LOG_DISPLAY_LIMIT:]:
            mtime = datetime.fromtimestamp(log.stat().st_mtime).isoformat(timespec="seconds")
            table.add_row(str(log), mtime)
        console.print(table)

    def _handle_log_read(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            console.print(f"Log not found: {path}")
            return
        console.print(self.manager.read_log(path))

    def _handle_tail(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            console.print(f"Log not found: {path}")
            return
        lines = self.manager.read_log(path).splitlines()
        tail = "\n".join(lines[-TAIL_LINES:])
        console.print(tail)

    def _handle_list(self) -> None:
        notebooks = sorted(Path(".").glob("*.ipynb"))
        if not notebooks:
            console.print("No notebooks found in cwd.")
            return
        table = Table("notebook", "modified")
        for nb in notebooks:
            mtime = datetime.fromtimestamp(nb.stat().st_mtime).isoformat(timespec="seconds")
            table.add_row(str(nb), mtime)
        console.print(table)

    def _handle_timeout(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            console.print(f"Default timeout: {self.default_timeout}s")
            return
        try:
            value = int(parts[1])
            if value <= 0:
                raise ValueError
            self.default_timeout = value
            console.print(f"Default timeout set to {value}s")
        except ValueError:
            console.print("Usage: /timeout <seconds>")

    def _handle_last(self) -> None:
        if not self.last_run:
            console.print("No runs yet.")
            return
        self._print_run_result(self.last_run)

    def _handle_env(self) -> None:
        masked_key = "****" if self.settings.api_key else "(none)"
        console.print(
            f"provider={self.settings.provider}, model={self.settings.model}, base_url={self.settings.base_url}, api_key={masked_key}, timeout={self.default_timeout}s"
        )

    def _chat(self, text: str) -> None:
        system = (
            "You are a concise AI coding partner inside a notebook automation CLI. "
            "Keep responses short. Do not execute code yourself; propose notebook changes when asked. "
            f"Current mode: {self.mode}."
        )
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": system}] + self.history[-CHAT_HISTORY_LIMIT:]
        reply = self.llm.chat(messages)
        self.history.append({"role": "assistant", "content": reply})
        console.print(reply)

    def _print_run_result(self, result: Dict[str, Any]) -> None:
        self.last_run = result
        console.print(f"Status: [bold]{result['status']}[/bold]")
        console.print(f"Executed notebook: {result['executed_path']}")
        console.print(f"Log: {result['log_path']}")
        if result.get("output_preview"):
            console.print("Output tail:")
            console.print("\n".join(result["output_preview"]))
        if result.get("error"):
            console.print(f"[red]Error:[/red] {result['error']}")

    def _input(self) -> str:
        if pt_prompt and WordCompleter:
            completer = WordCompleter(self.slash_commands, ignore_case=True, sentence=True)
            prefix = f"[{self.settings.provider}/{self.settings.model}]> "
            return pt_prompt(prefix, completer=completer, complete_while_typing=True)
        prefix = f"[{self.settings.provider}/{self.settings.model}]"
        return Prompt.ask(f"[white]{prefix}>[/white]")

    def _handle_provider(self, text: str) -> None:
        parts = text.split()
        if len(parts) == 1:
            console.print(f"Current provider: {self.settings.provider}")
            return
        choice = parts[1].lower()
        if choice not in {"mistral", "ollama", "lmstudio", "openai"}:
            console.print("Provider must be mistral|ollama|lmstudio|openai")
            return
        self.settings.provider = choice
        self._rebuild_llm()
        console.print(f"Provider set to {choice}. Model: {self.settings.model}, base_url: {self.settings.base_url}")

    def _handle_model(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            console.print(f"Current model: {self.settings.model}")
            return
        self.settings.model = parts[1].strip()
        self._rebuild_llm()
        console.print(f"Model set to {self.settings.model}")

    def _handle_baseurl(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            console.print(f"Current base_url: {self.settings.base_url}")
            return
        self.settings.base_url = parts[1].strip()
        self._rebuild_llm()
        console.print(f"Base URL set to {self.settings.base_url}")

    def _handle_setkey(self) -> None:
        key = Prompt.ask("Enter API key", password=True)
        if not key:
            console.print("Key unchanged.")
            return
        if self.settings.provider == "mistral":
            self.settings.api_key = key
            save_env_value("MISTRAL_API_KEY", key)
        elif self.settings.provider == "openai":
            self.settings.api_key = key
            save_env_value("OPENAI_API_KEY", key)
        else:
            self.settings.api_key = key
            save_env_value("AGENT_API_KEY", key)
        self._rebuild_llm()
        console.print("API key saved to .env and client refreshed.")

    def _rebuild_llm(self) -> None:
        try:
            self.llm = LLMClient(self.settings)
        except Exception as err:  # noqa: BLE001
            console.print(f"[red]LLM unavailable:[/red] {err}")
            self.llm = None

    def _warn_shell(self, path: Optional[Path] = None, nb: Optional[nbformat.NotebookNode] = None) -> None:
        try:
            if path and not nb:
                nb = nbformat.read(path, as_version=4)
            if not nb:
                return
            suspicious = []
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                src = cell.source
                if "!" in src or "%%bash" in src or "pip install" in src:
                    suspicious.append(src.strip().splitlines()[0] if src else "")
            if suspicious:
                console.print("[yellow]Warning:[/yellow] notebook includes shell/pip commands; review before execution.")
        except Exception:
            return

    def _notify_agent(self, result: Dict[str, Any], context: str) -> None:
        self.last_run = result
        if not self.llm:
            return
        status = result.get("status", "unknown")
        summary_lines = [f"Run result: {status}", f"Context: {context}"]
        if result.get("error"):
            summary_lines.append(f"Error: {result['error']}")
        if result.get("output_preview"):
            tail = "\n".join(result["output_preview"])
            summary_lines.append(f"Output tail:\n{tail}")
        message = "\n".join(summary_lines)
        system = "You are assisting with notebook automation. Suggest the next action succinctly."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]
        try:
            reply = self.llm.chat(messages)
            console.print(f"[cyan]Agent:[/cyan] {reply}")
            self.history.append({"role": "assistant", "content": reply})
        except Exception as err:  # noqa: BLE001
            console.print(f"[red]Notify failed:[/red] {err}")


def slugify(text: str) -> str:
    """Convert text to a safe filename slug.

    Args:
        text: Input text to slugify

    Returns:
        Lowercase string with only alphanumeric characters and hyphens
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def validate_path(path: Path, workdir: Path) -> bool:
    """Validate that a path is safe and within the working directory.

    Args:
        path: Path to validate
        workdir: Working directory to check against

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve to absolute path and check if it's within workdir
        resolved = path.resolve()
        workdir_resolved = workdir.resolve()
        # Check if path is within workdir (prevents directory traversal)
        return resolved == workdir_resolved or workdir_resolved in resolved.parents
    except (ValueError, OSError):
        return False


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def save_env_value(key: str, value: str) -> None:
    lines: List[str] = []
    found = False
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                lines.append(line)
                continue
            k, _ = line.split("=", 1)
            if k.strip() == key:
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

@app.command()
def chat() -> None:
    """Interactive CLI loop with slash commands."""
    load_env()
    try:
        settings = AgentSettings()
        llm = LLMClient(settings)
    except Exception as err:  # noqa: BLE001
        console.print(f"[red]LLM unavailable:[/red] {err}")
        llm = None
    manager = NotebookManager(Path(".").resolve())
    console.print(f"[dim]LLM provider: {settings.provider}, model: {settings.model}, base_url: {settings.base_url}[/dim]")
    if settings.provider == "mistral" and not settings.api_key:
        console.print("[yellow]MISTRAL_API_KEY missing. Use /setkey to provide it.[/yellow]")
    loop = ChatLoop(settings, llm, manager)
    loop.run()


@app.command()
def sysinfo() -> None:
    """Print machine stats."""
    console.print(format_sysinfo())


@app.command()
def run(notebook: Path, timeout: int = typer.Option(600, help="Execution timeout seconds")) -> None:
    """Execute a notebook once and exit."""
    load_env()
    settings = AgentSettings()
    manager = NotebookManager(Path(".").resolve())
    result = manager.run(notebook, timeout=timeout)
    ChatLoop(settings, None, manager)._print_run_result(result)


@app.command()
def new(prompt: str) -> None:
    """Generate a notebook from a prompt."""
    load_env()
    settings = AgentSettings()
    if settings.provider in {"openai", "mistral"} and not settings.api_key:
        console.print("Set MISTRAL_API_KEY or OPENAI_API_KEY first.")
        raise typer.Exit(1)
    llm = LLMClient(settings)
    generator = NotebookGenerator(llm)
    nb = generator.generate(prompt)
    target = Path(f"{slugify(prompt) or 'notebook'}_{int(time.time())}.ipynb")
    NotebookManager(Path(".").resolve()).write_notebook(nb, target)
    console.print(f"Wrote {target}")


if __name__ == "__main__":
    app()
