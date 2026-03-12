"""LangGraph-based LLM fallback for dependency resolution.

Uses LangGraph state graphs with LangChain's ChatOllama/ChatOpenAI
for structured, retry-aware LLM queries. Only invoked when deterministic
strategies (AST parsing, regex error parsing, binary search) are exhausted.

Graph architecture:
  suggest_version:  query → validate → retry(up to 3) → result
  identify_module:  query → validate → retry(up to 3) → result
"""
import os
import re
from typing import Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END


# ── Pydantic schemas for structured LLM output ──────────────────────────────

class ModuleVersion(BaseModel):
    module: str = Field(description="Name of the module")
    version: str = Field(description="The version of the module to use")


class Module(BaseModel):
    module: str = Field(description="Name of the module")


# ── LangGraph state types ───────────────────────────────────────────────────

class VersionState(TypedDict):
    module: str
    versions_str: str
    python_version: str
    excluded_versions: str
    error_context: str
    result: Optional[str]
    attempts: int


class ModuleState(TypedDict):
    error_msg: str
    installed_modules: str
    result: Optional[str]
    attempts: int


# ── LLM Client ──────────────────────────────────────────────────────────────

def _create_chat_model(base_url, model, temperature):
    """Create the appropriate LangChain chat model."""
    if 'gpt' in model:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('OPENAI_KEY', '')
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=base_url, model=model,
            format="json", temperature=temperature
        )


class LLMClient:
    """LangGraph-based LLM client for strategic dependency resolution queries.

    Uses state graphs with automatic retry and structured output parsing.
    Compatible with Ollama (local) and OpenAI (GPT) models.
    """

    def __init__(self, base_url="http://localhost:11434", model="phi3:medium",
                 temperature=0.3, max_retries=3):
        self.base_url = base_url
        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries

        self._chat_model = _create_chat_model(base_url, model, temperature)
        self._version_graph = self._build_version_graph()
        self._module_graph = self._build_module_graph()

    # ── Version suggestion graph ─────────────────────────────────────────

    def _build_version_graph(self):
        """Build a LangGraph for version suggestion with retry logic."""
        chat_model = self._chat_model
        max_retries = self.max_retries

        def suggest_node(state: VersionState) -> dict:
            """Query the LLM to suggest a module version."""
            parser = JsonOutputParser(pydantic_object=ModuleVersion)

            exclude_clause = ""
            if state["excluded_versions"]:
                exclude_clause = (
                    f"Do NOT pick any of these previously failed versions: "
                    f"{state['excluded_versions']}."
                )

            error_clause = ""
            if state["error_context"]:
                error_clause = (
                    f"Context error: {state['error_context'][:300]}"
                )

            prompt = PromptTemplate(
                template=(
                    "Available versions for '{module}' (oldest to newest): "
                    "{versions_str}\n\n"
                    "Pick ONE version of '{module}' compatible with Python "
                    "{python_version}. {exclude_clause} {error_clause}\n\n"
                    "Return ONLY valid JSON: "
                    '{{\"module\": \"{module}\", \"version\": \"X.Y.Z\"}}'
                ),
                input_variables=[],
                partial_variables={
                    "module": state["module"],
                    "versions_str": state["versions_str"][:2000],
                    "python_version": state["python_version"],
                    "exclude_clause": exclude_clause,
                    "error_clause": error_clause,
                },
            )

            try:
                chain = prompt | chat_model | parser
                out = chain.invoke({})

                if out and isinstance(out, dict) and "version" in out:
                    ver = str(out["version"]).strip()
                    # Validate: must look like a version, not prose
                    if (ver and len(ver) < 30
                            and any(c.isdigit() for c in ver)
                            and ver.lower() not in ('none', '')):
                        # Check not in excluded
                        excluded = set(
                            v.strip()
                            for v in state["excluded_versions"].split(",")
                            if v.strip()
                        )
                        if ver not in excluded:
                            return {
                                "result": ver,
                                "attempts": state["attempts"] + 1,
                            }
            except Exception as e:
                print(f"  LLM suggest_version attempt "
                      f"{state['attempts'] + 1}: {e}")

            return {"result": None, "attempts": state["attempts"] + 1}

        def should_retry(state: VersionState) -> str:
            if state["result"] is not None:
                return END
            if state["attempts"] >= max_retries:
                return END
            return "suggest"

        graph = StateGraph(VersionState)
        graph.add_node("suggest", suggest_node)
        graph.add_conditional_edges("suggest", should_retry,
                                    {"suggest": "suggest", END: END})
        graph.set_entry_point("suggest")
        return graph.compile()

    # ── Module identification graph ──────────────────────────────────────

    def _build_module_graph(self):
        """Build a LangGraph for identifying error-causing modules."""
        chat_model = self._chat_model
        max_retries = self.max_retries

        def identify_node(state: ModuleState) -> dict:
            """Query the LLM to identify the offending module."""
            parser = JsonOutputParser(pydantic_object=Module)

            prompt = PromptTemplate(
                template=(
                    "Given this Python error:\n{error_msg}\n\n"
                    "Installed modules: {installed_modules}\n\n"
                    "Which ONE installed module is most likely causing this "
                    "error? Return ONLY valid JSON: "
                    '{{\"module\": \"module_name\"}}'
                ),
                input_variables=[],
                partial_variables={
                    "error_msg": state["error_msg"][:500],
                    "installed_modules": state["installed_modules"],
                },
            )

            try:
                chain = prompt | chat_model | parser
                out = chain.invoke({})

                if out and isinstance(out, dict) and "module" in out:
                    mod = str(out["module"]).strip().lower()
                    if mod and len(mod) < 50 and re.match(r'^[a-zA-Z0-9_.-]+$', mod):
                        return {
                            "result": mod,
                            "attempts": state["attempts"] + 1,
                        }
            except Exception as e:
                print(f"  LLM identify_module attempt "
                      f"{state['attempts'] + 1}: {e}")

            return {"result": None, "attempts": state["attempts"] + 1}

        def should_retry(state: ModuleState) -> str:
            if state["result"] is not None:
                return END
            if state["attempts"] >= max_retries:
                return END
            return "identify"

        graph = StateGraph(ModuleState)
        graph.add_node("identify", identify_node)
        graph.add_conditional_edges("identify", should_retry,
                                    {"identify": "identify", END: END})
        graph.set_entry_point("identify")
        return graph.compile()

    # ── Public API ───────────────────────────────────────────────────────

    def suggest_version(self, module, versions_str, python_version,
                        excluded_versions="", error_context=""):
        """Ask LLM to suggest a module version via LangGraph.

        Uses a retry-aware state graph for robust structured output.
        """
        initial_state: VersionState = {
            "module": module,
            "versions_str": versions_str,
            "python_version": python_version,
            "excluded_versions": excluded_versions,
            "error_context": error_context,
            "result": None,
            "attempts": 0,
        }

        final_state = self._version_graph.invoke(initial_state)
        return final_state.get("result")

    def identify_module_from_error(self, error_msg, installed_modules):
        """Ask LLM to identify the error-causing module via LangGraph.

        Uses a retry-aware state graph for robust structured output.
        """
        modules_str = (
            ', '.join(installed_modules) if installed_modules else 'unknown'
        )

        initial_state: ModuleState = {
            "error_msg": error_msg,
            "installed_modules": modules_str,
            "result": None,
            "attempts": 0,
        }

        final_state = self._module_graph.invoke(initial_state)
        return final_state.get("result")

    def is_available(self):
        """Check if the LLM backend is reachable."""
        try:
            import requests
            resp = requests.get(
                f"{self.base_url.rstrip('/')}/api/tags", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False
