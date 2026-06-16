"""Ctx: the single object your solve(ctx) gets. The harness builds it; you read it. One surface,
two backends, so the agent you tune locally is the agent we grade:

  ctx.instruction            the task to solve
  ctx.model(messages, ...)   the fixed model through the metered proxy (OpenAI-shaped dict back)
  ctx.mcp.call(name, args)   the AppWorld MCP meta-tool surface
  ctx.retrieve(query)        your RAG hook over the API docs
  ctx.memory.read()/.write() persisted across tasks via FLYWHEEL_MEMORY_DIR, a directory that
                             survives the whole task stream (wiped between tasks only on the off-arm)
  ctx.reflect(note)          record a self-correction / retry
  ctx.execute(code)          LOCAL ONLY: run Python against AppWorld for fast iteration

GRADED run (the sandbox sets FLYWHEEL_MCP_URL): tools come from FLYWHEEL_MCP_URL, the model from
FLYWHEEL_PROXY_URL with FLYWHEEL_PROXY_TOKEN. Memory is FLYWHEEL_MEMORY_DIR: a persistent directory
that survives across tasks, and whatever you write there (a JSON file, sqlite, a vector DB you bundle
in your image) IS your memory. There is no memory service. The trusted TRACE the gate reads is the
gateways' own, so the model + tool calls it counts as evidence have to flow through ctx; that is about
trace evidence, not where you store memory.

LOCAL run (run_local.py / quickstart.py): no gateways, so AppWorld runs in-process and the model uses
FLYWHEEL_URL + FLYWHEEL_KEY. Memory is the same directory contract. ctx.mcp and ctx.execute both reach
the same in-process AppWorld so a ctx.mcp-based agent is exercised the same way it will be graded.
"""
import os

from flywheel.memory import Memory
from flywheel.mcp import MCP
from flywheel.proxy import chat
from flywheel.trace import Trace

try:  # observability shim: real Langfuse locally, no-op (and import-safe) in the graded sandbox
    import obs
except Exception:  # pragma: no cover
    class _ObsNull:
        def _n(self, *a, **k):
            class _C:
                def __enter__(self_): return self_
                def __exit__(self_, *a): return False
                def update(self_, **k): pass
            return _C()
        task = generation = span = _n
        def flush(self): pass
        def enabled(self): return False
    obs = _ObsNull()


class Ctx:
    def __init__(self, instruction, proxy_url, key, memory_dir, trace_file=None,
                 max_steps=None, mcp_url=None, memory_url=None, env=None, retriever=None):
        self.instruction = instruction
        self._proxy = (proxy_url or "").rstrip("/")
        self._key = key
        self._env = env  # local AppWorld backend, or None on the graded run
        # no step cap: the real bounds are your token budget and the per-task timeout.
        # max_steps stays as a loop-safe finite number for `range(ctx.max_steps)` loops.
        self.max_steps = int(max_steps if max_steps is not None else os.environ.get("FLYWHEEL_MAX_STEPS", "100000"))
        self._retriever = retriever
        self.trace = Trace(trace_file)
        self.memory = Memory(memory_dir, self.trace, url=memory_url)
        self.mcp = MCP(mcp_url, self.trace) if mcp_url else _LocalMCP(env, self.trace)

    @classmethod
    def from_env(cls, task_id=None, experiment_name="flywheel"):
        """Build a Ctx from the environment contract. The graded sandbox sets FLYWHEEL_MCP_URL /
        FLYWHEEL_PROXY_URL / FLYWHEEL_PROXY_TOKEN and a persistent FLYWHEEL_MEMORY_DIR; locally you
        set FLYWHEEL_KEY / FLYWHEEL_URL / APPWORLD_ROOT (see .env.example)."""
        mcp_url = os.environ.get("FLYWHEEL_MCP_URL")
        trace_file = os.environ.get("FLYWHEEL_TRACE_FILE")
        memory_dir = os.environ.get("FLYWHEEL_MEMORY_DIR", "./.memory")
        max_steps = int(os.environ.get("FLYWHEEL_MAX_STEPS", "100000"))
        if mcp_url:  # graded run: gateways, no AppWorld in-process
            return cls(
                instruction=os.environ.get("FLYWHEEL_TASK_INSTRUCTION", ""),
                proxy_url=os.environ.get("FLYWHEEL_PROXY_URL", ""),
                key=os.environ.get("FLYWHEEL_PROXY_TOKEN", ""),
                memory_dir=memory_dir, trace_file=trace_file, max_steps=max_steps,
                mcp_url=mcp_url, memory_url=os.environ.get("FLYWHEEL_MEMORY_URL"))
        from flywheel.appworld_env import AppWorldEnv  # local-only import
        os.environ.setdefault("APPWORLD_ROOT", os.environ.get("APPWORLD_ROOT", "./aw"))
        tid = task_id or os.environ.get("FLYWHEEL_TASK_ID")
        if not tid:
            raise SystemExit("no task id: set FLYWHEEL_TASK_ID or pass task_id= (local run)")
        env = AppWorldEnv(tid, experiment_name=experiment_name)
        return cls(
            instruction=env.instruction,
            proxy_url=os.environ.get("FLYWHEEL_URL", "https://homodeus-flywheel.fly.dev") + "/v1",
            key=os.environ.get("FLYWHEEL_KEY", ""),
            memory_dir=memory_dir, trace_file=trace_file, max_steps=max_steps, env=env)

    def model(self, messages, tools=None, response_format=None):
        """The fixed model through the metered proxy. Returns the raw OpenAI response dict.
        Pass `tools` for function-calling, `response_format` for structured output. The proxy
        pins the model and temperature; never send `model` or `max_tokens`."""
        self.trace("model")
        with obs.generation("model", model="gemini-3-flash-preview", input=messages) as g:
            data, remaining = chat(self._proxy, self._key, messages, tools=tools,
                                   response_format=response_format)
            try:
                usage = (data or {}).get("usage") or {}
                msg = ((data or {}).get("choices") or [{}])[0].get("message", {})
                g.update(output=msg.get("content") or msg,
                         usage_details={"input": usage.get("prompt_tokens"),
                                        "output": usage.get("completion_tokens"),
                                        "total": usage.get("total_tokens")},
                         metadata={"tokens_remaining": remaining})
            except Exception:
                pass
        if remaining is not None:
            self.trace("budget", remaining=remaining)
        return data

    def retrieve(self, query):
        """Your RAG hook over the API docs. Graded run: a `search_apis` MCP call, traced as a
        retrieval. Locally: your wired retriever if set, else the same search_apis path. You
        cannot fit 457 API docs in context, so this is load-bearing -- index the docs
        (tools/dump_api_docs.py) and pass your retriever via Ctx(retriever=...)."""
        self.trace("retrieval", query=query)
        with obs.span("retrieve", input=query) as s:
            out = self._retriever(query) if self._retriever is not None \
                else self.mcp.call("search_apis", {"query": query})
            try:
                s.update(output=str(out)[:2000])
            except Exception:
                pass
            return out

    def reflect(self, note):
        """Record a self-correction (a failed step you're about to retry differently)."""
        self.trace("reflect", note=note)

    def run_code(self, code):
        """Run a Python snippet with the AppWorld `apis` object in scope (loops, pagination, bulk
        writes). The 5th MCP tool, and the lever on heavy tasks: a 40-item task is ONE run_code,
        not 40 call_api turns. Returns the snippet's stdout (a traceback string on error). Works on
        both backends: the graded run routes to the run_code gateway tool, local runs in-process,
        so the agent you tune locally is the agent we grade. State (logins, records) persists across
        calls within a task; keep Python variables yourself by re-deriving or stashing in memory."""
        self.trace("execute")
        with obs.span("run_code", input=code) as s:
            if self._env is not None:  # local AppWorld
                out = self._env.execute(code)
            else:
                res = self.mcp.call("run_code", {"code": code})  # graded gateway
                out = (res.get("stdout") or res.get("error") or "") if isinstance(res, dict) else res
            try:
                s.update(output=str(out)[:5000])
            except Exception:
                pass
            return out

    def execute(self, code):
        """Run Python against AppWorld (`apis.<app>.<method>(...)`). Alias of ctx.run_code; works on
        the graded run too (routes to the run_code tool)."""
        return self.run_code(code)

    def evaluate(self):
        """Deterministic oracle for the current task (local self-testing only; the real grader
        calls it itself). True iff the goal state was reached."""
        if self._env is None:
            raise RuntimeError("ctx.evaluate is local-only; the grader runs the oracle itself")
        return self._env.evaluate()


class _LocalMCP:
    """Local stand-in for the graded MCP gateway, backed by in-process AppWorld so a ctx.mcp-based
    agent runs the same locally as under grading. It exposes the same five meta-tools as the real
    gateway: search_apis, api_doc, call_api, run_code, complete_task."""
    def __init__(self, env, trace):
        self._env = env
        self._t = trace

    def list(self):
        return [
            {"name": "search_apis"},
            {"name": "api_doc"},
            {"name": "call_api"},
            {"name": "run_code"},
            {"name": "complete_task"},
        ]

    def call(self, name, args=None):
        self._t("tool", name=name)
        if self._env is None:
            return {"error": "no MCP url and no local AppWorld; set FLYWHEEL_MCP_URL or APPWORLD_ROOT"}
        args = args or {}
        if name == "search_apis":
            return self._search(args.get("query", ""))
        if name == "api_doc":
            return self._api_doc(args.get("app", ""), args.get("api", ""))
        if name == "call_api":
            return self._call_api(args.get("app", ""), args.get("api", ""), args.get("arguments") or {})
        if name == "run_code":
            return {"stdout": self._env.execute(args.get("code", ""))}
        if name == "complete_task":
            return self._complete(args.get("answer"))
        return {"error": f"unknown tool {name}"}

    def _run_json(self, code):
        import json
        marker = "__FW_LOCAL__"
        out = self._env.execute(code.replace("__MARKER__", marker))
        for line in reversed((out or "").splitlines()):
            if line.startswith(marker):
                return json.loads(line[len(marker):])
        return {"error": "no_result", "raw": (out or "")[-500:]}

    def _search(self, query):
        import json
        code = (
            "import json\n"
            f"q = {json.dumps(str(query))}.lower()\n"
            "hits = []\n"
            "for app in apis.api_docs.show_app_descriptions():\n"
            "    an = app['name']\n"
            "    for d in apis.api_docs.show_api_descriptions(app_name=an):\n"
            "        blob = (an + ' ' + d['name'] + ' ' + d.get('description','')).lower()\n"
            "        if all(w in blob for w in q.split()):\n"
            "            hits.append({'app': an, 'api': d['name'], 'description': d.get('description','')})\n"
            "print('__MARKER__' + json.dumps({'results': hits[:20]}))")
        return self._run_json(code)

    def _api_doc(self, app, api):
        import json
        code = (
            "import json\n"
            f"doc = apis.api_docs.show_api_doc(app_name={json.dumps(str(app))}, api_name={json.dumps(str(api))})\n"
            "print('__MARKER__' + json.dumps({'doc': doc}, default=str))")
        return self._run_json(code)

    def _call_api(self, app, api, arguments):
        import json
        code = (
            "import json\n"
            f"args = json.loads({json.dumps(json.dumps(arguments or {}))})\n"
            f"res = getattr(getattr(apis, {json.dumps(str(app))}), {json.dumps(str(api))})(**args)\n"
            "print('__MARKER__' + json.dumps({'result': res}, default=str))")
        return self._run_json(code)

    def _complete(self, answer):
        import json
        if answer is None or str(answer).strip() in ("", "<<not_given>>"):
            code = "import json\napis.supervisor.complete_task()\nprint('__MARKER__' + json.dumps({'ok': True}))"
        else:
            code = (
                "import json\n"
                f"apis.supervisor.complete_task(answer={json.dumps(str(answer))})\n"
                "print('__MARKER__' + json.dumps({'ok': True}))")
        return self._run_json(code)
