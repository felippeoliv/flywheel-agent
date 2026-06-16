"""Offline BM25 retriever over the dumped AppWorld API docs (api_docs_dump/apis/*.txt).

Zero dependencies (pure stdlib), so it runs in the offline graded sandbox as long as the
api_docs_dump/ directory is committed to the repo. The point: you cannot fit 457 API docs in
context, so for each task we retrieve the handful that matter and inject their exact params +
response shapes into the prompt. That kills the #1 crash (guessed field names) before it happens.

BM25 (Okapi) over snake_case/camelCase-aware tokens, with a light boost when the query mentions an
app name. Each chunk is one API: "<app>.<api> [METHOD path]\n<desc>\n  - param...\n  response: ...".
"""
import math
import os
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# all 11 AppWorld apps, so a query that names one boosts its chunks
APPS = (
    "spotify", "amazon", "gmail", "phone", "venmo", "splitwise",
    "todoist", "simple_note", "file_system", "supervisor", "api_docs",
)


def _tokenize(text):
    text = _CAMEL_RE.sub(" ", text or "")
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, docs_dir=None, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.dir = docs_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "api_docs_dump", "apis"
        )
        self.ids = []           # ["spotify__login", ...]
        self.texts = []         # raw chunk text, for prompt injection
        self.apps = []          # app of each chunk
        self._tf = []           # per-doc term frequency Counter
        self._df = Counter()    # document frequency per term
        self._len = []          # per-doc length
        self._avglen = 0.0
        self._idf = {}
        self._load()

    def _load(self):
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(
                f"api_docs_dump not found at {self.dir}. Run tools/dump_api_docs.py and commit it."
            )
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(self.dir, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            doc_id = fn[:-4]
            app = doc_id.split("__", 1)[0]
            toks = _tokenize(text)
            tf = Counter(toks)
            self.ids.append(doc_id)
            self.texts.append(text)
            self.apps.append(app)
            self._tf.append(tf)
            self._len.append(len(toks))
            for term in tf:
                self._df[term] += 1
        n = len(self.ids)
        self._avglen = (sum(self._len) / n) if n else 0.0
        for term, df in self._df.items():
            # BM25 idf with +1 floor so common terms never go negative
            self._idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def _score(self, q_terms, i):
        tf, dl = self._tf[i], self._len[i]
        denom_norm = self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1))
        s = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            s += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
        return s

    def search(self, query, k=12, app_hint=None):
        """Return up to k {'id','app','api','text','score'} dicts, best first."""
        q_terms = _tokenize(query)
        named = {a for a in APPS if a in query.lower()}
        if app_hint:
            named.add(app_hint)
        scored = []
        for i in range(len(self.ids)):
            s = self._score(q_terms, i)
            if self.apps[i] in named:
                s *= 1.5  # query names this app: lift its endpoints
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, i in scored[:k]:
            app, api = self.ids[i].split("__", 1)
            out.append({"id": self.ids[i], "app": app, "api": api,
                        "text": self.texts[i], "score": round(s, 3)})
        return out

    def detect_apps(self, query):
        """Apps explicitly named in the query (so we can inject their full catalog)."""
        ql = query.lower()
        return [a for a in APPS if a in ql and a not in ("api_docs",)]

    def app_catalog(self, app):
        """Every api of one app as compact 'app.api : one-line description' rows. Guarantees the
        model sees the full endpoint surface of an app it must operate (no missed list API)."""
        rows = []
        for i, doc_id in enumerate(self.ids):
            if self.apps[i] != app:
                continue
            lines = self.texts[i].splitlines()
            head = lines[0].split("  [", 1)[0].strip()  # "app.api"
            desc = (lines[1].strip() if len(lines) > 1 else "")
            rows.append(f"{head} : {desc}")
        return rows

    def context_block(self, query, k=12, app_hint=None, char_budget=4000):
        """Top-k chunks concatenated as a prompt-ready block, trimmed to char_budget."""
        hits = self.search(query, k=k, app_hint=app_hint)
        block, used = [], 0
        for h in hits:
            chunk = h["text"].strip()
            if used + len(chunk) > char_budget and block:
                break
            block.append(chunk)
            used += len(chunk) + 2
        return "\n\n".join(block), hits


_SINGLETON = None


def get_retriever():
    """Process-wide singleton so the index builds once per task run."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = BM25Retriever()
    return _SINGLETON


if __name__ == "__main__":
    r = get_retriever()
    print(f"indexed {len(r.ids)} api docs across {len(set(r.apps))} apps\n")
    for q in [
        "follow every artist of classical genre songs in my playlists",
        "comment and like each coworker venmo payment from last 7 days",
        "top 4 most played r&b song titles across libraries",
    ]:
        print(f"Q: {q}")
        for h in r.search(q, k=5):
            print(f"   {h['score']:6.2f}  {h['id']}")
        print()
