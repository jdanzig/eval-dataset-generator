"""Load .env from project root so ANTHROPIC_API_KEY is available. Values already in env win."""
import os
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_p):
    for _l in open(_p, encoding="utf-8"):
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _, _v = _l.partition("="); os.environ.setdefault(_k.strip(), _v.strip())
