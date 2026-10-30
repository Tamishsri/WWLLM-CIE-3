"""
Studio service layer.

Sits between the HTTP API (``src.api``) and the ML pipelines
(``src.pipelines``): detects which models can run, loads them lazily,
falls back to offline engines, and turns every operation into a stream
of progress events.
"""
