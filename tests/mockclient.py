"""MockClient: same chat() surface as core.ollama.OllamaClient, but returns
scripted or computed responses instantly — so the whole engine (retries,
forfeits, clock, records, tournaments) is testable in milliseconds."""


class MockClient:
    """Either pass `script` (a list consumed in order) or `responder`
    (a callable (model, messages) -> item). An item is a string (the content)
    or a dict: {"content": str, "think_chars": int, "timeout": bool}."""

    base_url = "mock://"

    def __init__(self, script=None, responder=None):
        self.script = list(script or [])
        self.responder = responder
        self.calls = []

    def list_installed(self):
        return [("mock:latest", 1)]

    def chat(self, model, messages, *, schema=None, temperature=0.7,
             num_predict=2048, think=True, think_effort=None, deadline=None,
             on_think=None):
        self.calls.append({"model": model, "messages": messages, "schema": schema})
        if self.responder is not None:
            item = self.responder(model, messages)
        elif self.script:
            item = self.script.pop(0)
        else:
            raise AssertionError("MockClient: no scripted responses left")
        if isinstance(item, str):
            item = {"content": item}
        if item.get("think_chars") and on_think:
            on_think("…")
        return (item.get("content", ""), item.get("think_chars", 0),
                item.get("timeout", False))
