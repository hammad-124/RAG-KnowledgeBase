# Token Tracking & Budgeting

Pre-flight token budgeting for LLM API calls using `tiktoken` exact encoding counts.

## Classes

### `TokenBudget`
Core token accounting — counts tokens via model-specific encoders, enforces per-request limits, and aggregates usage stats.

| Method | Purpose |
|--------|---------|
| `count_tokens(text)` | Exact token count using `tiktoken` |
| `check_budget(text)` | Returns `(within_budget, token_count)` |
| `record_usage(input, output)` | Logs API call metrics |
| `get_stats()` | Total/avg tokens + request count |

### `BudgetedLLM`
Production wrapper around `ChatOpenAI` with automatic budget enforcement.

- **Pre-flight check**: Rejects queries exceeding `max_tokens` **before** an API call
- **Post-call recording**: Pulls exact token usage from OpenAI response metadata
- **Raises `ValueError`** when query exceeds budget

## Usage

```python
client = BudgetedLLM(model_name="gpt-4o-mini", max_tokens=4000)

response = client.invoke("Your prompt here")
print(client.get_stats())
# => { total_input, total_output, requests, total_tokens, avg_per_request }
```

## Production Efficiency

### 1. Zero-cost pre-flight rejection
Token counting via `tiktoken` is **local** (no network call). Expensive API requests are rejected immediately — you never pay for or wait on a response that would exceed budget.

```python
# Without guard: burns money + time on every oversized query
response = openai.chat.completions.create(model="gpt-4o", messages=[...])
# $0.010 * 8K tokens = $0.08 wasted if rejected

# With BudgetedLLM: local check in ~0.1ms, zero cost
client.invoke(huge_query)  # ValueError raised, no API call made
```

### 2. Exact metadata over estimation
After a successful call, actual token usage is pulled from OpenAI's `response_metadata` rather than relying on a second local count, eliminating drift between billed and tracked amounts.

### 3. Aggregate observability
`get_stats()` gives you `total_tokens`, `avg_per_request`, and request count — feed this into monitoring (Prometheus, Datadog, etc.) to set team-level budgets, detect anomalous spikes, and forecast costs.

```python
# Example: wrap in a route handler
@app.post("/chat")
def chat(query: str):
    try:
        reply = client.invoke(query)
        stats = client.get_stats()
        if stats["avg_per_request"] > 3000:
            alert_team("Usage spike detected")
        return reply
    except ValueError as e:
        return {"error": str(e)}, 429  # rate-limit signal
```

### 4. Per-user cost attribution (BYO wrapper)
Since `BudgetedLLM` isolates state per instance, you can maintain one instance per user to track individual usage — enabling per-user budgets, billing, or abuse detection.

```python
class PerUserBudgetedLLM:
    def __init__(self, max_tokens_per_user: int = 100_000):
        self._clients: Dict[str, BudgetedLLM] = {}
        self.max_per_user = max_tokens_per_user

    def for_user(self, user_id: str) -> BudgetedLLM:
        if user_id not in self._clients:
            self._clients[user_id] = BudgetedLLM(max_tokens=self.max_per_user)
        return self._clients[user_id]

    def get_user_stats(self, user_id: str) -> dict:
        return self._clients[user_id].get_stats()

    def all_stats(self) -> dict:
        return {uid: cl.get_stats() for uid, cl in self._clients.items()}


manager = PerUserBudgetedLLM(max_tokens_per_user=5000)

res_a = manager.for_user("alice").invoke("Tell me a joke")
res_b = manager.for_user("bob").invoke("Explain quantum computing")

print(manager.get_user_stats("alice"))
# => {"total_input": 12, "total_output": 45, "requests": 1, "total_tokens": 57, "avg_per_request": 57.0}

print(manager.all_stats())
# => {"alice": {...}, "bob": {...}}
```

This lets you attribute every token spent to a specific user — essential for SaaS billing, team cost centers, or capping free-tier usage.

### 5. Predictable cost ceiling
With `max_tokens=4000` and `gpt-4o-mini` (~$0.150/1M input), the **worst-case cost per request** is `4000 × 0.150 / 1_000_000 = $0.0006`. No surprise bills from runaway prompts.

## Dependencies

- `tiktoken` — token counting
- `langchain-openai` — LLM interface
- `python-dotenv` — env loading
- OpenAI API key in `.env`
