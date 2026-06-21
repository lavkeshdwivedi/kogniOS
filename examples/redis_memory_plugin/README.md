# kognios-redis-memory

A Redis-backed long-term memory plugin for [kogniOS](https://github.com/lavkeshdwivedi/kogniOS).

Facts are stored in a Redis hash so they persist across restarts and can be
shared across multiple agent processes.

## Installation

```bash
pip install kognios redis>=5.0
# — or install this example package directly —
pip install -e examples/redis_memory_plugin/
```

## Quick start

```python
from redis_memory import RedisLongTermMemory
from kognios import Agent
from kognios.models.anthropic import AnthropicModel

# Create memory (connects to localhost:6379 by default)
memory = RedisLongTermMemory(prefix="mybot", ttl=86400)   # facts expire after 24 h

# Store and retrieve facts directly
memory.remember("user_name", "Alice")
memory.remember("preferred_language", "Python")

print(memory.recall("user_name"))   # Alice
print(memory.facts())               # ['user_name=Alice', 'preferred_language=Python']

# Inject into an agent
agent = Agent(
    model=AnthropicModel("claude-sonnet-4-5"),
    system=memory.as_system_prompt_fragment(),
)
response = agent.run("What is my name?")
print(response)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `"localhost"` | Redis server hostname |
| `port` | `6379` | Redis server port |
| `db` | `0` | Redis database index |
| `prefix` | `"kognios:memory"` | Namespace prefix for all keys |
| `ttl` | `None` | Seconds before keys expire; `None` = never |
| `redis_client` | `None` | Pass a pre-built `redis.Redis` instance (skips host/port/db) |

## Entry-point registration

When installed, the plugin registers itself via the `kognios.memory` entry-point
group so kogniOS can discover it automatically:

```toml
[project.entry-points."kognios.memory"]
redis = "redis_memory:RedisLongTermMemory"
```

## Running the tests

No real Redis server is required — tests use `unittest.mock`:

```bash
cd examples/redis_memory_plugin
python -m pytest tests/ -v
```
