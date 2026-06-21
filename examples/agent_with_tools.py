"""Agent that calls a custom tool."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kognios import Agent, AnthropicModel, tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Stub — replace with a real weather API call.
    return f"It is 22°C and sunny in {city}."


@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    from datetime import datetime, timezone as tz
    import zoneinfo
    try:
        dt = datetime.now(zoneinfo.ZoneInfo(timezone))
        return dt.strftime("%H:%M %Z")
    except Exception:
        return datetime.utcnow().strftime("%H:%M UTC")


model = AnthropicModel(model="claude-haiku-4-5-20251001")
agent = Agent(
    model=model,
    tools=[get_weather, get_time],
    instructions="You are a helpful assistant with access to weather and time tools.",
)

answer = agent.run("What is the weather in Tokyo and what time is it there?")
print(answer)
