"""Team of specialised agents routed by an LLM router."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kognios import Agent, Team, AnthropicModel, tool


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {"AAPL": "189.25", "GOOG": "175.40", "MSFT": "420.00"}
    return f"{ticker}: ${prices.get(ticker.upper(), '??')}"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"It is 18°C and cloudy in {city}."


model = AnthropicModel(model="claude-haiku-4-5-20251001")

finance_agent = Agent(
    model=model,
    tools=[get_stock_price],
    instructions="You are a financial assistant. Answer questions about stocks and markets.",
    name="finance",
    description="Handles stock prices, financial data, and market questions.",
)

weather_agent = Agent(
    model=model,
    tools=[get_weather],
    instructions="You are a weather assistant. Answer questions about weather and climate.",
    name="weather",
    description="Handles weather forecasts and climate questions.",
)

team = Team(
    router_model=model,
    agents={"finance": finance_agent, "weather": weather_agent},
)

print("Query 1: Finance")
print(team.run("What is the current price of Apple stock?"))

print("\nQuery 2: Weather")
print(team.run("What is the weather like in London?"))
