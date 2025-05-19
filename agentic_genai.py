import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import asyncio

load_dotenv()

# This agent is specialized for executive sales insights
async def run_sales_insight_agent(prompt: str, model: str = "gemini-1.5-flash") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "<div class='alert alert-warning'>GEMINI_API_KEY not set in .env</div>"
    model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
    agent = AssistantAgent(
        name="SalesInsightAgent",
        description="Generates detailed, actionable executive sales insights from forecast and sales data.",
        model_client=model_client,
        system_message="""
        You are a senior executive sales analyst. Given a prompt containing forecast tables, sales summaries, and branch/manager/head details, generate:
        - A concise executive summary of the forecast
        - Key trends and risks
        - Actionable recommendations for improvement
        - Highlight best/worst performers and explain why
        - Use clear, business-friendly language
        - Always provide at least 3 actionable insights
        - If a table is provided, reference it in your analysis
        - Never output code, only formatted text/HTML
        - End with a summary paragraph for leadership
        """
    )
    # Run the agent and return the result
    result = await agent.run(prompt)
    await model_client.close()
    return result.response if hasattr(result, 'response') else str(result)
