import asyncio
import os
import requests
from dotenv import load_dotenv


from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console

load_dotenv()

# Main async
async def main():


    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    planner = AssistantAgent(
        name="Planner",
        description="Delegates tasks to travel, weather, hotel, and flight agents.",
        model_client=model_client,
        system_message="""You are the Planner. Your task is to:
        - Assign flight queries to FlightAgent
        - Assign train queries to WebSearchAgent (or a TrainAgent if available)
        - Assign hotel searches to HotelAgent
        - Assign weather queries to WeatherAgent
        - Delegate complete itinerary optimization to TravelAdvisor

        use these templates to respond finally:
        Always output the final result using the following format:

        1. Flights/Trains Table:
        | FLIGHT NAME  | Date       | Time     | Origin         | Destination       | Duration | Price (INR) | Comfort |
        |--------------|------------|----------|----------------|-------------------|----------|-------------|---------|
        | ...          | ...        | ...      | ...            | ...               | ...      | ...         | ...     |

        2. Hotels Table:
        | Hotel Name         | Address                          | Price (INR) | Check-In | Check-Out | Rating |
        |--------------------|----------------------------------|-------------|----------|-----------|--------|
        | ...                | ...                              | ...         | ...      | ...       | ...    |

        3. Popular Places to visit:
        |  Name of the tourist place | Address                          | Weather     | Good time to visit |
        |----------------------------|----------------------------------|-------------|--------------------|
        | ...                        | ...                              | ...         | ...                | 

        For final responses:
        - Present flight/train comparisons in a TABLE with columns: Mode, Date, Time, Origin, Destination, Duration, Price, Comfort
        - Present hotel options in a TABLE with: Name, Address, Price, Check-in, Check-out, Rating
        - Always show at least 3 flight/train/hotel options when available.
        - Use 'TERMINATE' at the end of the final message.
        """
    )

    weather_agent = AssistantAgent(
        name="WeatherAgent",
        description="Fetches weather info.",
        model_client=model_client,
        tools=[weather_tool],
        system_message="Use weather_tool to respond to weather queries."
    )

    web_search_agent = AssistantAgent(
        name="WebSearchAgent",
        description="Performs general search queries.",
        model_client=model_client,
        tools=[web_search],
        system_message="Use web_search for general info. Do not guess."
    )

    hotel_agent = AssistantAgent(
        name="HotelAgent",
        description="Finds hotels using Google Hotels.",
        model_client=model_client,
        tools=[google_hotels_search],
        system_message="""
        Use google_hotels_search to find hotels by location and date.
        Give detailed information which should include:
        Hotel Name, Address, Check in time, check out time, Price per night, total Price, rating, Amenities in detail
        """
    )

    flight_agent = AssistantAgent(
        name="FlightAgent",
        description="Finds flights using Google Flights.",
        model_client=model_client,
        tools=[google_flights_search],
        system_message="""
        Use google_hotels_search to find hotels by location and date.
        Give detailed information which should include:
        Airline Name, Flight Number, Departure time, Arrival Time, Duration, Price, Class in detail
        """
    )

    travel_advisor = AssistantAgent(
        name="TravelAdvisor",
        description="Plans complete trips using hotel, flight, weather and search tools.",
        model_client=model_client,
        tools=[
            google_flights_search,
            google_hotels_search,
            weather_tool,
            web_search
        ],
        system_message="Combine all tools to give structured trip plans."
    )

    team = SelectorGroupChat(
        participants=[
            planner,
            weather_agent,
            web_search_agent,
            hotel_agent,
            flight_agent,
            travel_advisor
        ],
        model_client=model_client,
        termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(30),
        selector_prompt="""Choose the best agent for each task.

        {roles}

        Conversation:
        {history}

        Pick one from {participants}. Let the planner assign tasks first.
        """
    )

    task = (
        "Plan a trip from hyderabad to outskirts of hyderbad not more than 80Kms, on bike the weather with some good places to visit have some good vibes for a one day activity, give 5"
    )

    # await Console(team.run_stream(task=task))
    result = await Console(team.run_stream(task=task))
    
    if result and hasattr(result, 'summary'):
        summary = result.summary
        print("\n---------- Summary ----------")
        print(f"Number of messages: {summary.num_messages}")
        print(f"Finish reason: {summary.finish_reason}")
        print(f"Total prompt tokens: {summary.prompt_tokens}")
        print(f"Total completion tokens: {summary.completion_tokens}")
        print(f"Total tokens: {summary.prompt_tokens + summary.completion_tokens}")
        print(f"Duration: {summary.duration:.2f} seconds")
    else:
        print("\nNo summary available")
    
    await model_client.close()
    
if __name__ == "__main__":

    setup_tracing(service_name="TripPlanner")

    model_client = asyncio.run(main())
    
    try:
        usage = model_client.last_usage()  
        print("Total tokens used:", usage.get("total_tokens", "N/A"))
        print("Prompt tokens:", usage.get("prompt_tokens", "N/A"))
        print("Completion tokens:", usage.get("completion_tokens", "N/A"))
    except Exception as e:
        print(f"Error accessing token usage: {e}")

#  plan a trip from bangalore to hyderbad today after 6pm, to reach hyderabad list the hotels and flights avaiable under 10k budget, I will be leaving from JP nagar, bngalore, Have to reach Hyderabad Madhapur and also list the hotels avaiable.
