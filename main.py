import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # Load environment variables from .env file


def get_geocode(location: str) -> dict:
    """
    Get the latitude and longitude for a given location using the Geocode API.
    """
    api_key = os.getenv("GEOCODE_API_KEY")
    api_url = os.getenv("GEOCODE_API_URL")

    if not api_key or not api_url:
        return {"status": "error", "error": "Geocode API key or URL is not set."}

    params = {"q": location, "api_key": api_key}

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"status": "error", "error": f"Geocode request failed: {exc}"}

    if isinstance(data, list) and len(data) > 0:
        return {"status": "success", "data": {"lat": data[0]["lat"], "lng": data[0]["lon"]}}

    return {"status": "error", "error": f"No geocode data available for {location}."}


@tool
def get_weather(location: str) -> str:
    """
    You are a friendly weather assistant. Your primary goal is to present
    current weather conditions in a clear, natural, human-readable format
    that anyone can understand at a glance.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    api_url = os.getenv("WEATHER_API_URL")

    if not api_key or not api_url:
        return "Weather API key or URL is not set."

    if isinstance(location, str):
        geocode = get_geocode(location)
        if geocode.get("status") == "error":
            return geocode["error"]
        location = f"{geocode['data']['lat']},{geocode['data']['lng']}"

    lat, lng = location.split(",", 1)
    params = {
        "lat": lat,
        "lng": lng,
        "params": "airTemperature,humidity,windSpeed",
        "source": "noaa",
    }
    headers = {"Authorization": api_key}


    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Failed to retrieve weather data: {exc}"

    if "hours" in data and len(data["hours"]) > 0:
        weather_info = data["hours"][0]
        return (
            f"Current weather at {location}: "
            f"Temperature: {weather_info['airTemperature']['noaa']}°C, "
            f"Humidity: {weather_info['humidity']['noaa']}%, "
            f"Wind Speed: {weather_info['windSpeed']['noaa']} m/s"
        )

    return f"No weather data available for {location}."


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    llm_with_tools = llm.bind_tools([get_weather])
    user_message = "What is the weather in Gurgaon?"
    messages = [
        ("user", user_message)
    ]
    response = llm_with_tools.invoke(messages)

print("Tool calls:")
print(response.tool_calls)


# --------------------------------------------------
# Execute tool calls
# --------------------------------------------------

if response.tool_calls:

    messages.append(response)

    for tool_call in response.tool_calls:

        if tool_call["name"] == "get_weather":
            print(f"\nExecuting tool call: {tool_call['name']} with args: {tool_call['args']['location']}")
            tool_result = get_weather.invoke('gurgaon')

            print("\nTool result:")
            print(tool_result)

            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"],
                }
            )


    # --------------------------------------------------
    # Send tool result back to Gemini
    # --------------------------------------------------

    final_response = llm_with_tools.invoke(messages)

else:
    final_response = response


print("\nFinal LLM response:")
print(final_response.content[0]["text"])