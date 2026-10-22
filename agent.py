from smolagents import ToolCallingAgent, TransformersModel, tool

@tool
def calculate_fuel(distance_km: float, consumption_per_100km: float = 5.0) -> str:
    """
    Calculates required fuel for a motorcycle trip.

    Args:
        distance_km: Distance in kilometers.
        consumption_per_100km: Fuel consumption in liters per 100km.

    Returns:
        String message with exact fuel required.
    """
    fuel_needed = (distance_km / 100) * consumption_per_100km
    return f"To travel {distance_km} km, you will need {fuel_needed:.2f} liters of fuel."

# 1. ენის მოდელი (GPU-ზე გაშვება)
model = TransformersModel(
    model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    device_map="auto"
)

# 2. აგენტი
agent = ToolCallingAgent(
    tools=[calculate_fuel], 
    model=model
)

# 3. გაშვების მაგალითი
if __name__ == "__main__":
    response = agent.run("Calculate fuel for distance_km 360 and consumption_per_100km 4.5")
    print(response)
