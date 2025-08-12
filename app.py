from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import TypedDict, Annotated, List
import gradio as gr
import re
import os
# Initialize Tavily client with your API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # <-- Replace with your real API key

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# LLM setup (add your Groq API key if you want to use LLM for itinerary)
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

# Prompt for itinerary generation
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a practical, relatable travel assistant. Fetch real travel tips and create a mini-itinerary for {city} based on the user's interests: {interests}. Make it suitable for group brainstorming and discussion. Provide a brief, bulleted itinerary and practical tips."),
    ("human", "Create an itinerary for my day trip."),
])

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "the messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

def fetch_travel_tips(city, interests):
    query = f"best spots in {city} for {', '.join(interests)} on a budget"
    try:
        response = tavily_client.search(query)
        results = response.get('results', [])
        if not results:
            return "No travel tips found."

        all_points = []
        for result in results:
            summary = result.get('snippet') or result.get('content') or result.get('title', 'No summary available')
            # Split the summary by numbered or hyphenated list markers
            points = re.split(r'\s*(?:\d+\.\s+|-)\s*', summary)
            # Filter out any empty strings resulting from the split
            cleaned_points = [p.strip() for p in points if p and p.strip()]
            all_points.extend(cleaned_points)

        if not all_points:
            # Fallback to the original content if splitting fails
            return "\n".join([res.get('content', '') for res in results])

        # Format the collected points as a Markdown bulleted list
        return "\n".join([f"- {point}" for point in all_points])
    except Exception as e:
        return f"Error fetching travel tips: {e}"

def fetch_budget_estimate(city, interests):
    query = f"average daily travel cost in {city} for {', '.join(interests)}"
    try:
        response = tavily_client.search(query)
        for result in response.get('results', []):
            if 'cost' in result.get('title', '').lower() or 'budget' in result.get('title', '').lower():
                summary = result.get('snippet') or result.get('content') or result.get('title', 'No summary available')
                return summary
        if response.get('results'):
            result = response['results'][0]
            summary = result.get('snippet') or result.get('content') or result.get('title', 'No summary available')
            return summary
        return "No budget estimate found."
    except Exception as e:
        return f"Error fetching budget estimate: {e}"

def fetch_nearby_hotels(city, interests):
    query = f"best budget hotels in {city} for {', '.join(interests)}"
    try:
        response = tavily_client.search(query)
        results = response.get('results', [])
        if not results:
            return "No nearby hotels found."

        all_points = []
        for result in results:
            summary = result.get('snippet') or result.get('content') or result.get('title', 'No summary available')
            # Split the summary by numbered or hyphenated list markers
            points = re.split(r'\s*(?:\d+\.\s+|-)\s*', summary)
            # Filter out any empty strings resulting from the split
            cleaned_points = [p.strip() for p in points if p and p.strip()]
            all_points.extend(cleaned_points)

        if not all_points:
            # Fallback to the original content if splitting fails
            return "\n".join([res.get('content', '') for res in results])

        # Format the collected points as a Markdown bulleted list
        return "\n".join([f"- {point}" for point in all_points])
    except Exception as e:
        return f"Error fetching nearby hotels: {e}"

# --- StateGraph CLI Version ---
def input_city(state: PlannerState) -> PlannerState:
    user_message = input("Please enter the city you want to visit for your day trip: ")
    return {
        **state,
        "city": user_message,
        "messages": state['messages'] + [HumanMessage(content=user_message)]
    }

def input_interest(state: PlannerState) -> PlannerState:
    user_message = input(f"Please enter your interest for the trip to : {state['city']} (comma-separated): ")
    return {
        **state,
        "interests": [interest.strip() for interest in user_message.split(",")],
        "messages": state['messages'] + [HumanMessage(content=user_message)]
    }

def create_itinerary(state: PlannerState) -> PlannerState:
    print(f"Creating an itinerary for {state['city']} based on interests : {', '.join(state['interests'])}")
    travel_tips = fetch_travel_tips(state['city'], state['interests'])
    budget_estimate = fetch_budget_estimate(state['city'], state['interests'])
    nearby_hotels = fetch_nearby_hotels(state['city'], state['interests'])
    response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=','.join(state['interests'])))
    content = response.content
    practical_tips = ""
    if "Practical Tips:" in content:
        parts = content.split("Practical Tips:", 1)
        itinerary = parts[0].strip()
        practical_tips = parts[1].strip()
    else:
        itinerary = content
    print("\nTravel Tips:")
    print(travel_tips)
    print("\nBudget Estimation:")
    print(budget_estimate)
    print("\nNearby Hotels:")
    print(nearby_hotels)
    print("\nFinal Itinerary: ")
    print(itinerary)
    if practical_tips:
        print("\nPractical Tips:")
        print(practical_tips)
    combined = f"Travel Tips:\n{travel_tips}\n\nBudget Estimation:\n{budget_estimate}\n\nNearby Hotels:\n{nearby_hotels}\n\nItinerary:\n{itinerary}"
    if practical_tips:
        combined += f"\n\nPractical Tips:\n{practical_tips}"
    return {
        **state,
        "messages": state['messages'] + [AIMessage(content=combined)],
        "itinerary": combined,
    }

def run_cli():
    workflow = StateGraph(PlannerState)
    workflow.add_node("input_city", input_city)
    workflow.add_node("input_interest", input_interest)
    workflow.add_node("create_itinerary", create_itinerary)
    workflow.set_entry_point("input_city")
    workflow.add_edge("input_city", "input_interest")
    workflow.add_edge("input_interest", "create_itinerary")
    workflow.add_edge("create_itinerary", END)
    app = workflow.compile()
    print("Welcome to the Travel Tip Agent CLI!")
    user_request = input("Enter your initial request (e.g., 'I want to plan a day trip'): ")
    state = {
        "messages": [HumanMessage(content=user_request)],
        "city": "",
        "interests": [],
        "itinerary": "",
    }
    for output in app.stream(state):
        pass

# --- Gradio Web App Version ---
def input_city_gr(city: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "city": city,
        "messages": state['messages'] + [HumanMessage(content=city)],
    }

def input_interests_gr(interests: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "interests": [interest.strip() for interest in interests.split(',')],
        "messages": state['messages'] + [HumanMessage(content=interests)],
    }

def create_itinerary_gr(state: PlannerState) -> str:
    travel_tips = fetch_travel_tips(state['city'], state['interests'])
    budget_estimate = fetch_budget_estimate(state['city'], state['interests'])
    nearby_hotels = fetch_nearby_hotels(state['city'], state['interests'])
    response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=", ".join(state['interests'])))
    content = response.content
    practical_tips = ""
    if "Practical Tips:" in content:
        parts = content.split("Practical Tips:", 1)
        itinerary = parts[0].strip()
        practical_tips = parts[1].strip()
    else:
        itinerary = content
    # Ensure all text is black and travel tips are broken down as points
    output = f"""
### Travel Tips
{travel_tips}

### Budget Estimation
{budget_estimate}

### Nearby Hotels
{nearby_hotels}

### Itinerary
{itinerary}
"""
    if practical_tips:
        output += f"\n### Practical Tips\n{practical_tips}"
    return output

def travel_planner_gr(city: str, interests: str):
    state = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }
    state = input_city_gr(city, state)
    state = input_interests_gr(interests, state)
    itinerary = create_itinerary_gr(state)
    return itinerary

def run_gradio():
    # The URL for the world map background
    world_map_url = 'https://mapsnworld.com/world-map-big-size.jpg'

    with gr.Blocks(css="""
        /* --- Main Page Styling --- */
        body { background-color: #1a1a2e !important; } /* Dark background for the whole page */
        .gradio-container {
            background-color: transparent !important;
            position: relative;
            z-index: 1;
        }

        /* --- Full-Screen Background Image --- */
        .gradio-container::before {
            content: '';
            position: fixed; /* Cover the entire browser window */
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-image: url('""" + world_map_url + """');
            background-size: cover;
            background-position: center;
            opacity: 0.2; /* KEY: Adjust opacity here (0.1 to 1.0) */
            z-index: -1; /* Place it behind all content */
        }

        /* --- Title Styling --- */
        .main-heading {
            font-size: 3em; font-weight: bold; color: #fa7436; text-align: center;
            padding: 1em 0; text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7);
        }

        /* --- "Frosted Glass" Input Area --- */
        .input-container {
            background-color: rgba(0, 0, 0, 0.3); /* Dark, semi-transparent background */
            backdrop-filter: blur(10px); /* This creates the frosted effect */
            -webkit-backdrop-filter: blur(10px); /* For Safari compatibility */
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2em; margin: 0 auto; max-width: 800px;
        }

        /* --- Input Field Labels & Text --- */
        .input-container .gr-label span {
            color: #f0f0f0 !important; font-weight: bold;
        }
        .input-container .gr-text-input input {
            background-color: rgba(0, 0, 0, 0.5) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
        }

        /* --- Button Styling --- */
        #generate-btn {
            background: #fa7436 !important; color: white !important; font-weight: bold;
            border: none !important; border-radius: 8px !important; margin-top: 1em;
            box-shadow: 0 4px 15px rgba(250, 116, 54, 0.4);
            transition: transform 0.2s ease-in-out;
        }
        #generate-btn:hover { transform: scale(1.05); }

        /* --- Output Card Styling --- */
        .generate-heading { display: none; } /* Hide the old "Generated Itinerary" text */
        .output-card {
            background-color: rgba(255, 255, 255, 0.98);
            color: #22223b !important;
            margin-top: 2em;
        }
        .output-card * { color: #22223b !important; }
    """) as demo:
        gr.HTML("<div class='main-heading'>Travel Planner</div>")
        
        with gr.Column(elem_classes="input-container"):
            initial_request = gr.Textbox(label="Enter your initial request (e.g., 'I want to plan a day trip')", scale=1)
            city = gr.Textbox(label="Please enter the city you want to visit for your day trip", scale=1)
            interests = gr.Textbox(label="Please enter your interest for the trip (comma-separated)", scale=1)

        btn = gr.Button("Generate Itinerary", elem_id="generate-btn", variant="primary")
        gr.HTML("<div class='generate-heading'>Generated Itinerary</div>")
        output = gr.Markdown(elem_classes="output-card")
        
        def on_submit(initial_request, city, interests):
            return create_itinerary_gr(state={
                "messages": [HumanMessage(content=initial_request)],
                "city": city,
                "interests": [i.strip() for i in interests.split(",")],
                "itinerary": "",
            })
        
        btn.click(on_submit, inputs=[initial_request, city, interests], outputs=output)
    demo.launch()
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        run_gradio()
    else:
        run_cli()
