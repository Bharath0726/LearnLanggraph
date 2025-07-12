from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# Try different model providers in order of preference
def get_model():
    # Option 1: Try Groq (free alternative) first
    try:
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            model = ChatGroq(model="llama3-8b-8192", temperature=0)
            print("Using Groq llama3-8b-8192")
            return model
        else:
            print("GROQ_API_KEY not found in environment")
    except ImportError:
        print("langchain_groq not installed")
    except Exception as e:
        print(f"Groq not available: {e}")
    
    # Option 2: Try OpenAI (requires payment and may not work)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            # Try different OpenAI models
            for model_name in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]:
                try:
                    model = ChatOpenAI(model=model_name, temperature=0)
                    print(f"Using OpenAI {model_name}")
                    return model
                except Exception as e:
                    print(f"OpenAI {model_name} not available: {e}")
                    continue
        except Exception as e:
            print(f"OpenAI setup failed: {e}")
    else:
        print("OPENAI_API_KEY not found in environment")
    
    # Option 3: Fallback to a mock model for testing
    from langchain_core.language_models.fake import FakeListLLM
    print("Using fake model for testing purposes - all paid models failed")
    return FakeListLLM(responses=[
        "This is a test response from a fake model.", 
        "I can help you with basic tasks.",
        "The result is 42.",
        "I'm a fallback model used when real models aren't available."
    ])

# Initialize with a safe fallback
model = get_model()

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        # Get model at runtime to handle API errors gracefully
        try:
            current_model = get_model()
            return {"messages":[current_model.invoke(state['messages'])]}
        except Exception as e:
            print(f"Model call failed: {e}")
            # Return a fallback response
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="I'm sorry, I'm currently unable to process your request due to model access limitations.")]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent=graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    def call_model(state):
        # Get model at runtime to handle API errors gracefully
        try:
            current_model = get_model()
            model_with_tools = current_model.bind_tools([add])
            return {"messages": [model_with_tools.invoke(state["messages"])]}
        except Exception as e:
            print(f"Model call failed: {e}")
            # Return a fallback response
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content="I'm sorry, I'm currently unable to process your request due to model access limitations.")]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

agent=make_alternative_graph()

