
from pprint import pprint
from typing import List, Union
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import tool
# from langchain.chat_models.openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
# from langchain.agents.react.output_parser import ReActOutputParser


# Note: agent.invoke() is a while loop that looks for a tool to use, use it, and stops when LLM finds answer or limiting iterations.  


# decorator to register a function as a tool
# doc strings (get_text_length.description) are used to describe the tool and helps llm decide which tool to use
# tool has: get_text_length.name: str, get_text_length.description: str, return_direct: bool set to False, optional args_schema: Pydantic BaseModel
# @tool is a decorator that places it in langchain tool class (registers a function as a tool).
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}") # prints tool name and argument
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length] # list of tools for  reasoning agent.
# This prompt is an implemenation of the react paper. A few shot prompt.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: 
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), # converts the tools name and description ,which is an Object, to a string because LLM can't understand objects, only text
        tool_names=", ".join([t.name for t in tools]), # the names of the tools concatenated and separated by commas
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"]) 
    # Reasoning agent
    # LLM stops generating text when it outputed 'backslash n' Observation token else. Comment out ReActSingleInputOutputParser to see '\n'
    # it will keep generating text until the max tokens is reached and Observation is a result of running the tool.
    # Then, it may cause the LLM to halucinate.

    intermediate_steps = []
    # Dict | prompt | llm is a small pipeline. result of prompt is passed to llm then passed to ReActSingleInputOutputParser()
    # see "https://python.langchain.com/docs/expression_language/interface" for the vals that can be passed thru pipeline.
    agent = (
        { # Dict is lambda to access the input key of the dict (agent_step) and returns it. (agent_step["input"]
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser() 
        # parses ReAct-style LLM calls that have a single tool input. So, output of LLM format is parsed.
        # expects a '/n' input. If not working, try modifying prompt to be shorter and essential words only.
        # Expects below:
        # Thought: agent thought here
        # Action: search or select a tool to use or get_text_length()
        # Action Input: dog
        # Parse returns AgentAction or AgentFinish. AgentAction is obj that holds tool name and tool input (supplied after parsed) which
        # is used to invoke the tool that LLM selected from reasoning agent. 
        # regex is used to extract 'action' and 'action input', then if answer not included we can set: 'action', 'action_input', 'tool_input'.
        # return AgentAction(action, tool_input, text). text is from LLM output.
        )

    agent_step: Union[AgentAction, AgentAction] = agent.invoke(
        {
            "input": "What is the length in characters of the text DOG ?",
        }
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction): # holds info of tool we want to run and get observation from.
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input)) # func is expecting string.
        print(f"{observation=}")
