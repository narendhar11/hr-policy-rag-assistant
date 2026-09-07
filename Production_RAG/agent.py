"""Step 7: Create an agent for HR policy using the Groq LLM model and the defined tools."""
from langchain.agents import create_agent
from prompts.prompt_templates import HR_SYSTEM_PROMPT

def create_hr_policy_agent(llm, tools):
    """Create an agent for HR policy retrieval and return it."""
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=HR_SYSTEM_PROMPT
        )