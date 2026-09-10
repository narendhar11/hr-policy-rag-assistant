"""Step 7: Create an agent for HR policy using the Groq LLM model and the defined tools."""
from langchain.agents import create_agent
from prompts.prompt_templates import HR_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

def create_hr_policy_agent(llm, tools):
    """Create an agent for HR policy retrieval and return it."""
    logger.info("Creating HR policy agent with LLM model '%s' and %d tools", llm.model, len(tools))
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=HR_SYSTEM_PROMPT
        )
    logger.info("HR policy agent created successfully")
    return agent