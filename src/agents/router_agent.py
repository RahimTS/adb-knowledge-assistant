from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from utils.llm_factory import create_router_llm


class RouterAgent:
    """Route queries to appropriate specialist agent"""

    QUERY_TYPES = [
        "command_lookup",  # Looking for specific ADB command
        "troubleshooting",  # Debugging errors or issues
        "code_generation",  # Want Python code example
        "conceptual",  # Understanding concepts
        "workflow",  # Step-by-step process
    ]

    def __init__(self):
        self.llm = create_router_llm()

    def classify_query(self, query: str) -> dict:
        """Classify user query into category"""

        system_prompt = f"""You are a query classifier for an ADB/Android knowledge assistant.

Classify the user's query into ONE of these categories:
{", ".join(self.QUERY_TYPES)}

Respond with ONLY the category name and a brief reason.

Examples:
Query: "How do I list installed packages?"
Classification: command_lookup
Reason: User wants specific ADB command

Query: "Device shows as unauthorized"
Classification: troubleshooting
Reason: User has an error to fix

Query: "Show me Python code to push a file"
Classification: code_generation
Reason: User wants code example

Query: "What's the difference between pairing and connection ports?"
Classification: conceptual
Reason: User wants to understand a concept

Query: "How do I set up wireless debugging?"
Classification: workflow
Reason: User wants step-by-step process"""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Query: {query}")]

        response = self.llm.invoke(messages)

        # Parse response
        content = response.content.lower()

        # Find which category appears in response
        detected_type = "conceptual"  # default
        for query_type in self.QUERY_TYPES:
            if query_type in content:
                detected_type = query_type
                break

        logger.info(f"Query classified as: {detected_type}")

        return {"query_type": detected_type, "classification_reasoning": response.content}
