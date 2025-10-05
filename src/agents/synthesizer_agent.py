from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from utils.llm_factory import create_synthesizer_llm


class SynthesizerAgent:
    """Synthesizes responses from specialist agents into final answer"""

    def __init__(self):
        self.llm = create_synthesizer_llm()

    def synthesize(
        self,
        query: str,
        query_type: str,
        agent_responses: dict[str, str],
        retrieved_docs: list[dict],
    ) -> str:
        """Synthesize final response from agent outputs"""

        logger.info(f"SynthesizerAgent synthesizing for query type: {query_type}")

        # Get the main agent response
        main_response = agent_responses.get(query_type, "")

        if not main_response:
            # Fallback to any available response
            main_response = next(iter(agent_responses.values()), "")

        # If we have a good response and it's comprehensive, return it
        if main_response and len(main_response) > 100:
            logger.info("Returning specialist agent response")
            return main_response

        # Otherwise, synthesize from available context
        system_prompt = """You are a synthesis agent that creates comprehensive, accurate answers.

Your role:
- Combine information from multiple sources
- Ensure accuracy and completeness
- Format answers clearly and professionally
- Include examples where appropriate
- Be concise but thorough

Create a well-structured response that directly answers the user's query."""

        # Build context from responses and docs
        context_parts = []

        if main_response:
            context_parts.append(f"Primary Response:\n{main_response}\n")

        # Add document context
        if retrieved_docs:
            context_parts.append("\nAdditional Context:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                content = doc.get("content", "")[:500]  # First 500 chars
                context_parts.append(f"{i}. {content}\n")

        context = "\n".join(context_parts)

        user_message = f"""User Query: {query}
Query Type: {query_type}

Available Information:
{context}

Synthesize a comprehensive answer to the user's query."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

        response = self.llm.invoke(messages)
        logger.info("SynthesizerAgent completed synthesis")

        return response.content
