from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from utils.llm_factory import create_llm


class CommandExpertAgent:
    """Specialized agent for ADB command lookup and explanation"""

    def __init__(self):
        self.llm = create_llm()

    def process(self, query: str, retrieved_docs: list[dict]) -> str:
        """Process command-related queries"""

        logger.info(f"CommandExpertAgent processing: {query[:100]}...")

        # Format retrieved documents
        context = self._format_context(retrieved_docs)

        system_prompt = """You are an expert in Android Debug Bridge (ADB) commands.

Your role:
- Provide accurate ADB command syntax and usage
- Explain command parameters and options
- Give practical examples
- Mention related commands
- Warn about common pitfalls

Format your response clearly with:
1. Command syntax
2. Description
3. Parameters/options (if any)
4. Examples
5. Common issues (if applicable)

Use the retrieved context to provide accurate information."""

        user_message = f"""Query: {query}

Retrieved Context:
{context}

Provide a comprehensive answer about the ADB command(s) relevant to this query."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

        response = self.llm.invoke(messages)
        logger.info("CommandExpertAgent completed processing")

        return response.content

    def _format_context(self, retrieved_docs: list[dict]) -> str:
        """Format retrieved documents as context"""
        if not retrieved_docs:
            return "No relevant documentation found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)

            context_parts.append(f"[Document {i}] (Relevance: {score:.2f})")
            context_parts.append(f"Type: {metadata.get('type', 'unknown')}")
            context_parts.append(f"Category: {metadata.get('category', 'unknown')}")
            context_parts.append(f"Content:\n{content}\n")

        return "\n".join(context_parts)
