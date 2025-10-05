from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from utils.llm_factory import create_generator_llm


class CodeGeneratorAgent:
    """Specialized agent for generating Python code for ADB operations"""

    def __init__(self):
        self.llm = create_generator_llm()

    def process(self, query: str, retrieved_docs: list[dict]) -> str:
        """Process code generation queries"""

        logger.info(f"CodeGeneratorAgent processing: {query[:100]}...")

        # Format retrieved documents
        context = self._format_context(retrieved_docs)

        system_prompt = """You are an expert Python developer specializing in ADB automation.

Your role:
- Generate clean, production-ready Python code for ADB operations
- Include error handling and validation
- Add helpful comments
- Follow best practices from the retrieved examples
- Use subprocess or similar libraries for ADB commands

Format your response:
1. Brief explanation of the approach
2. Complete, runnable Python code
3. Usage example
4. Important notes or warnings

Use the retrieved code patterns as reference for style and best practices."""

        user_message = f"""Request: {query}

Retrieved Code Examples and Patterns:
{context}

Generate clean Python code that accomplishes this task."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

        response = self.llm.invoke(messages)
        logger.info("CodeGeneratorAgent completed processing")

        return response.content

    def _format_context(self, retrieved_docs: list[dict]) -> str:
        """Format retrieved documents as context"""
        if not retrieved_docs:
            return "No code examples found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)

            context_parts.append(f"[Code Example {i}] (Relevance: {score:.2f})")

            # Highlight code patterns
            if metadata.get("type") == "code_pattern":
                operation = metadata.get("operation", "")
                context_parts.append(f"Operation: {operation}")

            context_parts.append(f"Reference Code:\n{content}\n")

        return "\n".join(context_parts)
