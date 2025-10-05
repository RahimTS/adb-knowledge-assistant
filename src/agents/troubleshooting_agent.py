from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from utils.llm_factory import create_llm


class TroubleshootingAgent:
    """Specialized agent for debugging and problem-solving"""

    def __init__(self):
        self.llm = create_llm()

    def process(self, query: str, retrieved_docs: list[dict]) -> str:
        """Process troubleshooting queries"""

        logger.info(f"TroubleshootingAgent processing: {query[:100]}...")

        # Format retrieved documents
        context = self._format_context(retrieved_docs)

        system_prompt = """You are an expert ADB troubleshooter and problem solver.

Your role:
- Diagnose ADB and Android connectivity issues
- Provide step-by-step solutions
- Explain error messages clearly
- Suggest multiple approaches when applicable
- Prioritize solutions by likelihood of success

Format your response as:
1. Problem diagnosis
2. Root cause explanation
3. Step-by-step solution
4. Alternative solutions (if applicable)
5. Prevention tips

Use the retrieved context to provide accurate troubleshooting steps."""

        user_message = f"""Problem/Error: {query}

Retrieved Context (similar issues and solutions):
{context}

Provide a clear diagnosis and step-by-step solution."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

        response = self.llm.invoke(messages)
        logger.info("TroubleshootingAgent completed processing")

        return response.content

    def _format_context(self, retrieved_docs: list[dict]) -> str:
        """Format retrieved documents as context"""
        if not retrieved_docs:
            return "No similar issues found in knowledge base."

        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)

            context_parts.append(f"[Solution {i}] (Relevance: {score:.2f})")

            # Highlight error patterns
            if metadata.get("type") == "error_pattern":
                error_indicator = metadata.get("error_indicator", "")
                severity = metadata.get("severity", "medium")
                context_parts.append(f"Error Pattern: {error_indicator} (Severity: {severity})")

            context_parts.append(f"Content:\n{content}\n")

        return "\n".join(context_parts)
