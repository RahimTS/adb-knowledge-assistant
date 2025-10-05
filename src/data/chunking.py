from loguru import logger

from utils.config import settings


class TextChunker:
    """Chunk text documents for embedding"""

    def __init__(
        self, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: dict = None) -> list[dict]:
        """Split text into overlapping chunks"""
        chunks = []
        text = text.strip()

        if len(text) == 0:
            return chunks

        # Simple character-based chunking
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size * 0.5:
                    end = start + break_point + 1
                    chunk_text = text[start:end]

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_size": len(chunk_text),
                    "start_char": start,
                    "end_char": end,
                }
            )

            chunks.append({"text": chunk_text.strip(), "metadata": chunk_metadata})

            start = end - self.chunk_overlap
            chunk_id += 1

        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_json_knowledge(self, knowledge_entry: dict) -> list[dict]:
        """Chunk structured knowledge entries - handles multiple formats"""
        chunks = []

        # Determine entry type
        entry_type = knowledge_entry.get("type", "unknown")

        # If no type, infer from structure
        if entry_type == "unknown":
            if "command" in knowledge_entry:
                entry_type = "command"
            elif "issue" in knowledge_entry:
                entry_type = "troubleshooting"
            elif "url" in knowledge_entry:
                entry_type = "documentation"

        # Build comprehensive text representation based on entry type
        text_parts = []

        # Handle different entry formats
        if entry_type == "command":
            # ADB command format
            if "command" in knowledge_entry:
                text_parts.append(f"Command: {knowledge_entry['command']}")
            if "description" in knowledge_entry:
                text_parts.append(f"Description: {knowledge_entry['description']}")
            if "syntax" in knowledge_entry:
                text_parts.append(f"Syntax: {knowledge_entry['syntax']}")

            # Parameters
            if "parameters" in knowledge_entry and knowledge_entry["parameters"]:
                params_text = "Parameters:\n"
                for param in knowledge_entry["parameters"]:
                    if isinstance(param, dict):
                        for key, val in param.items():
                            params_text += f"  {key}: {val}\n"
                    else:
                        params_text += f"  {param}\n"
                text_parts.append(params_text)

            # Examples
            if "examples" in knowledge_entry and knowledge_entry["examples"]:
                examples_text = "Examples:\n"
                for ex in knowledge_entry["examples"]:
                    if isinstance(ex, dict):
                        examples_text += f"  {ex.get('command', '')}: {ex.get('explanation', '')}\n"
                text_parts.append(examples_text)

            # Common issues
            if "common_issues" in knowledge_entry and knowledge_entry["common_issues"]:
                issues_text = "Common Issues:\n"
                for issue in knowledge_entry["common_issues"]:
                    issues_text += f"  - {issue}\n"
                text_parts.append(issues_text)

        elif entry_type == "troubleshooting":
            # Troubleshooting format
            if "issue" in knowledge_entry:
                text_parts.append(f"Issue: {knowledge_entry['issue']}")
            if "symptoms" in knowledge_entry and knowledge_entry["symptoms"]:
                symptoms_text = "Symptoms:\n"
                for symptom in knowledge_entry["symptoms"]:
                    symptoms_text += f"  - {symptom}\n"
                text_parts.append(symptoms_text)
            if "solutions" in knowledge_entry and knowledge_entry["solutions"]:
                solutions_text = "Solutions:\n"
                for sol in knowledge_entry["solutions"]:
                    if isinstance(sol, dict):
                        solutions_text += f"  Step {sol.get('step', '')}: {sol.get('action', '')}\n"
                        if "details" in sol:
                            solutions_text += f"    Details: {sol['details']}\n"
                text_parts.append(solutions_text)

        elif entry_type == "documentation":
            # Documentation format
            if "title" in knowledge_entry:
                text_parts.append(f"Title: {knowledge_entry['title']}")
            if "url" in knowledge_entry:
                text_parts.append(f"URL: {knowledge_entry['url']}")
            if "content" in knowledge_entry:
                # Limit content length for docs
                content = knowledge_entry["content"][:5000]  # First 5000 chars
                text_parts.append(f"Content: {content}")

        else:
            # Generic format (for code_pattern, best_practice, etc.)
            if "title" in knowledge_entry:
                text_parts.append(f"Title: {knowledge_entry['title']}")
            if "name" in knowledge_entry:
                text_parts.append(f"Name: {knowledge_entry['name']}")
            if "operation" in knowledge_entry:
                text_parts.append(f"Operation: {knowledge_entry['operation']}")
            if "description" in knowledge_entry:
                text_parts.append(f"Description: {knowledge_entry['description']}")
            if "command" in knowledge_entry:
                text_parts.append(f"Command: {knowledge_entry['command']}")
            if "solution" in knowledge_entry:
                text_parts.append(f"Solution: {knowledge_entry['solution']}")
            if "implementation" in knowledge_entry:
                text_parts.append(f"Implementation: {knowledge_entry['implementation']}")
            if "python_code" in knowledge_entry:
                text_parts.append(f"Python Example:\n{knowledge_entry['python_code']}")
            if "steps" in knowledge_entry:
                steps_text = "Steps:\n"
                for i, step in enumerate(knowledge_entry["steps"]):
                    if isinstance(step, str):
                        steps_text += f"  {i + 1}. {step}\n"
                    elif isinstance(step, dict):
                        steps_text += f"  {step.get('step', i + 1)}. {step.get('action', '')}\n"
                text_parts.append(steps_text)

        full_text = "\n\n".join(text_parts)

        # Create metadata
        metadata = {
            "type": entry_type,
            "category": knowledge_entry.get("category", "general"),
            "source": knowledge_entry.get("source", "unknown"),
            "tags": knowledge_entry.get("tags", []),
        }

        # Add type-specific metadata
        if entry_type == "command":
            metadata["command"] = knowledge_entry.get("command", "")
        elif entry_type == "troubleshooting":
            metadata["issue"] = knowledge_entry.get("issue", "")
        elif entry_type == "error_pattern":
            metadata["error_indicator"] = knowledge_entry.get("error_indicator", "")
            metadata["severity"] = knowledge_entry.get("severity", "medium")
        elif entry_type == "code_pattern":
            metadata["operation"] = knowledge_entry.get("operation", "")

        # For most knowledge entries, create single chunk
        # Only chunk if text is very long
        if len(full_text) > self.chunk_size * 2:
            return self.chunk_text(full_text, metadata)
        else:
            return [{"text": full_text, "metadata": metadata}]
