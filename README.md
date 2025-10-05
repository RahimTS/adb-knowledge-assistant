# ADB Knowledge Assistant

A **multi-agent RAG system** for Android Debug Bridge (ADB) operations, combining production code patterns, official Android documentation, and curated troubleshooting knowledge using hybrid retrieval and specialized AI agents.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

- **Multi-Agent Architecture**: Specialized agents for different query types (commands, troubleshooting, code generation, concepts)
- **Hybrid RAG System**: Combines vector similarity search and keyword matching for optimal retrieval
- **Local Development**: Works with local MongoDB without Docker using scikit-learn cosine similarity
- **Production Code Patterns**: Real-world ADB operations extracted from production testing frameworks
- **OpenRouter Integration**: Powered by Claude 3.5 Sonnet via OpenRouter API
- **FastAPI Backend**: RESTful API with async support and comprehensive error handling
- **Comprehensive Knowledge Base**: 
  - 35+ production code patterns
  - 10+ ADB commands with examples
  - 4+ common troubleshooting scenarios
  - Official Android documentation


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- MongoDB (local installation)
- OpenRouter API Key ([Get here](https://openrouter.ai/keys))
- `uv` package manager ([Install](https://github.com/astral-sh/uv))

### Installation

1. **Clone the repository**
[adb-knowledge-assistant](https://github.com/RahimTS/adb-knowledge-assistant)

2. **Create virtual environment**
uv venv

3. **Install dependencies**
uv sync

4. **Configure environment**
cp .env.example .env

Edit .env and add your OPENROUTER_API_KEY

5. **Setup knowledge base**
python scripts/setup_system.py

6. **Start the server**
python src/main.py


Server will be running at `http://localhost:8000`

## 📖 Usage

### API Endpoints

**Health Check**
curl http://localhost:8000/health

**Query Knowledge Base**

curl -X POST "http://localhost:8000/query"
-H "Content-Type: application/json"
-d '{"query": "How do I pair a device wirelessly?"}'

**Get Statistics**
curl http://localhost:8000/stats

### Example Queries

**Command Lookup**
{
"query": "How do I list installed packages on Android?"
}

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI 0.115+ |
| **Package Manager** | uv |
| **Database** | MongoDB (local or Atlas) |
| **Vector Search** | scikit-learn (cosine similarity) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Claude 3.5 Sonnet (via OpenRouter) |
| **Agent Framework** | LangGraph 0.2+ |
| **LLM Integration** | LangChain 0.3+ |
| **Logging** | loguru |
| **Code Quality** | Ruff |


## ⚙️ Configuration

Key environment variables in `.env`:


## 📊 Knowledge Base

The system includes:

- **Production Code Patterns**: 35+ entries from real ADB handler implementations
  - Error patterns with solutions
  - Python code examples
  - Best practices
  - Common pitfalls
  - Complete workflows

- **ADB Commands**: 10+ commands with syntax, parameters, examples, and common issues

- **Troubleshooting**: 4+ curated solutions for connectivity, pairing, and file operation issues

- **Official Docs**: Android Developer documentation for ADB and testing tools

## 🎯 Use Cases

- **Android Developers**: Quick ADB command reference with examples
- **QA Engineers**: Troubleshoot device connectivity and testing issues
- **Automation Engineers**: Generate Python code for ADB operations
- **Team Knowledge Sharing**: Centralized repository of ADB expertise
- **Learning Resource**: Understand ADB concepts and best practices

## 🔄 Workflow Examples

### Wireless Debugging Setup
Query: *"How do I set up wireless debugging?"*

Returns complete step-by-step workflow with pairing and connection instructions.

### Error Diagnosis
Query: *"Connection refused error when connecting to device"*

Returns diagnosis, root cause, and multiple solution approaches with prevention tips.

### Code Generation
Query: *"Python code to upload image and update gallery"*

Returns production-ready Python code with error handling and comments.

## 🚧 Roadmap

- [ ] Add WebSocket support for streaming responses
- [ ] Implement conversation history and context tracking
- [ ] Add more specialized agents (performance, security)
- [ ] Support for additional knowledge sources (Stack Overflow, GitHub)
- [ ] Docker containerization
- [ ] MongoDB Atlas migration for production vector search
- [ ] Frontend UI with React/Next.js
- [ ] Rate limiting and authentication
- [ ] Export knowledge as documentation/cheatsheets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Format code (`ruff format src/`)
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- LLM inference via [OpenRouter](https://openrouter.ai/)
- Vector embeddings with [sentence-transformers](https://www.sbert.net/)

---

⭐ Star this repo if you find it helpful!

