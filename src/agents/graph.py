from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger

from agents.code_generator_agent import CodeGeneratorAgent
from agents.command_expert_agent import CommandExpertAgent
from agents.router_agent import RouterAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.troubleshooting_agent import TroubleshootingAgent


class AgentState(TypedDict):
    """State passed between agents"""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    query_type: str
    retrieved_docs: list
    agent_responses: dict
    final_answer: str


class ADBAgentGraph:
    """Multi-agent system for ADB knowledge"""

    def __init__(self):
        self.router = RouterAgent()
        self.command_expert = CommandExpertAgent()
        self.troubleshooting_agent = TroubleshootingAgent()
        self.code_generator = CodeGeneratorAgent()
        self.synthesizer = SynthesizerAgent()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph"""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("command_expert", self._command_expert_node)
        workflow.add_node("troubleshooting", self._troubleshooting_node)
        workflow.add_node("code_generator", self._code_generator_node)
        workflow.add_node("conceptual", self._conceptual_node)
        workflow.add_node("synthesizer", self._synthesizer_node)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional routing from router
        workflow.add_conditional_edges(
            "router",
            self._route_to_specialist,
            {
                "command_lookup": "command_expert",
                "troubleshooting": "troubleshooting",
                "code_generation": "code_generator",
                "conceptual": "conceptual",
                "workflow": "command_expert",
            },
        )

        # All specialist agents go to synthesizer
        workflow.add_edge("command_expert", "synthesizer")
        workflow.add_edge("troubleshooting", "synthesizer")
        workflow.add_edge("code_generator", "synthesizer")
        workflow.add_edge("conceptual", "synthesizer")

        # Synthesizer is the end
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def _route_query(self, state: AgentState) -> AgentState:
        """Router node"""
        query = state["query"]
        classification = self.router.classify_query(query)
        state["query_type"] = classification["query_type"]
        logger.info(f"Routed to: {classification['query_type']}")
        return state

    def _route_to_specialist(self, state: AgentState) -> str:
        """Determine which specialist to use"""
        return state["query_type"]

    def _command_expert_node(self, state: AgentState) -> AgentState:
        """Command expert processing"""
        response = self.command_expert.process(state["query"], state.get("retrieved_docs", []))
        state["agent_responses"]["command_expert"] = response
        return state

    def _troubleshooting_node(self, state: AgentState) -> AgentState:
        """Troubleshooting processing"""
        response = self.troubleshooting_agent.process(
            state["query"], state.get("retrieved_docs", [])
        )
        state["agent_responses"]["troubleshooting"] = response
        return state

    def _code_generator_node(self, state: AgentState) -> AgentState:
        """Code generation processing"""
        response = self.code_generator.process(state["query"], state.get("retrieved_docs", []))
        state["agent_responses"]["code_generator"] = response
        return state

    def _conceptual_node(self, state: AgentState) -> AgentState:
        """Conceptual explanation processing"""
        response = self.command_expert.process(state["query"], state.get("retrieved_docs", []))
        state["agent_responses"]["conceptual"] = response
        return state

    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final response"""
        final_answer = self.synthesizer.synthesize(
            query=state["query"],
            query_type=state["query_type"],
            agent_responses=state["agent_responses"],
            retrieved_docs=state.get("retrieved_docs", []),
        )
        state["final_answer"] = final_answer
        return state

    def query(self, user_query: str, retrieved_docs: list = None) -> str:
        """Process a user query through the agent graph"""

        initial_state = {
            "query": user_query,
            "query_type": "",
            "retrieved_docs": retrieved_docs or [],
            "agent_responses": {},
            "final_answer": "",
            "messages": [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state["final_answer"]
