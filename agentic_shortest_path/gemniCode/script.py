import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
import logging
from typing import TypedDict, List, Dict, Any, Tuple, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# Setup logging configuration for production visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 1. STATE DEFINITION LAYER
# ============================================================================

class GraphState(TypedDict):
    """
    Production state schema representing the exact context configuration
    passed between nodes during execution.
    """
    graph_dict: Dict[str, List[str]]
    current_node: str
    target_node: str
    path: List[str]
    visited_edges: List[Tuple[str, str]]
    thought_process: str
    status: str  # "searching" | "found" | "failed" | "error"
    error_message: Optional[str]
    step_count: int

# ============================================================================
# 2. LANGGRAPH ENGINE LAYER
# ============================================================================

def agent_decision_node(state: GraphState) -> Dict[str, Any]:
    """
    LLM node that presents the local environment to the model and prompts
    for the next heuristic move or a logical backtrack step.
    """
    current: str = state["current_node"]
    target: str = state["target_node"]
    path: List[str] = state["path"]
    neighbors: List[str] = state["graph_dict"].get(current, [])
    
    # Initialize connection to localized Ollama instance
    try:
        llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.0, timeout=10.0)
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Ollama connection timed out or is unavailable. Details: {str(e)}",
            "thought_process": "System Error: Model connection lost."
        }

    prompt = PromptTemplate.from_template(
        """You are an advanced pathfinding agent traversing a graph structure to find the shortest path.
        Target Destination Node: {target}
        Your Current Position: {current}
        Path History Taken: {path}
        Direct Available Neighbors from current position: {neighbors}
        
        Strict Operational Protocol:
        1. Select exactly ONE node identifier from the Available Neighbors to advance.
        2. If you are stuck, hit a dead end, or determine that your path history is sub-optimal, output "BACK" to return to the previous node in your history.
        3. Do not include markdown formatting, pleasantries, punctuation, or thinking explanations.
        4. Output strictly the node identifier or the word "BACK".
        
        Decision:"""
    )
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "target": target,
            "current": current,
            "path": path,
            "neighbors": neighbors
        })
        
        decision: str = response.content.strip()
        # Canonicalize output
        if decision.upper() == "BACK":
            decision = "BACK"
            
        return {
            "thought_process": f"Evaluated node {current} with neighbors {neighbors}. Selected Action: {decision}",
            "step_count": state["step_count"] + 1
        }
    except Exception as e:
        logger.error(f"Inference failure: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Inference execution failed: {str(e)}",
            "thought_process": "System Error: Inference failure."
        }

def execute_move_node(state: GraphState) -> Dict[str, Any]:
    """
    State transformation node that validates LLM output against the actual graph
    topology and structurally alters state variables.
    """
    if state["status"] == "error":
        return {}

    # Extract clean decision token from the thought log
    try:
        decision: str = state["thought_process"].split("Selected Action: ")[-1].strip()
    except IndexError:
        decision = "BACK"

    current: str = state["current_node"]
    path: List[str] = list(state["path"])
    visited_edges: List[Tuple[str, str]] = list(state["visited_edges"])
    graph_dict: Dict[str, List[str]] = state["graph_dict"]
    
    # Check if the model opted to backtrack
    if decision == "BACK":
        if len(path) > 1:
            path.pop()  # Drop current leaf node
            next_node = path[-1]  # Return to parent node
            return {
                "current_node": next_node,
                "path": path,
                "status": "searching",
                "thought_process": f"Backtracked successfully to node: {next_node}"
            }
        else:
            return {
                "status": "failed",
                "error_message": "Agent requested backtrack from the initial starting node.",
                "thought_process": "Failed: Terminal backtracking limit reached."
            }
            
    # Check if the node choice matches real graph topology
    valid_neighbors = graph_dict.get(current, [])
    if decision in valid_neighbors:
        next_node = decision
        visited_edges.append((current, next_node))
        path.append(next_node)
        
        status = "found" if next_node == state["target_node"] else "searching"
        return {
            "current_node": next_node,
            "path": path,
            "visited_edges": visited_edges,
            "status": status
        }
    else:
        # Graceful fallback handler for hallucinated nodes
        return {
            "status": "searching",
            "thought_process": f"Invalid move '{decision}' attempted. Forcing algorithmic cycle recovery.",
            "step_count": state["step_count"] + 1
        }

def runtime_router(state: GraphState) -> str:
    """
    Conditional routing agent checking exit criteria or infinite-loop prevention.
    """
    if state["status"] in ["found", "failed", "error"]:
        return END
    # Hard stop safety boundary to prevent infinite billing/compute loop scenarios
    if state["step_count"] >= 25:
        logger.warning("Agent execution suspended: step ceiling threshold violated.")
        state["status"] = "failed"
        return END
    return "agent_decision"

def compile_agentic_workflow() -> Any:
    """
    Compiles individual functional components into a deterministic DAG.
    """
    builder = StateGraph(GraphState)
    builder.add_node("agent_decision", agent_decision_node)
    builder.add_node("execute_move", execute_move_node)
    
    builder.set_entry_point("agent_decision")
    builder.add_edge("agent_decision", "execute_move")
    builder.add_conditional_edges("execute_move", runtime_router)
    
    return builder.compile()

# ============================================================================
# 3. PRODUCTION UX/UI LAYER (STREAMLIT)
# ============================================================================

def generate_production_graph() -> Dict[str, List[str]]:
    """ Returns a structured graph network with structural layout alternatives. """
    return {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B", "H"],
        "E": ["B", "I"],
        "F": ["C", "J"],
        "G": ["C", "K"],
        "H": ["D", "L"],
        "I": ["E", "M"],
        "J": ["F", "N"],
        "K": ["G", "N"],
        "L": ["H", "O"],
        "M": ["I", "O"],
        "N": ["J", "K", "O"],
        "O": ["L", "M", "N"]
    }

def render_visualization_frame(graph_dict: Dict[str, List[str]], current_node: str, path: List[str], start: str, target: str):
    """
    Constructs high-performance static visualization plot of the runtime system state.
    """
    G = nx.Graph(graph_dict)
    # Use a layout with fixed random states for coordinate invariance during ticks
    pos = nx.kamada_kawai_layout(G)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Draw standard edges
    nx.draw_networkx_edges(G, pos, edge_color="#CBD5E1", width=1.5, ax=ax, alpha=0.6)
    
    # Highlight path vectors traversed
    if len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="#3B82F6", width=3.0, ax=ax)

    # Resolve color mappings per node role
    colors = []
    for node in G.nodes():
        if node == current_node:
            colors.append("#F59E0B")  # Amber/Orange for Active Agent Location
        elif node == start:
            colors.append("#10B981")  # Emerald/Green for Source
        elif node == target:
            colors.append("#EF4444")  # Rose/Red for Destination Goal
        elif node in path:
            colors.append("#93C5FD")  # Light Blue for Visited History Line
        else:
            colors.append("#F1F5F9")  # Slate for inactive infrastructure
            
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, edgecolors="#475569", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_family="sans-serif", font_weight="bold", font_color="#1E293B", ax=ax)
    
    ax.axis("off")
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Enterprise Agentic Path Finder", page_icon="🕸️")
    
    st.markdown("""
        <style>
            .reportview-container { background: #FAFBFD; }
            .stMetric { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        </style>
    """, unsafe_allow_html=True)

    st.title("🕸️ White-Box LangGraph Path Optimization Engine")
    st.caption("Production system tracking state mutation, backtracking heuristics, and active graph space exploration utilizing Ollama (Qwen2.5).")
    st.divider()

    # Session persistence infrastructure management
    if "running" not in st.session_state:
        st.session_state.running = False
    
    graph_data = generate_production_graph()
    nodes_list = sorted(list(graph_data.keys()))

    # Layout Columns splitting interface controls and monitoring console
    control_col, display_col = st.columns([1, 2], gap="large")

    with control_col:
        st.subheader("⚙️ Control Panel Configuration")
        start_node = st.selectbox("Source Vertex (Start)", nodes_list, index=0, disabled=st.session_state.running)
        target_node = st.selectbox("Target Destination (End)", nodes_list, index=14, disabled=st.session_state.running)
        
        execution_speed = st.slider("Step Refresh Intermission Time (s)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        
        st.write("---")
        run_triggered = st.button("Initialize Agent Solver", type="primary", use_container_width=True, disabled=st.session_state.running)

    with display_col:
        st.subheader("📊 Dynamic Environment Stream")
        plot_frame = st.empty()
        
        # Performance matrix monitoring layout block
        m_col1, m_col2, m_col3 = st.columns(3)
        metric_status = m_col1.empty()
        metric_steps = m_col2.empty()
        metric_path_len = m_col3.empty()
        
        st.subheader("📝 White-Box System Execution Ledger")
        log_stream_container = st.empty()

    # Render static baseline map configuration on load
    initial_fig = render_visualization_frame(graph_data, start_node, [start_node], start_node, target_node)
    plot_frame.pyplot(initial_fig)
    plt.close(initial_fig)
    
    metric_status.metric("System State", "Idle", delta=None)
    metric_steps.metric("Evaluations Count", "0")
    metric_path_len.metric("Active Path Vector Length", "1")
    log_stream_container.info("System primed. Select network execution variables to launch solver.")

    if run_triggered:
        st.session_state.running = True
        workflow_engine = compile_agentic_workflow()
        
        # State Initialization payload
        runtime_state: GraphState = {
            "graph_dict": graph_data,
            "current_node": start_node,
            "target_node": target_node,
            "path": [start_node],
            "visited_edges": [],
            "thought_process": "Initializing runtime execution environment graph context...",
            "status": "searching",
            "error_message": None,
            "step_count": 0
        }

        execution_logs: List[str] = []
        
        # Step stream iteration process loop
        for update in workflow_engine.stream(runtime_state):
            # Capture the updated node's modifications to state variables
            for node_key, updated_attributes in update.items():
                runtime_state.update(updated_attributes)
                
                # Immediate UI Context updates
                current_active = runtime_state.get("current_node", start_node)
                current_path = runtime_state.get("path", [start_node])
                current_status = runtime_state.get("status", "searching").upper()
                
                # Re-render dynamic image matrix map
                loop_fig = render_visualization_frame(graph_data, current_active, current_path, start_node, target_node)
                plot_frame.pyplot(loop_fig)
                plt.close(loop_fig)
                
                # Update visual analytical indicators
                metric_status.metric("System State", current_status, 
                                     delta="Active Run" if current_status == "SEARCHING" else "Settled")
                metric_steps.metric("Evaluations Count", str(runtime_state.get("step_count", 0)))
                metric_path_len.metric("Active Path Vector Length", str(len(current_path)))
                
                # Handle text stream processing
                thought = runtime_state.get("thought_process", "")
                if thought:
                    execution_logs.append(f"`[{node_key}]` {thought}")
                    formatted_logs = "\n\n".join(execution_logs[::-1]) # Keep most recent logs at the top
                    log_stream_container.markdown(formatted_logs)
                
                # Check for runtime error generation breaks
                if runtime_state.get("status") == "error":
                    st.error(runtime_state.get("error_message", "Unknown internal engine error."))
                    st.session_state.running = False
                    return

                time.sleep(execution_speed)

        # Post-execution operational state handling
        final_status = runtime_state.get("status")
        if final_status == "found":
            st.balloons()
            st.success(f"🎯 Optimization Goal Reached! Final Complete Route Path: {' ➡️ '.join(runtime_state['path'])}")
        elif final_status == "failed":
            st.error(f"⚠️ Traversal path optimization failed. Reason: {runtime_state.get('error_message', 'Maximum search step cutoff exceeded without finding target path.')}")
            
        st.session_state.running = False

if __name__ == "__main__":
    main()