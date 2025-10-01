from dataclasses import dataclass
from typing import List, Optional
import re
import textwrap
from cot_reasoning import VisualizationConfig

@dataclass
class BSNode:
    """Data class representing a node in the Beam Search tree"""
    id: str
    content: str
    score: float
    parent_id: Optional[str] = None
    children: List['BSNode'] = None
    is_best_path: bool = False
    path_score: Optional[float] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class BSResponse:
    """Data class representing a complete Beam Search response"""
    question: str
    root: BSNode
    answer: Optional[str] = None
    best_score: Optional[float] = None
    result_nodes: List[BSNode] = None

    def __post_init__(self):
        if self.result_nodes is None:
            self.result_nodes = []

def parse_bs_response(response_text: str, question: str) -> BSResponse:
    """Parse Beam Search response text to extract nodes and build the tree"""
    # Parse nodes
    node_pattern = r'<node id="([^"]+)"(?:\s+parent="([^"]+)")?\s*score="([^"]+)"(?:\s+path_score="([^"]+)")?\s*>\s*(.*?)\s*</node>'
    nodes_dict = {}
    result_nodes = []
    
    # First pass: create all nodes
    for match in re.finditer(node_pattern, response_text, re.DOTALL):
        node_id = match.group(1)
        parent_id = match.group(2)
        score = float(match.group(3))
        path_score = float(match.group(4)) if match.group(4) else None
        content = match.group(5).strip()
        
        node = BSNode(
            id=node_id, 
            content=content, 
            score=score, 
            parent_id=parent_id,
            path_score=path_score
        )
        nodes_dict[node_id] = node
        
        # Collect result nodes
        if node_id.startswith('result'):
            result_nodes.append(node)

    # Second pass: build tree relationships
    root = None
    for node in nodes_dict.values():
        if node.parent_id is None:
            root = node
        else:
            parent = nodes_dict.get(node.parent_id)
            if parent:
                parent.children.append(node)

    # Parse answer if present
    answer_pattern = r'<answer>\s*Best path \(path_score: ([^\)]+)\):\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
    answer = None
    best_score = None
    
    if answer_match:
        best_score = float(answer_match.group(1))
        answer = answer_match.group(2).strip()
        
        # Mark the best path based on path_score
        current_path_score = best_score
        for node in nodes_dict.values():
            if node.path_score and abs(node.path_score - current_path_score) < 1e-6:
                # Mark all nodes in the path as best
                current = node
                while current:
                    current.is_best_path = True
                    current = nodes_dict.get(current.parent_id)

    return BSResponse(
        question=question, 
        root=root, 
        answer=answer, 
        best_score=best_score,
        result_nodes=result_nodes
    )

def create_mermaid_diagram(bs_response: BSResponse, config: VisualizationConfig) -> str:
    """Convert Beam Search response to Mermaid diagram"""
    diagram = ['graph TD']
    
    # Add question node
    question_content = wrap_text(bs_response.question, config)
    diagram.append(f'    Q["{question_content}"]')
    
    def add_node_and_children(node: BSNode, parent_id: Optional[str] = None):
        # Format content to include scores
        score_info = f"Score: {node.score:.2f}"
        if node.path_score:
            score_info += f"<br>Path Score: {node.path_score:.2f}"
        node_content = f"{wrap_text(node.content, config)}<br>{score_info}"
        
        # Determine node style based on type and path
        if node.id.startswith('result'):
            node_style = 'result'
            if node.is_best_path:
                node_style = 'best_result'
        else:
            node_style = 'intermediate'
            if node.is_best_path:
                node_style = 'best_intermediate'
        
        # Add node
        diagram.append(f'    {node.id}["{node_content}"]')
        diagram.append(f'    class {node.id} {node_style};')
        
        # Add connection from parent
        if parent_id:
            diagram.append(f'    {parent_id} --> {node.id}')
        
        # Process children
        for child in node.children:
            add_node_and_children(child, node.id)
    
    # Build tree structure
    if bs_response.root:
        diagram.append(f'    Q --> {bs_response.root.id}')
        add_node_and_children(bs_response.root)
    
    # Add final answer
    if bs_response.answer:
        answer_content = wrap_text(
            f"Final Answer (Path Score: {bs_response.best_score:.2f}):<br>{bs_response.answer}",
            config
        )
        diagram.append(f'    Answer["{answer_content}"]')
        
        # Connect all result nodes to the answer
        for result_node in bs_response.result_nodes:
            diagram.append(f'    {result_node.id} --> Answer')
        
        diagram.append('    class Answer final_answer;')
    
    # Add styles
    diagram.extend([
        '    classDef intermediate fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef best_intermediate fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef result fill:#f3f4f6,stroke:#4b5563,stroke-width:2px;',
        '    classDef best_result fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px;',
        '    classDef final_answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    class Q question;',
        '    linkStyle default stroke:#666,stroke-width:2px;'
    ])
    
    return '\n'.join(diagram)

def wrap_text(text: str, config: VisualizationConfig) -> str:
    """Wrap text to fit within box constraints"""
    text = text.replace('\n', ' ').replace('"', "'")
    wrapped_lines = textwrap.wrap(text, width=config.max_chars_per_line)
    
    if len(wrapped_lines) > config.max_lines:
        # Option 1: Simply truncate and add ellipsis to the last line
        wrapped_lines = wrapped_lines[:config.max_lines]
        wrapped_lines[-1] = wrapped_lines[-1][:config.max_chars_per_line-3] + "..."
        
        # Option 2 (alternative): Include part of the next line to show continuity
        # original_next_line = wrapped_lines[config.max_lines] if len(wrapped_lines) > config.max_lines else ""
        # wrapped_lines = wrapped_lines[:config.max_lines-1]
        # wrapped_lines.append(original_next_line[:config.max_chars_per_line-3] + "...")
    
    return "<br>".join(wrapped_lines)
