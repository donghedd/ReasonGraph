from dataclasses import dataclass
from typing import List, Optional
import re
import textwrap
from cot_reasoning import VisualizationConfig, AnthropicAPI

@dataclass
class ToTNode:
    """Data class representing a node in the Tree of Thoughts"""
    id: str
    content: str
    parent_id: Optional[str] = None
    children: List['ToTNode'] = None
    is_answer: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class ToTResponse:
    """Data class representing a complete ToT response"""
    question: str
    root: ToTNode
    answer: Optional[str] = None

def parse_tot_response(response_text: str, question: str) -> ToTResponse:
    """Parse ToT response text to extract nodes and build the tree"""
    # Parse nodes
    node_pattern = r'<node id="([^"]+)"(?:\s+parent="([^"]+)")?\s*>\s*(.*?)\s*</node>'
    nodes_dict = {}
    
    # First pass: create all nodes
    for match in re.finditer(node_pattern, response_text, re.DOTALL):
        node_id = match.group(1)
        parent_id = match.group(2)
        content = match.group(3).strip()
        
        node = ToTNode(id=node_id, content=content, parent_id=parent_id)
        nodes_dict[node_id] = node

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
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    
    if answer:
        # Mark the node leading to the answer
        for node in nodes_dict.values():
            if node.content.strip() in answer.strip():
                node.is_answer = True

    return ToTResponse(question=question, root=root, answer=answer)

def create_mermaid_diagram(tot_response: ToTResponse, config: VisualizationConfig) -> str:
    """Convert ToT response to Mermaid diagram"""
    diagram = ['<div class="mermaid">', 'graph TD']
    
    # Add question node
    question_content = wrap_text(tot_response.question, config)
    diagram.append(f'    Q["{question_content}"]')
    
    # Track leaf nodes for connecting to answer
    leaf_nodes = []
    
    def add_node_and_children(node: ToTNode, parent_id: Optional[str] = None):
        content = wrap_text(node.content, config)
        node_style = 'answer' if node.is_answer else 'default'
        
        # Add node
        diagram.append(f'    {node.id}["{content}"]')
        
        # Add connection from parent
        if parent_id:
            diagram.append(f'    {parent_id} --> {node.id}')
        
        # Process children
        if node.children:
            for child in node.children:
                add_node_and_children(child, node.id)
        else:
            # This is a leaf node
            leaf_nodes.append(node.id)
    
    # Build tree structure
    if tot_response.root:
        diagram.append(f'    Q --> {tot_response.root.id}')
        add_node_and_children(tot_response.root)
    
    # Add final answer node if answer exists
    if tot_response.answer:
        answer_content = wrap_text(tot_response.answer, config)
        diagram.append(f'    Answer["{answer_content}"]')
        # Connect all leaf nodes to the answer
        for leaf_id in leaf_nodes:
            diagram.append(f'    {leaf_id} --> Answer')
        diagram.append('    class Answer final_answer;')
    
    # Add styles
    diagram.extend([
        '    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    classDef final_answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    class Q question;',
        '    linkStyle default stroke:#666,stroke-width:2px;'
    ])
    
    diagram.append('</div>')
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