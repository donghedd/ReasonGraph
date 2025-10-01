import re
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import Counter
from cot_reasoning import CoTStep, CoTResponse, VisualizationConfig, wrap_text

@dataclass
class SCRPath:
    """Data class representing a single self-consistency reasoning path"""
    path_id: int
    steps: List[CoTStep]
    answer: Optional[str] = None

@dataclass
class SCRResponse:
    """Data class representing a complete self-consistency response"""
    question: str
    paths: List[SCRPath]
    final_answer: Optional[str] = None
    vote_counts: Optional[Dict[str, int]] = None

def parse_scr_response(response_text: str, question: str, num_paths: int = 5) -> SCRResponse:
    """
    Parse self-consistency response text to extract multiple reasoning paths and answers.
    
    Args:
        response_text: The raw response from the API containing multiple paths
        question: The original question
        num_paths: Expected number of reasoning paths
    
    Returns:
        SCRResponse object containing all paths and aggregated answer
    """
    # Split the response into individual paths
    path_pattern = r'Path\s+(\d+):(.*?)(?=Path\s+\d+:|$)'
    path_matches = re.finditer(path_pattern, response_text, re.DOTALL)
    
    paths = []
    answers = []
    
    for match in path_matches:
        path_id = int(match.group(1))
        path_content = match.group(2).strip()
        
        # Extract steps for this path
        step_pattern = r'<step number="(\d+)">\s*(.*?)\s*</step>'
        steps = []
        for step_match in re.finditer(step_pattern, path_content, re.DOTALL):
            number = int(step_match.group(1))
            content = step_match.group(2).strip()
            steps.append(CoTStep(number=number, content=content))
        
        # Extract answer for this path
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_match = re.search(answer_pattern, path_content, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None
        
        if answer:
            answers.append(answer)
        
        # Sort steps by number
        steps.sort(key=lambda x: x.number)
        
        paths.append(SCRPath(path_id=path_id, steps=steps, answer=answer))
    
    # Determine final answer through voting
    vote_counts = Counter(answers)
    final_answer = vote_counts.most_common(1)[0][0] if vote_counts else None
    
    return SCRResponse(
        question=question,
        paths=paths,
        final_answer=final_answer,
        vote_counts=dict(vote_counts)
    )

def create_mermaid_diagram(scr_response: SCRResponse, config: VisualizationConfig) -> str:
    """
    Convert self-consistency paths to Mermaid diagram.
    
    Args:
        scr_response: SCRResponse object containing multiple reasoning paths
        config: VisualizationConfig for text formatting
    
    Returns:
        Mermaid diagram markup as a string
    """
    diagram = ['graph TD']
    
    # Add question node
    question_content = wrap_text(scr_response.question, config)
    diagram.append(f'    Q["{question_content}"]')
    
    # Process each path
    for path in scr_response.paths:
        path_id = f'P{path.path_id}'
        
        # Add path label
        diagram.append(f'    {path_id}["Path {path.path_id}"]')
        diagram.append(f'    Q --> {path_id}')
        
        # Add steps for this path
        prev_node = path_id
        for step in path.steps:
            content = wrap_text(step.content, config)
            node_id = f'P{path.path_id}S{step.number}'
            diagram.append(f'    {node_id}["{content}"]')
            diagram.append(f'    {prev_node} --> {node_id}')
            prev_node = node_id
        
        # Add path answer
        if path.answer:
            answer_content = wrap_text(path.answer, config)
            answer_id = f'A{path.path_id}'
            diagram.append(f'    {answer_id}["{answer_content}"]')
            diagram.append(f'    {prev_node} --> {answer_id}')
    
    # Add final answer with vote counts
    if scr_response.final_answer and scr_response.vote_counts:
        vote_info = [f"{ans}: {count} votes" for ans, count in scr_response.vote_counts.items()]
        final_content = wrap_text(
            f"Final Answer (by voting):\\n{scr_response.final_answer}\\n\\n" + 
            "Vote Distribution:\\n" + "\\n".join(vote_info),
            config
        )
        diagram.append(f'    F["{final_content}"]')
        
        # Connect all path answers to final answer
        for path in scr_response.paths:
            if path.answer:
                diagram.append(f'    A{path.path_id} --> F')
    
    # Add styles
    diagram.extend([
        '    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef path fill:#fff3e0,stroke:#f57c00,stroke-width:2px;',
        '    classDef answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    classDef final fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    class Q question;',
        '    class F final;'
    ])
    
    # Apply path style to all path nodes
    for path in scr_response.paths:
        diagram.append(f'    class P{path.path_id} path;')
    
    # Apply answer style to all answer nodes
    for path in scr_response.paths:
        if path.answer:
            diagram.append(f'    class A{path.path_id} answer;')
    
    return '\n'.join(diagram)
