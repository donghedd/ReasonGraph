import re
from dataclasses import dataclass
from typing import List, Optional
from cot_reasoning import VisualizationConfig, wrap_text

@dataclass
class SelfRefineStep:
    """Data class representing a single step in self-refine reasoning"""
    number: int
    content: str
    is_revised: bool = False
    revision_of: Optional[int] = None

@dataclass
class SelfRefineResponse:
    """Data class representing a complete self-refine response"""
    question: str
    steps: List[SelfRefineStep]
    answer: Optional[str] = None
    revision_check: Optional[str] = None
    revised_answer: Optional[str] = None

def parse_selfrefine_response(response_text: str, question: str) -> SelfRefineResponse:
    """
    Parse self-refine response text to extract steps, answers, and revisions.
    
    Args:
        response_text: The raw response from the API
        question: The original question
    
    Returns:
        SelfRefineResponse object containing all components
    """
    # Extract initial steps
    step_pattern = r'<step number="(\d+)">\s*(.*?)\s*</step>'
    steps = []
    for match in re.finditer(step_pattern, response_text, re.DOTALL):
        number = int(match.group(1))
        content = match.group(2).strip()
        steps.append(SelfRefineStep(number=number, content=content))
    
    # Extract initial answer
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    
    # Extract revision check
    check_pattern = r'<revision_check>\s*(.*?)\s*</revision_check>'
    check_match = re.search(check_pattern, response_text, re.DOTALL)
    revision_check = check_match.group(1).strip() if check_match else None
    
    # Extract revised steps
    revised_step_pattern = r'<revised_step number="(\d+)" revises="(\d+)">\s*(.*?)\s*</revised_step>'
    for match in re.finditer(revised_step_pattern, response_text, re.DOTALL):
        number = int(match.group(1))
        revises = int(match.group(2))
        content = match.group(3).strip()
        steps.append(SelfRefineStep(
            number=number,
            content=content,
            is_revised=True,
            revision_of=revises
        ))
    
    # Extract revised answer
    revised_answer_pattern = r'<revised_answer>\s*(.*?)\s*</revised_answer>'
    revised_answer_match = re.search(revised_answer_pattern, response_text, re.DOTALL)
    revised_answer = revised_answer_match.group(1).strip() if revised_answer_match else None
    
    return SelfRefineResponse(
        question=question,
        steps=steps,
        answer=answer,
        revision_check=revision_check,
        revised_answer=revised_answer
    )

def create_mermaid_diagram(sr_response: SelfRefineResponse, config: VisualizationConfig) -> str:
    """
    Create a Mermaid diagram for self-refine reasoning.
    
    Args:
        sr_response: SelfRefineResponse object containing the reasoning steps
        config: VisualizationConfig for text formatting
    
    Returns:
        Mermaid diagram markup as a string
    """
    diagram = ['<div class="mermaid">', 'graph TD']
    
    # Add question node
    question_content = wrap_text(sr_response.question, config)
    diagram.append(f'    Q["{question_content}"]')
    
    # Track original and revised steps
    original_steps = [s for s in sr_response.steps if not s.is_revised]
    revised_steps = [s for s in sr_response.steps if s.is_revised]
    
    # Add original steps and connect them
    prev_node = 'Q'
    for step in original_steps:
        node_id = f'S{step.number}'
        content = wrap_text(step.content, config)
        diagram.append(f'    {node_id}["{content}"]')
        diagram.append(f'    {prev_node} --> {node_id}')
        prev_node = node_id
    
    # Add initial answer if present
    if sr_response.answer:
        answer_content = wrap_text(sr_response.answer, config)
        diagram.append(f'    A["{answer_content}"]')
        diagram.append(f'    {prev_node} --> A')
        prev_node = 'A'
    
    # Add revision check if present
    if sr_response.revision_check:
        check_content = wrap_text(sr_response.revision_check, config)
        diagram.append(f'    RC["{check_content}"]')
        diagram.append(f'    {prev_node} --> RC')
    
    # Add revised steps if any
    if revised_steps:
        # Process each revision step
        for i, step in enumerate(revised_steps):
            rev_node_id = f'R{step.number}'
            content = wrap_text(step.content, config)
            diagram.append(f'    {rev_node_id}["{content}"]')
            
            # Connect from the revision check to problematic step, then to revision
            if step.revision_of:
                orig_node = f'S{step.revision_of}'
                # Add connection from revision check to problematic step
                diagram.append(f'    RC --> {orig_node}')
                # Add connection from problematic step to its revision
                diagram.append(f'    {orig_node} --> {rev_node_id}')
                
                # Connect subsequent revised steps
                if i < len(revised_steps) - 1:
                    next_node = f'R{revised_steps[i + 1].number}'
                    diagram.append(f'    {rev_node_id} --> {next_node}')
    
    # Add revised answer if present
    if sr_response.revised_answer:
        revised_content = wrap_text(sr_response.revised_answer, config)
        diagram.append(f'    RA["{revised_content}"]')
        last_node = f'R{revised_steps[-1].number}' if revised_steps else 'RC'
        diagram.append(f'    {last_node} --> RA')
    
    # Add styles
    diagram.extend([
        '    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    classDef revision fill:#fff3cd,stroke:#ffc107,stroke-width:2px;',
        '    class Q question;',
        '    class A,RA answer;',
        '    class RC revision;'
    ])
    
    # Style revision nodes
    for step in revised_steps:
        diagram.append(f'    class R{step.number} revision;')
    
    diagram.append('</div>')
    return '\n'.join(diagram)