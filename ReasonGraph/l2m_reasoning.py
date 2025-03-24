import re
from dataclasses import dataclass
from typing import List, Optional
import textwrap

@dataclass
class L2MStep:
    """Data class representing a single L2M step"""
    number: int
    question: str  # The sub-question for this step
    reasoning: str  # The reasoning process
    answer: str    # The answer to this sub-question

@dataclass
class L2MResponse:
    """Data class representing a complete L2M response"""
    main_question: str
    steps: List[L2MStep]
    final_answer: Optional[str] = None

def parse_l2m_response(response_text: str, question: str) -> L2MResponse:
    """
    Parse L2M response text to extract steps and final answer.
    
    Args:
        response_text: The raw response from the API
        question: The original question
    
    Returns:
        L2MResponse object containing main question, steps, and final answer
    """
    # Extract all steps
    step_pattern = r'<step number="(\d+)">\s*<question>(.*?)</question>\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*</step>'
    steps = []
    
    for match in re.finditer(step_pattern, response_text, re.DOTALL):
        number = int(match.group(1))
        sub_question = match.group(2).strip()
        reasoning = match.group(3).strip()
        answer = match.group(4).strip()
        steps.append(L2MStep(
            number=number,
            question=sub_question,
            reasoning=reasoning,
            answer=answer
        ))
    
    # Extract final answer
    final_answer_pattern = r'<final_answer>\s*(.*?)\s*</final_answer>'
    final_answer_match = re.search(final_answer_pattern, response_text, re.DOTALL)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else None
    
    # Sort steps by number
    steps.sort(key=lambda x: x.number)
    
    return L2MResponse(main_question=question, steps=steps, final_answer=final_answer)

def wrap_text(text: str, max_chars: int = 40, max_lines: int = 4) -> str:
    """Wrap text to fit within box constraints with proper line breaks."""
    text = text.replace('\n', ' ').replace('"', "'")
    wrapped_lines = textwrap.wrap(text, width=max_chars)
    
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        wrapped_lines[-1] = wrapped_lines[-1][:max_chars-3] + "..."
    
    return "<br>".join(wrapped_lines)

def create_mermaid_diagram(l2m_response: L2MResponse, config: 'VisualizationConfig') -> str:
    """
    Convert L2M steps to Mermaid diagram.
    
    Args:
        l2m_response: L2MResponse object containing the reasoning steps
        config: VisualizationConfig for text formatting
    
    Returns:
        Mermaid diagram markup as a string
    """
    diagram = ['<div class="mermaid">', 'graph TD']
    
    # Add main question node
    question_content = wrap_text(l2m_response.main_question, config.max_chars_per_line, config.max_lines)
    diagram.append(f'    Q["{question_content}"]')
    
    # Add decomposition node
    diagram.append(f'    D["Problem Decomposition"]')
    diagram.append(f'    Q --> D')
    
    # Add all step nodes with sub-questions, reasoning, and answers
    if l2m_response.steps:
        # Connect decomposition to first step
        diagram.append(f'    D --> S{l2m_response.steps[0].number}')
        
        for i, step in enumerate(l2m_response.steps):
            # Create sub-question node
            sq_content = wrap_text(f"Q{step.number}: {step.question}", config.max_chars_per_line, config.max_lines)
            sq_id = f'S{step.number}'
            diagram.append(f'    {sq_id}["{sq_content}"]')
            
            # Create reasoning node
            r_content = wrap_text(step.reasoning, config.max_chars_per_line, config.max_lines)
            r_id = f'R{step.number}'
            diagram.append(f'    {r_id}["{r_content}"]')
            
            # Create answer node
            a_content = wrap_text(f"A{step.number}: {step.answer}", config.max_chars_per_line, config.max_lines)
            a_id = f'A{step.number}'
            diagram.append(f'    {a_id}["{a_content}"]')
            
            # Connect the nodes
            diagram.append(f'    {sq_id} --> {r_id}')
            diagram.append(f'    {r_id} --> {a_id}')
            
            # Connect to next step if exists
            if i < len(l2m_response.steps) - 1:
                next_id = f'S{l2m_response.steps[i + 1].number}'
                diagram.append(f'    {a_id} --> {next_id}')
    
    # Add final answer node if exists
    if l2m_response.final_answer:
        final_content = wrap_text(f"Final: {l2m_response.final_answer}", config.max_chars_per_line, config.max_lines)
        diagram.append(f'    F["{final_content}"]')
        if l2m_response.steps:
            diagram.append(f'    A{l2m_response.steps[-1].number} --> F')
        else:
            diagram.append('    D --> F')
    
    # Add styles
    diagram.extend([
        '    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef reasoning fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    classDef decomp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;',
        '    class Q,S1,S2,S3,S4,S5 question;',
        '    class R1,R2,R3,R4,R5 reasoning;',
        '    class A1,A2,A3,A4,A5,F answer;',
        '    class D decomp;',
        '    linkStyle default stroke:#666,stroke-width:2px;'
    ])
    
    diagram.append('</div>')
    return '\n'.join(diagram)