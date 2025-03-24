from flask import Flask, render_template, request, jsonify
from api_base import create_api  # New import for API factory
from plain_text_reasoning import (
    create_mermaid_diagram as create_plain_diagram,
    parse_plain_text_response
)
from cot_reasoning import (
    VisualizationConfig,
    create_mermaid_diagram as create_cot_diagram,
    parse_cot_response
)
from tot_reasoning import (
    create_mermaid_diagram as create_tot_diagram,
    parse_tot_response
)
from l2m_reasoning import (
    create_mermaid_diagram as create_l2m_diagram,
    parse_l2m_response
)
from selfconsistency_reasoning import (
    create_mermaid_diagram as create_scr_diagram,
    parse_scr_response
)
from selfrefine_reasoning import (
    create_mermaid_diagram as create_srf_diagram,
    parse_selfrefine_response
)
from bs_reasoning import (
    create_mermaid_diagram as create_bs_diagram,
    parse_bs_response
)
from configs import config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/index.html')
def index_direct():
    """Directly render the main page when accessed via index.html"""
    return render_template('index.html')

@app.route('/index_cn.html')
def index_cn():
    """Render the Chinese version of the main page"""
    return render_template('index_cn.html')

@app.route('/config')
def get_config():
    """Get initial configuration"""
    return jsonify(config.get_initial_values())

@app.route('/method-config/<method_id>')
def get_method_config(method_id):
    """Get configuration for specific method"""
    method_config = config.get_method_config(method_id)
    if method_config:
        return jsonify(method_config)
    return jsonify({"error": "Method not found"}), 404

@app.route('/provider-api-key/<provider>')
def get_provider_api_key(provider):
    """Get default API key for specific provider"""
    try:
        api_key = config.general.get_default_api_key(provider)
        return jsonify({
            'success': True,
            'api_key': api_key
        })
    except Exception as e:
        logger.error(f"Error getting API key for provider {provider}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/save-api-key', methods=['POST'])
def save_api_key():
    """Save API key for a provider in memory only (no file storage)"""
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        provider = data.get('provider')
        api_key = data.get('api_key')

        if not provider or not api_key:
            return jsonify({
                'success': False,
                'error': 'Provider and API key are required'
            }), 400

        # Update API key in config (this updates the in-memory API keys only)
        config.general.provider_api_keys[provider] = api_key
        logger.info(f"Saved API key for provider: {provider} (in memory only)")

        return jsonify({
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error saving API key: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/select-method', methods=['POST'])
def select_method():
    """Let the model select the most appropriate reasoning method"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Extract parameters
        api_key = data.get('api_key')
        provider = data.get('provider', 'anthropic')
        model = data.get('model')
        question = data.get('question')

        if not all([api_key, model, question]):
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400

        # Create the selection prompt
        methods = config.methods
        prompt = f"""Given this question: "{question}"

Please select the most appropriate reasoning method from the following options to solve it:

{chr(10).join(f'- {method_id}: {config.name}' for method_id, config in methods.items())}

Consider the characteristics of each method and the nature of the question.
Output your selection in exactly this format:
<selected_method>method_id</selected_method>
where method_id is strictly one of: {', '.join(methods.keys())}.
Do not use the method or words that are not in {', '.join(methods.keys())}."""

        # Get model's selection
        try:
            api = create_api(provider, api_key, model)
            response = api.generate_response(prompt, max_tokens=100)
            
            # Extract method ID using basic string parsing
            import re
            match = re.search(r'<selected_method>(\w+)</selected_method>', response)
            if match and match.group(1) in methods:
                selected_method = match.group(1)
                return jsonify({
                    'success': True,
                    'selected_method': selected_method,
                    'raw_response': response
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Invalid method selection in response'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'API call failed: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in method selection: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process', methods=['POST'])
def process():
    """Process the reasoning request"""
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Extract parameters
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key is required'
            }), 400

        question = data.get('question')
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        # Get optional parameters with defaults
        provider = data.get('provider', 'anthropic')  # New parameter for provider
        model = data.get('model', config.general.available_models[0])
        max_tokens = int(data.get('max_tokens', config.general.max_tokens))
        prompt_format = data.get('prompt_format')
        chars_per_line = int(data.get('chars_per_line', config.general.chars_per_line))
        max_lines = int(data.get('max_lines', config.general.max_lines))
        reasoning_method = data.get('reasoning_method', 'cot')
        
        # Initialize API with factory function
        try:
            api = create_api(provider, api_key, model)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to initialize API: {str(e)}'
            }), 400
        
        # Get model response
        logger.info(f"Generating response for question using {provider} {model}")
        try:
            raw_response = api.generate_response(
                question,
                max_tokens=max_tokens,
                prompt_format=prompt_format
            )
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'API call failed: {str(e)}'
            }), 500
        
        # Create visualization config
        viz_config = VisualizationConfig(
            max_chars_per_line=chars_per_line,
            max_lines=max_lines
        )
        
        # Generate visualization based on reasoning method
        visualization = None
        try:
            if reasoning_method == 'cot':
                result = parse_cot_response(raw_response, question)
                visualization = create_cot_diagram(result, viz_config)
            elif reasoning_method == 'tot':
                result = parse_tot_response(raw_response, question)
                visualization = create_tot_diagram(result, viz_config)
            elif reasoning_method == 'l2m':
                result = parse_l2m_response(raw_response, question)
                visualization = create_l2m_diagram(result, viz_config)
            elif reasoning_method == 'scr':
                result = parse_scr_response(raw_response, question)
                visualization = create_scr_diagram(result, viz_config)
            elif reasoning_method == 'srf':
                result = parse_selfrefine_response(raw_response, question)
                visualization = create_srf_diagram(result, viz_config)
            elif reasoning_method == 'bs':
                result = parse_bs_response(raw_response, question)
                visualization = create_bs_diagram(result, viz_config)
            elif reasoning_method == 'plain':
                parse_plain_text_response(raw_response, question)
                visualization = None
                
            logger.info("Successfully generated visualization")
        except Exception as viz_error:
            logger.error(f"Visualization generation failed: {str(viz_error)}")
            # Continue without visualization
        
        # Return successful response
        return jsonify({
            'success': True,
            'raw_output': raw_response,
            'visualization': visualization
        })
        
    except Exception as e:
        # Log the error and return error response
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    try:
        # Run the application
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=False  # Disable debug mode in production
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")