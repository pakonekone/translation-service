"""
Cost calculation and logging utilities
"""
import logging
from typing import Dict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def calculate_costs(input_tokens: int, output_tokens: int) -> Dict:
    """Calculate costs based on token usage"""
    input_cost = input_tokens * float(os.getenv('CLAUDE_COST_INPUT', '0.001'))
    output_cost = output_tokens * float(os.getenv('CLAUDE_COST_OUTPUT', '0.005'))
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

def log_translation_request(text: str, target_language: str):
    """Log translation request details"""
    logger.info(f"Translation request - Language: {target_language}")
    logger.info(f"Text preview: {text[:100]}...") 