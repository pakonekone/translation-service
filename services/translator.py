"""
Translation Service Module using Anthropic's Claude API

This module provides translation and quality evaluation services using the Claude AI model.
It handles text translation between multiple languages and provides quality assessment
of the translations.

Key Features:
    - Text translation using Claude AI
    - Translation quality evaluation
    - Token usage tracking
    - Error handling and fallbacks

Environment Variables Required:
    - ANTHROPIC_API_KEY: API key for Anthropic's services
    - MODEL_NAME: Claude model to use (default: claude-3-haiku-20240307)
    - MAX_TOKENS: Maximum tokens for response (default: 1024)
    - TEMPERATURE: Model temperature setting (default: 0.3)

Usage:
    from services.translator import translate_text, evaluate_quality
    
    text, usage = await translate_text("Hello world", "es", "Translate professionally")
    score, usage = await evaluate_quality(source_text, translated_text, "es")
"""

import anthropic
import logging
from typing import Tuple, Dict, Optional
import os
import re
import json
from fastapi import HTTPException

# Configure logging
logger = logging.getLogger(__name__)

class AnthropicClient:
    _instance: Optional[anthropic.Anthropic] = None
    
    @classmethod
    def get_client(cls) -> Optional[anthropic.Anthropic]:
        """Get or initialize Anthropic client"""
        if cls._instance is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found in environment variables")
                return None
                
            try:
                cls._instance = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                return None
                
        return cls._instance

async def translate_text(text: str, target_language: str, system_prompt: str) -> Tuple[str, Dict]:
    """
    Translates text using Claude API with specified parameters.

    Args:
        text (str): The source text to translate
        target_language (str): Target language code (e.g., 'es', 'fr')
        system_prompt (str): System prompt to guide the translation

    Returns:
        Tuple[str, Dict]: A tuple containing:
            - str: The translated text
            - Dict: Usage statistics containing:
                - input_tokens: Number of tokens in the input
                - output_tokens: Number of tokens in the output

    Raises:
        ValueError: If the API key is invalid or missing
        Exception: For other API-related errors

    Example:
        >>> text, usage = await translate_text(
        ...     "Hello world",
        ...     "es",
        ...     "You are a professional translator"
        ... )
        >>> print(text)
        "Hola mundo"
        >>> print(usage)
        {"input_tokens": 10, "output_tokens": 5}
    """
    client = AnthropicClient.get_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="Translation service unavailable - check configuration"
        )

    try:
        # Get translation prompt
        translation_prompt = Translator.get_translation_prompt(text, target_language)

        response = client.messages.create(
            model=os.getenv('MODEL_NAME', 'claude-3-haiku-20240307'),
            messages=[{"role": "user", "content": translation_prompt}],
            system=system_prompt,
            max_tokens=int(os.getenv('MAX_TOKENS', '1024')),
            temperature=float(os.getenv('TEMPERATURE', '0.3'))
        )
        
        translated_text = response.content[0].text
        
        # Validate placeholders
        original_placeholders = re.findall(r'\[.*?\]', text)
        translated_placeholders = re.findall(r'\[.*?\]', translated_text)
        
        if original_placeholders != translated_placeholders:
            # Restore original placeholders if modified
            for orig, trans in zip(original_placeholders, translated_placeholders):
                translated_text = translated_text.replace(trans, orig)
        
        return translated_text, {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"Translation failed: {str(e)}", {
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e)
        }

async def evaluate_quality(source: str, translation: str, target_language: str) -> Tuple[float, Dict]:
    """Evaluates translation quality with specific criteria"""
    client = AnthropicClient.get_client()
    if not client:
        return 50, {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning": "Service not configured"
        }

    try:
        # Get prompts
        system_prompt = Translator.get_system_prompt()
        few_shot = Translator.get_few_shot_examples()
        eval_prompt = Translator.get_evaluation_prompt(source, translation, f"English to {target_language}")

        prompt = f"{system_prompt}\n\n{few_shot}\n\n{eval_prompt}"

        response = client.messages.create(
            model=os.getenv('MODEL_NAME', 'claude-3-haiku-20240307'),
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            max_tokens=int(os.getenv('MAX_TOKENS', '128')),
            temperature=float(os.getenv('TEMPERATURE', '0.1'))
        )
        
        # Process response
        response_text = response.content[0].text.strip()
        
        try:
            # Extract JSON and reasoning
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            clean_json_text = response_text[json_start:json_end]
            
            result = json.loads(clean_json_text)
            
            # Get reasoning if available
            reasoning_text = response_text[json_end:].strip()
            reasoning = reasoning_text[reasoning_text.find("Reasoning:") + 10:].strip() if "Reasoning:" in reasoning_text else "Not available"
            
            # Validate score
            score = result["score"]
            if not result.get("analysis", {}).get("placeholder_check"):
                score = score * 0.5  # Penalize placeholder issues
            
            score = max(0, min(100, score))
            
            return score, {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning": reasoning
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return 50, {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning": "Error processing response"
            }
            
    except Exception as e:
        logger.error(f"Quality evaluation failed: {e}")
        return 50, {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning": "Evaluation failed"
        }

class Translator:
    @staticmethod
    def get_system_prompt() -> str:
        return """You are an expert translation evaluator with deep knowledge of linguistics and cross-cultural communication.
Your task is to provide reference-less evaluation of translations, mimicking human direct assessment.

Key responsibilities:
1. Evaluate translation quality from 0-100 (0=completely incorrect, 100=perfect)
2. Preserve and check special elements like placeholders [placeholder]
3. Consider both semantic accuracy and linguistic fluency
4. Be consistent in scoring across different texts
5. Provide clear reasoning for scores"""

    @staticmethod
    def get_evaluation_prompt(source_text: str, translation: str, lang_pair: str) -> str:
        return f"""<instructions>
Evaluate this translation from {lang_pair}.

Follow these evaluation criteria in order:

1. Semantic Accuracy (50%):
   - Is the core meaning preserved completely?
   - Are all key concepts transferred correctly?
   - Are there any omissions or additions?
   - Check placeholder preservation: [] tokens must remain exactly as in source

2. Linguistic Quality (30%):
   - Grammar correctness
   - Natural expression in target language
   - Appropriate register/style
   - Proper word choice

3. Cross-Cultural Adaptation (20%):
   - Cultural context preservation
   - Cultural nuance adaptation
   - Regional language variants consideration

Source text: {source_text}
Translation: {translation}

Provide output in this JSON format:
{{
    "score": float,  // 0-100 score
    "analysis": {{
        "semantic": float,  // 0-50 points
        "linguistic": float,  // 0-30 points 
        "cultural": float,  // 0-20 points
        "placeholder_check": bool,  // true if all [] preserved
        "critical_errors": [str],  // list of major issues found
        "strengths": [str]  // list of positive aspects
    }}
}}
</instructions>"""
    @staticmethod
    def get_few_shot_examples() -> str:
        return """<examples>
<example>
Source: "See how [brokerName] compares to [number] other brokers across [dataPoints]+ criteria"
Translation: "Vea cómo [brokerName] se compara con [number] otros brokers en [dataPoints]+ criterios"
Score: 95
Reasoning: Perfect placeholder preservation, accurate meaning transfer, natural Spanish expression
</example>

<example>
Source: "Calculate stock trade commission at [brokerName] with our fee calculator."
Translation: "Calcula la comisión de operaciones bursátiles en [brokerName] con nuestra calculadora de tarifas."
Score: 100
Reasoning: Perfect semantic accuracy, natural phrasing, correct domain terminology
</example>

<example>
Source: "Access to archived threads is only available to registered users."
Translation: "El acceso a los hilos archivados solo está disponible para usuarios registrados."
Score: 90
Reasoning: Accurate meaning, good grammar, slightly formal register but acceptable
</example>
</examples>"""

    @staticmethod
    def get_translation_prompt(text: str, target_language: str) -> str:
        return f"""<instructions>
Translate the following text from English to {target_language}.

Critical rules:
1. Preserve ALL placeholders in square brackets [] exactly as they appear
2. Maintain numerical formats and units
3. Use official financial terminology for the target region
4. Keep technical terms consistent
5. Preserve any HTML/formatting tags

Examples of correct translations:
<examples>
<example>
English: "See how [brokerName] compares to [number] other brokers across [dataPoints]+ criteria"
Spanish: "Vea cómo [brokerName] se compara con [number] otros brokers en [dataPoints]+ criterios"
</example>

<example>
English: "Access to archived threads is only available to registered users. Without this, we can only keep them for you for 1 month."
French: "L'accès aux discussions archivées n'est disponible qu'aux utilisateurs inscrits. Sans cela, nous ne pouvons les conserver que pendant 1 mois."
</example>

<example>
English: "Calculate stock trade commission at [brokerName] with our fee calculator."
German: "Berechnen Sie die Aktienhandelskommission bei [brokerName] mit unserem Gebührenrechner."
</example>
</examples>

Text to translate:
{text}

Provide only the translation, no explanations.
</instructions>"""
