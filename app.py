"""
Translation Service API
A simple web service for text translation using Claude API
"""

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict
import os
import pandas as pd
import re
import logging
import sys
import subprocess
import pathlib

from services.translator import translate_text, evaluate_quality, Translator
from utils.costs import calculate_costs, log_translation_request

# Configurar logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cargar variables de entorno solo en desarrollo
if os.getenv('ENVIRONMENT') != 'production':
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")

# Verificar variables críticas
REQUIRED_ENV_VARS = ['ANTHROPIC_API_KEY', 'MODEL_NAME', 'MAX_TOKENS']
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")

def install_and_import_translator():
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator
    except ImportError:
        logger.info("Installing deep-translator...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "deep-translator==1.11.4"])
            from deep_translator import GoogleTranslator
            return GoogleTranslator
        except Exception as e:
            logger.error(f"Failed to install deep-translator: {e}")
            return None

# Intentar importar GoogleTranslator
GoogleTranslator = install_and_import_translator()

app = FastAPI()

# Update static and template paths to be relative to the app
BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Función auxiliar para la traducción de Google
def get_google_translation(text: str, target_lang: str) -> str:
    if not GoogleTranslator:
        return "Google Translate not available"
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        logger.error(f"Error in Google translation: {str(e)}")
        return f"Translation failed: {str(e)}"

# Models
class TranslationRequest(BaseModel):
    text: str
    target_language: str

class TranslationResponse(BaseModel):
    translated_text: str
    quality_score: float
    source_language: str = "en"
    target_language: str
    processing_time: float
    costs: Dict
    reasoning: str

# Routes
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/translate", response_model=TranslationResponse)
async def handle_translation(request: TranslationRequest):
    import time
    start_time = time.time()
    
    try:
        # Verificar estado del servicio
        health = await health_check()
        if health["status"] != "healthy":
            raise HTTPException(
                status_code=503,
                detail="Translation service not properly configured"
            )
            
        # Log request
        log_translation_request(request.text, request.target_language)
        
        try:
            # Translate
            translated_text, translation_usage = await translate_text(
                request.text,
                request.target_language,
                Translator.get_translation_prompt(request.text, request.target_language)
            )
            
            # Calculate translation costs
            translation_costs = calculate_costs(
                translation_usage["input_tokens"],
                translation_usage["output_tokens"]
            )
            
            # Evaluate quality if enabled
            quality_score = 0.0
            quality_costs = {"input_tokens": 0, "output_tokens": 0, "input_cost": 0, "output_cost": 0, "total_cost": 0}
            reasoning = "No disponible"
            
            if os.getenv('QUALITY_EVALUATION_ENABLED', 'true').lower() == 'true':
                quality_score, quality_usage = await evaluate_quality(
                    request.text,
                    translated_text,
                    request.target_language
                )
                quality_costs = calculate_costs(
                    quality_usage["input_tokens"],
                    quality_usage["output_tokens"]
                )
                reasoning = quality_usage.get("reasoning", "No disponible")
            
            # Calculate total costs
            total_costs = {
                "translation": translation_costs,
                "quality_evaluation": quality_costs,
                "total": {
                    "input_tokens": translation_costs["input_tokens"] + quality_costs["input_tokens"],
                    "output_tokens": translation_costs["output_tokens"] + quality_costs["output_tokens"],
                    "total_cost": translation_costs["total_cost"] + quality_costs["total_cost"]
                }
            }
            
            return TranslationResponse(
                translated_text=translated_text,
                quality_score=quality_score,
                target_language=request.target_language,
                processing_time=time.time() - start_time,
                costs=total_costs,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def run_evaluation(limit: int = 2):  # Por defecto solo 5 textos
    try:
        results = await evaluate_translation_system(limit)
        return {
            "success": True,
            "results": results.to_dict(orient='records'),
            "summary": {
                "texts_evaluated": len(results['original'].unique()),
                "languages_tested": list(results['language'].unique()),
                "average_quality": results['our_quality_score'].mean(),
                "placeholders_preserved": results['placeholders_preserved'].mean(),
                "translations_by_language": results.groupby('language').size().to_dict()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def evaluate_translation_system(limit: int = None, target_languages: list = None):
    """
    Evaluates the translation system for a set of texts and languages.
    
    Args:
        limit (int, optional): Maximum number of texts to evaluate
        target_languages (list, optional): List of target languages
    """
    try:
        # Load dataset and limit number of texts
        df = pd.read_csv('translated_output.csv')
        if limit:
            df = df.head(limit)
        
        # If no languages specified, use defaults
        if target_languages is None:
            target_languages = ['es', 'fr', 'de', 'hu']
        
        results = []
        
        logger.info(f"Starting evaluation of {len(df)} texts in {len(target_languages)} languages...")
        
        for i, text in enumerate(df['english'], 1):
            logger.info(f"Processing text {i}/{len(df)}")
            
            for lang in target_languages:
                try:
                    # Translate with Claude
                    translated_text, translation_usage = await translate_text(
                        text,
                        lang,
                        Translator.get_translation_prompt(text, lang)
                    )
                    
                    # Evaluate quality
                    quality_score, quality_usage = await evaluate_quality(
                        text,
                        translated_text,
                        lang
                    )
                    
                    # Translate with Google
                    google_translator = GoogleTranslator(source='en', target=lang)
                    google_translation = google_translator.translate(text)
                    
                    # Verify placeholder preservation
                    original_placeholders = re.findall(r'\[.*?\]', text)
                    translated_placeholders = re.findall(r'\[.*?\]', translated_text)
                    placeholders_preserved = original_placeholders == translated_placeholders
                    
                    # Get Hungarian reference if exists
                    hungarian_reference = None
                    if lang == 'hu':
                        hungarian_reference = df[df['english'] == text]['translated_value'].iloc[0]
                    
                    results.append({
                        'original': text,
                        'language': lang,
                        'our_translation': translated_text,
                        'our_quality_score': quality_score,
                        'google_translation': google_translation,
                        'placeholders_preserved': placeholders_preserved,
                        'hungarian_reference': hungarian_reference,
                        'reasoning': quality_usage.get('reasoning', 'Not available')
                    })
                    
                    logger.info(f"Completed: {text[:30]}... -> {lang}")
                    
                except Exception as e:
                    logger.error(f"Error processing '{text[:30]}...' for {lang}: {str(e)}")
                    continue
        
        logger.info("Evaluation completed!")
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in evaluate_translation_system: {str(e)}")
        raise

# Configure templates
templates = Jinja2Templates(directory="templates")

@app.get("/evaluate/config")
async def evaluation_config(request: Request):
    """Shows the evaluation configuration page"""
    return templates.TemplateResponse("evaluate.html", {"request": request})

@app.get("/evaluate/html")
async def run_evaluation_html(
    request: Request,
    limit: int = Query(default=2, ge=1, le=10),
    languages: str = Query(default="es,fr,de")
):
    """
    Endpoint to evaluate translations and return HTML results.
    """
    try:
        # Verificar estado del servicio
        health = await health_check()
        if health["status"] != "healthy":
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error": "Service Configuration Error",
                    "details": "Translation service is not properly configured"
                },
                status_code=503
            )
            
        # Load texts and Hungarian references from CSV
        try:
            df = pd.read_csv('translated_output.csv')
            # Take only the first 'limit' rows
            df = df.head(limit)
            sample_texts = df['english'].tolist()
            hu_dict = dict(zip(df['english'], df['translated_value']))
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise HTTPException(status_code=500, detail="Error loading reference texts")

        # Create combinations of texts and target languages
        results_data = []
        for text in sample_texts:
            for lang in languages.split(','):
                results_data.append({
                    'original': text,
                    'language': lang,
                    'our_translation': '',  # Will be filled with Claude translation
                    'google_translation': '',  # Will be filled with Google translation
                    'quality_score': 0.0,  # Will be filled with quality evaluation
                    'hungarian_reference': hu_dict.get(text, '') if lang == 'hu' else ''
                })

        # Create DataFrame for processing
        results = pd.DataFrame(results_data)

        # Process each row
        for idx in range(len(results)):
            logger.info(f"Processing text {idx + 1}/{len(results)}")
            
            try:
                # Translate using Claude
                translation, _ = await translate_text(
                    results.loc[idx, 'original'],
                    results.loc[idx, 'language'],
                    Translator.get_translation_prompt(results.loc[idx, 'original'], results.loc[idx, 'language'])
                )
                results.loc[idx, 'our_translation'] = translation
                
                # Evaluate translation quality
                quality_score, _ = await evaluate_quality(
                    results.loc[idx, 'original'],
                    translation,
                    results.loc[idx, 'language']
                )
                results.loc[idx, 'quality_score'] = quality_score
                
                # Get Google translation
                google_translation = get_google_translation(
                    results.loc[idx, 'original'],
                    results.loc[idx, 'language']
                )
                results.loc[idx, 'google_translation'] = google_translation
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                results.loc[idx, 'our_translation'] = "Translation failed"
                results.loc[idx, 'quality_score'] = 0
        
        # Calculate average quality score
        avg_quality = results['quality_score'].mean()
        
        # Return template response with results
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "num_texts": len(sample_texts),
                "languages": ", ".join(languages.split(',')),
                "avg_quality": avg_quality,
                "results": results.to_dict('records')
            }
        )

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from services.translator import AnthropicClient
    
    env_status = {var: bool(os.getenv(var)) for var in REQUIRED_ENV_VARS}
    client = AnthropicClient.get_client()
    
    return {
        "status": "healthy" if client else "degraded",
        "environment": env_status,
        "services": {
            "anthropic": "configured" if client else "not configured",
            "google_translate": "available" if GoogleTranslator else "not available"
        }
    }

# Get port from Railway
port = int(os.getenv("PORT", 8000))

# At the bottom of app.py, add this if you're running the app directly:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)