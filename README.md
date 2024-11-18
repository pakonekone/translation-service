# Translation Service API

A professional translation service powered by Claude AI, specifically designed for financial and investment industry content. The service maintains high accuracy for technical terminology while preserving special formatting and placeholders.

## Features

- 🌐 Multi-language support (Spanish, French, German, Japanese, Arabic, Hindi, Portuguese, Hungarian)
- 📊 Quality evaluation system
- 🔄 Placeholder preservation
- 📈 Cost tracking and optimization
- 🔍 Comparison with Google Translate
- 💹 Specialized in financial/investment content

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd translation-service
```


2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```


3. Install dependencies:

```bash
pip install -r requirements.txt
```


4. Configure environment variables:
Create a `.env` file with the following variables:

```env
ANTHROPIC_API_KEY=your_api_key
QUALITY_EVALUATION_ENABLED=true
MODEL_NAME=claude-3-haiku-20240307
MAX_TOKENS=1024
TEMPERATURE=0.3
```


## Usage

1. Start the server:
```bash
uvicorn app:app --reload
```


2. Access the web interface:
- Translation interface: http://localhost:8000
- Evaluation interface: http://localhost:8000/evaluate/config

## Language Prioritization Strategy

Our language prioritization is based on:

1. Market size and financial activity
2. Technical complexity
3. Available reference data
4. Cost optimization

Priority tiers:
- **Tier 1**: Spanish, French, German (largest markets, similar structures)
- **Tier 2**: Japanese, Arabic (major markets, different scripts)
- **Tier 3**: Hindi, Portuguese (growing markets)

## Quality Evaluation Methodology

Our evaluation system uses a three-component scoring approach:

1. **Semantic Accuracy (50%)**
   - Core meaning preservation
   - Concept transfer accuracy
   - Placeholder preservation

2. **Linguistic Quality (30%)**
   - Grammar correctness
   - Natural expression
   - Technical terminology accuracy

3. **Cross-Cultural Adaptation (20%)**
   - Cultural context preservation
   - Regional language variants
   - Financial terminology localization

## Future Improvements

1. **Technical Enhancements**
   - Implement terminology management system
   - Develop specialized financial term database
   - Add context-aware translation memory

2. **Quality Improvements**
   - Expand evaluation datasets
   - Add industry expert review process
   - Implement automated regression testing

3. **Cost Optimization**
   - Batch processing for bulk translations
   - Smart token usage optimization
   - Use a local model for cost reduction

## Project Structure

```
translation-service/
├── app.py                # Main FastAPI application
├── services/
│   └── translator.py     # Translation service implementation
├── utils/
│   └── costs.py         # Cost calculation utilities
├── templates/
│   ├── evaluate.html    # Evaluation interface
│   └── results.html     # Results display
├── static/
│   └── index.html       # Main translation interface
└── requirements.txt     # Project dependencies
```

## API Endpoints

- `GET /`: Main translation interface
- `POST /translate`: Translation endpoint
- `GET /evaluate/config`: Evaluation configuration
- `GET /evaluate/html`: Run evaluation with HTML results
- `GET /evaluate`: Run evaluation with JSON results
