# Dota 2 Team Analyzer API

A FastAPI-based service that analyzes Dota 2 team compositions and predicts match outcomes based on hero matchups, team synergy, and role distribution.

## Features

- Analyzes team compositions for both Radiant and Dire teams
- Calculates win probabilities based on:
  - Hero matchups
  - Team synergy
  - Counter picks
  - Role distribution
- Provides confidence scores for predictions
- Supports both local development and serverless deployment

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API locally:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

## API Usage

### Analyze Teams

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
           "radiant_team": [1, 2, 3, 4, 5],
           "dire_team": [6, 7, 8, 9, 10]
         }'
```

Response:
```json
{
    "radiant_win_probability": 0.65,
    "radiant_synergy": 0.75,
    "radiant_counters": 0.60,
    "radiant_role_score": 0.85,
    "dire_win_probability": 0.35,
    "dire_synergy": 0.65,
    "dire_counters": 0.40,
    "dire_role_score": 0.70,
    "confidence": 0.90
}
```

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

The analysis can be tuned by modifying the following parameters in `team_analyzer.py`:

- `ROLE_BALANCE_WEIGHTS`: Weights for different hero roles
- `WIN_PROBABILITY_WEIGHTS`: Weights for different factors in win probability calculation
- `SCORE_AMPLIFICATION_FACTOR`: Base amplification factor for scores
- `TEAM_SYNERGY_AMPLIFICATION_FACTOR`: Amplification factor for team synergy
- `TEAM_COUNTERS_AMPLIFICATION_FACTOR`: Amplification factor for counter picks

## Project Structure

```
dota-analyst/
├── api.py              # FastAPI application
├── team_analyzer.py    # Core analysis logic
├── data_manager.py     # Data loading and management
├── requirements.txt    # Python dependencies
├── vercel.json        # Vercel configuration
└── README.md          # This file
```

## Notes

- The API uses serverless functions on Vercel, so the first request after a cold start will be slower
- Data is loaded into the `/tmp` directory in the serverless environment
- The analysis takes into account hero matchups, team synergy, and role distribution
- Confidence scores indicate the reliability of the prediction based on available data