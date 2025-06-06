# Dota 2 Draft Analyzer

A tool for analyzing Dota 2 draft compositions and calculating win probabilities using the STRATZ API.

## Features

- Calculate team win probabilities based on hero matchups
- Analyze individual hero win rates, team synergies, and counter picks
- REST API for easy integration
- Automatic hero name resolution

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your STRATZ API key:
```
REMOVED=your_api_key_here
STRATZ_API_URL=https://api.stratz.com/graphql
```

## Usage

### Running the API Server

Start the FastAPI server:
```bash
uvicorn api:app --reload
```

The API will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### POST /analyze-draft
Analyzes a draft and returns win probabilities.

Request body:
```json
{
  "radiant_ids": [1, 2, 3, 4, 5],
  "dire_ids": [6, 7, 8, 9, 10]
}
```

Response:
```json
{
  "radiant_heroes": [
    {"id": 1, "name": "Anti-Mage"},
    {"id": 2, "name": "Axe"},
    {"id": 3, "name": "Bane"},
    {"id": 4, "name": "Bloodseeker"},
    {"id": 5, "name": "Crystal Maiden"}
  ],
  "dire_heroes": [
    {"id": 6, "name": "Drow Ranger"},
    {"id": 7, "name": "Earthshaker"},
    {"id": 8, "name": "Ember Spirit"},
    {"id": 9, "name": "Invoker"},
    {"id": 10, "name": "Juggernaut"}
  ],
  "radiant_win_probability": 52,
  "dire_win_probability": 48
}
```

## How It Works

### Win Rate Calculation

1. **Individual Hero Win Rates**
```
individual_score = (win_rate - 0.5) * weight
```

2. **Team Synergy**
```
synergy_score = average(winRateHeroId1, winRateHeroId2) - 0.5
```

3. **Counter Advantage**
```
counter_score = winRateHeroId1 - 0.5 (if HeroId1 is in Radiant)
counter_score = 0.5 - winRateHeroId1 (if HeroId1 is in Dire)
```

4. **Final Score**
```
team_draft_score = avg_individual_win_rate + total_team_synergy_score + total_counter_advantage_score
```

### Key Parameters

#### 1. Weights
Weights control the relative influence of different components:
```python
weight_individual = 1.0  # Individual hero performance
weight_synergy = 0.5    # Team synergy
weight_counter = 0.8    # Counter picks
```

#### 2. Factor
Controls the sensitivity of win probability to score differences:
```python
def win_probability(score_radiant, score_dire):
    diff = score_radiant - score_dire
    return 1 / (1 + exp(-diff * factor))
```

Example probabilities with different factors:
```
Score Difference | Factor | Win Probability
0.5             | 1      | 62%
0.5             | 2      | 73%
0.5             | 5      | 92%
```

## STRATZ API Integration

### API Requirements
- Endpoint: https://api.stratz.com/graphql
- Required header: `User-Agent: STRATZ_API`

### GraphQL Query
```graphql
query {
  heroStats {
    heroVsHeroMatchup(heroId: 86) {
      advantage {
        with {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
        vs {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
      }
      disadvantage {
        with {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
        vs {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
      }
    }
  }
}
```