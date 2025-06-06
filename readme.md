# Dota 2 Draft Analyzer

A tool for analyzing Dota 2 draft compositions and calculating win probabilities using the STRATZ API.

## Features

- Win probability calculations for team compositions
- Individual hero analysis including:
  - Win rates
  - Synergy scores
  - Counter scores
  - Detailed matchup data
- Team composition analysis including:
  - Overall team win rates
  - Team synergy scores
  - Team counter scores
  - Strategic recommendations
- REST API integration with FastAPI
- Detailed error handling and validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dota-analyst.git
cd dota-analyst
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your STRATZ API key:
```
REMOVED=your_api_key_here
STRATZ_API_URL=https://api.stratz.com/api/v1
WEIGHT_INDIVIDUAL=1.0
WEIGHT_SYNERGY=1.0
WEIGHT_COUNTER=1.0
FACTOR=2.0
```

## Usage

### Running the API Server

Start the FastAPI server using Uvicorn:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST /analyze-draft
Analyzes a draft and returns detailed win probabilities and analysis.

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
    "radiant": {
        "heroes": [
            {
                "id": 1,
                "name": "Antimage",
                "win_rate": 0.5,
                "synergy_score": 0.5,
                "counter_score": 0.5,
                "synergies": [
                    {
                        "hero_id": 2,
                        "win_rate": 0.5
                    }
                ],
                "counters": [
                    {
                        "hero_id": 3,
                        "win_rate": 0.5
                    }
                ]
            }
        ],
        "team_win_rate": 0.5,
        "team_synergy_score": 0.5,
        "team_counter_score": 0.5,
        "synergy_description": "Antimage is synergistic with Axe",
        "counter_description": "Antimage is countered by Bane",
        "timing_and_strategy": {
            "early_game": "Focus on securing farm and objectives",
            "mid_game": "Look for team fights and map control",
            "late_game": "Push for high ground and end the game"
        },
        "conclusion": "Team composition has strong synergies and good counters"
    },
    "dire": {
        // Same structure as radiant team analysis
    }
}
```

### Analysis Methodology

The analyzer uses several factors to calculate win probabilities:

1. **Individual Hero Win Rates**
   - Base win rate for each hero
   - Performance in current meta
   - Recent match history

2. **Team Synergy**
   - Hero combination effectiveness
   - Complementary abilities
   - Role distribution
   - Team fight potential

3. **Counter Advantages**
   - Hero counter relationships
   - Lane matchups
   - Itemization counters
   - Ability counters

4. **Final Score Calculation**
   - Weighted combination of all factors
   - Normalized to probability range
   - Adjusted for team composition

### Key Parameters

The following parameters can be adjusted in the `.env` file:

- `WEIGHT_INDIVIDUAL`: Weight for individual hero performance (default: 1.0)
- `WEIGHT_SYNERGY`: Weight for team synergy (default: 1.0)
- `WEIGHT_COUNTER`: Weight for counter advantages (default: 1.0)
- `FACTOR`: Scaling factor for probability calculation (default: 2.0)

### STRATZ API Integration

The analyzer uses the STRATZ API to fetch hero matchup data. The API endpoint used is:
```
https://api.stratz.com/api/v1
```

Required GraphQL query structure:
```graphql
{
  heroStats {
    heroVsHeroMatchup(heroId: $heroId) {
      advantage {
        with {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
        }
        vs {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
        }
      }
      disadvantage {
        with {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
        }
        vs {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
        }
      }
    }
  }
}
```

## Error Handling

The API provides detailed error responses for various scenarios:

- Invalid hero IDs
- Duplicate heroes in draft
- API key issues
- Server errors

Example error response:
```json
{
    "error": "Invalid hero IDs: [999, 1000]",
    "status_code": 400
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.