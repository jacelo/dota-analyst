from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from team_analyzer import TeamAnalyzer
from data_manager import DataManager

# Initialize data manager and analyzer at module level
data_manager = DataManager()
data_manager.ensure_data_loaded()
analyzer = TeamAnalyzer()

app = FastAPI(
    title="Dota 2 Team Analyzer API",
    description="API for analyzing Dota 2 team compositions and predicting match outcomes",
    version="1.0.0"
)

class TeamAnalysisRequest(BaseModel):
    radiant_team: List[int] = Field(..., min_items=5, max_items=5, description="List of 5 hero IDs for Radiant team")
    dire_team: List[int] = Field(..., min_items=5, max_items=5, description="List of 5 hero IDs for Dire team")

class TeamAnalysisResponse(BaseModel):
    radiant_win_probability: float
    radiant_synergy: float
    radiant_counters: float
    radiant_role_score: float
    dire_win_probability: float
    dire_synergy: float
    dire_counters: float
    dire_role_score: float
    confidence: float

@app.post("/analyze", response_model=TeamAnalysisResponse)
async def analyze_teams(request: TeamAnalysisRequest):
    try:
        results = analyzer.analyze_teams(request.radiant_team, request.dire_team)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This is the handler that Vercel will use
app = app