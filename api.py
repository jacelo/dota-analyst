from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from team_analyzer import TeamAnalyzer
from data_manager import DataManager
from insight_generator import InsightGenerator
from vector_store import VectorStore

# Initialize data manager and analyzer at module level
data_manager = DataManager()
data_manager.ensure_data_loaded()
analyzer = TeamAnalyzer()
insight_generator = InsightGenerator()
vector_store = VectorStore()
vector_store.load_data()  # Load data into vector store

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
    insights: Dict[str, Any]
    radiant_team: List[str]
    dire_team: List[str]

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for heroes or matchups")
    search_type: str = Field(..., description="Type of search: 'heroes' or 'matchups'")
    n_results: int = Field(5, description="Number of results to return")

@app.post("/analyze", response_model=TeamAnalysisResponse)
async def analyze_teams(request: TeamAnalysisRequest):
    try:
        # Get basic analysis
        results = analyzer.analyze_teams(request.radiant_team, request.dire_team)

        # Generate insights
        insights = insight_generator.generate_insights(request.radiant_team, request.dire_team)

        # Combine results
        results["insights"] = insights
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    try:
        if request.search_type == "heroes":
            results = vector_store.search_heroes(request.query, request.n_results)
        elif request.search_type == "matchups":
            results = vector_store.search_matchups(request.query, request.n_results)
        else:
            raise HTTPException(status_code=400, detail="Invalid search type. Must be 'heroes' or 'matchups'")

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# This is the handler that Vercel will use
app = app