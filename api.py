from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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

# Only load vector store data if it hasn't been loaded yet
if not vector_store.is_loaded():
    vector_store.load_data()

app = FastAPI(
    title="Dota 2 Team Analyzer API",
    description="API for analyzing Dota 2 team compositions and predicting match outcomes",
    version="1.0.0"
)

class HeroInfo(BaseModel):
    hero_id: int
    role: str
    display_name: str
    short_name: str

class TeamInfo(BaseModel):
    heroes: List[HeroInfo]
    win_probability: float
    synergy_score: float
    counter_score: float
    role_score: float

class AnalysisResponse(BaseModel):
    radiant: TeamInfo
    dire: TeamInfo
    synergy_strength_analysis: Dict[str, str]
    timing_strategy_analysis: Dict[str, str]
    counter_analysis: str
    conclusion: str
    confidence: float
    timing: Optional[Dict[str, Any]]

class TeamAnalysisRequest(BaseModel):
    radiant_team: List[int] = Field(..., min_items=5, max_items=5, description="List of 5 hero IDs for Radiant team")
    dire_team: List[int] = Field(..., min_items=5, max_items=5, description="List of 5 hero IDs for Dire team")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_teams(request: TeamAnalysisRequest):
    try:
        # Get numerical analysis from TeamAnalyzer
        analysis = analyzer.analyze_teams(request.radiant_team, request.dire_team)

        # Get detailed insights from InsightGenerator
        insights = insight_generator.generate_insights(request.radiant_team, request.dire_team)

        # Format hero information for both teams
        def format_hero_info(team_data: List[Dict[str, Any]]) -> List[HeroInfo]:
            hero_info = []
            for hero_data in team_data:
                hero = next((h for h in data_manager.get_hero_directory() if h["id"] == hero_data["hero_id"]), None)
                if hero:
                    hero_info.append(HeroInfo(
                        hero_id=hero_data["hero_id"],
                        role=hero_data["role"],
                        display_name=hero["displayName"],
                        short_name=hero["shortName"]
                    ))
            return hero_info

        # Construct the response
        response = {
            "radiant": TeamInfo(
                heroes=format_hero_info(analysis["radiant_team"]),
                win_probability=analysis["radiant_win_probability"],
                synergy_score=analysis["radiant_synergy"],
                counter_score=analysis["radiant_counters"],
                role_score=analysis["radiant_role_score"]
            ),
            "dire": TeamInfo(
                heroes=format_hero_info(analysis["dire_team"]),
                win_probability=analysis["dire_win_probability"],
                synergy_score=analysis["dire_synergy"],
                counter_score=analysis["dire_counters"],
                role_score=analysis["dire_role_score"]
            ),
            "synergy_strength_analysis": {
                "radiant_synergy_analysis": insights["synergy_strength_analysis"]["radiant_synergy_analysis"],
                "dire_synergy_analysis": insights["synergy_strength_analysis"]["dire_synergy_analysis"]
            },
            "timing_strategy_analysis": {
                "early_game": insights["timing_strategy_analysis"]["early_game"],
                "mid_game": insights["timing_strategy_analysis"]["mid_game"],
                "late_game": insights["timing_strategy_analysis"]["late_game"]
            },
            "counter_analysis": insights["counter_analysis"],
            "conclusion": insights["conclusion"],
            "confidence": analysis["confidence"],
            "timing": insights["timing"]
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This is the handler that Vercel will use
app = app