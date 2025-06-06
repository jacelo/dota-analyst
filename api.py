from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from draft_analyzer import DotaDraftAnalyzer

app = FastAPI(
    title="Dota 2 Draft Analyzer API",
    description="API for analyzing Dota 2 draft compositions and calculating win probabilities",
    version="1.0.0"
)

# Initialize the analyzer once when the API starts
analyzer = DotaDraftAnalyzer()

class DraftRequest(BaseModel):
    """Request model for draft analysis"""
    radiant_ids: List[int]
    dire_ids: List[int]

class HeroInfo(BaseModel):
    """Model for hero information"""
    id: int
    name: str

class DraftResponse(BaseModel):
    """Response model for draft analysis"""
    radiant_heroes: List[HeroInfo]
    dire_heroes: List[HeroInfo]
    radiant_win_probability: int
    dire_win_probability: int

@app.post("/analyze-draft", response_model=DraftResponse)
async def analyze_draft(draft: DraftRequest):
    """
    Analyze a Dota 2 draft and return win probabilities.

    Args:
        draft (DraftRequest): The draft request containing hero IDs for both teams

    Returns:
        DraftResponse: Analysis results including hero IDs, names and win probabilities

    Raises:
        HTTPException: If the request is invalid or analysis fails
    """
    try:
        # Validate hero IDs
        for hero_id in draft.radiant_ids + draft.dire_ids:
            if hero_id not in analyzer.hero_directory:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid hero ID: {hero_id}"
                )

        # Get hero information
        radiant_heroes = [
            HeroInfo(id=hero_id, name=analyzer.hero_directory[hero_id])
            for hero_id in draft.radiant_ids
        ]
        dire_heroes = [
            HeroInfo(id=hero_id, name=analyzer.hero_directory[hero_id])
            for hero_id in draft.dire_ids
        ]

        # Calculate probabilities
        prob_radiant, prob_dire = analyzer.evaluate_draft(draft.radiant_ids, draft.dire_ids)

        return DraftResponse(
            radiant_heroes=radiant_heroes,
            dire_heroes=dire_heroes,
            radiant_win_probability=round(prob_radiant * 100),
            dire_win_probability=round(prob_dire * 100)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Dota 2 Draft Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze-draft": "POST - Analyze a draft and get win probabilities"
        }
    }