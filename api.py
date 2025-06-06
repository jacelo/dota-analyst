from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from draft_analyzer import DotaDraftAnalyzer
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# API Documentation
API_TITLE = "Dota 2 Draft Analyzer API"
API_DESCRIPTION = "API for analyzing Dota 2 draft compositions and calculating win probabilities"
API_VERSION = "1.0.0"

# Response Examples
ROOT_RESPONSE_EXAMPLE = {
    "name": API_TITLE,
    "version": API_VERSION,
    "endpoints": {
        "/analyze-draft": "POST - Analyze a draft and get win probabilities"
    }
}

# Error Response Examples
ERROR_RESPONSE_EXAMPLES = {
    "invalid_heroes": {
        "error": "Invalid hero IDs: [999, 1000]",
        "status_code": 400
    },
    "unauthorized": {
        "error": "Invalid API key",
        "status_code": 401
    },
    "server_error": {
        "error": "Analysis failed",
        "detail": "API request failed",
        "status_code": 500
    }
}

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analyzer once when the API starts
analyzer = DotaDraftAnalyzer()

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    detail: Optional[str] = None
    status_code: int

class HeroSynergy(BaseModel):
    """Model for hero synergy information"""
    hero_id: int
    win_rate: float

class HeroCounter(BaseModel):
    """Model for hero counter information"""
    hero_id: int
    win_rate: float

class TimingAndStrategy(BaseModel):
    """Model for hero timing and strategy information"""
    early_game: str
    mid_game: str
    late_game: str

class HeroAnalysis(BaseModel):
    """Model for detailed hero analysis"""
    id: int
    name: str
    win_rate: float
    synergy_score: float
    counter_score: float
    synergies: List[HeroSynergy]
    counters: List[HeroCounter]

class TeamAnalysis(BaseModel):
    """Model for team analysis"""
    heroes: List[HeroAnalysis]
    team_win_rate: float
    team_synergy_score: float
    team_counter_score: float
    synergy_description: str
    counter_description: str
    timing_and_strategy: TimingAndStrategy
    conclusion: str

class DraftRequest(BaseModel):
    """Request model for draft analysis"""
    radiant_ids: List[int] = Field(..., min_items=5, max_items=5, description="List of 5 Radiant hero IDs")
    dire_ids: List[int] = Field(..., min_items=5, max_items=5, description="List of 5 Dire hero IDs")

class DraftResponse(BaseModel):
    """Response model for draft analysis"""
    radiant: TeamAnalysis
    dire: TeamAnalysis

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            status_code=500
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).dict()
    )

@app.post("/analyze-draft",
         response_model=DraftResponse,
         responses={
             200: {
                 "description": "Successful draft analysis",
                 "content": {
                     "application/json": {
                         "example": {
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
                             }
                         }
                     }
                 }
             },
             400: {
                 "model": ErrorResponse,
                 "description": "Invalid request data",
                 "content": {
                     "application/json": {
                         "example": ERROR_RESPONSE_EXAMPLES["invalid_heroes"]
                     }
                 }
             },
             401: {
                 "model": ErrorResponse,
                 "description": "Unauthorized - Invalid API key",
                 "content": {
                     "application/json": {
                         "example": ERROR_RESPONSE_EXAMPLES["unauthorized"]
                     }
                 }
             },
             500: {
                 "model": ErrorResponse,
                 "description": "Internal server error",
                 "content": {
                     "application/json": {
                         "example": ERROR_RESPONSE_EXAMPLES["server_error"]
                     }
                 }
             }
         })
async def analyze_draft(draft: DraftRequest):
    """
    Analyze a Dota 2 draft and return win probabilities.

    Args:
        draft (DraftRequest): The draft request containing hero IDs for both teams

    Returns:
        DraftResponse: Analysis results including detailed hero and team analysis

    Raises:
        HTTPException: If the request is invalid or analysis fails
    """
    try:
        # Validate hero IDs
        invalid_heroes = []
        for hero_id in draft.radiant_ids + draft.dire_ids:
            if hero_id not in analyzer.hero_directory:
                invalid_heroes.append(hero_id)

        if invalid_heroes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid hero IDs: {invalid_heroes}"
            )

        # Check for duplicate heroes
        all_heroes = draft.radiant_ids + draft.dire_ids
        if len(set(all_heroes)) != len(all_heroes):
            raise HTTPException(
                status_code=400,
                detail="Duplicate heroes are not allowed in the draft"
            )

        # Get detailed analysis for both teams
        radiant_analysis, dire_analysis = analyzer.evaluate_draft(draft.radiant_ids, draft.dire_ids)

        return DraftResponse(
            radiant=radiant_analysis,
            dire=dire_analysis
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/",
         responses={
             200: {
                 "description": "API information",
                 "content": {
                     "application/json": {
                         "example": ROOT_RESPONSE_EXAMPLE
                     }
                 }
             }
         })
async def root():
    """Root endpoint with API information"""
    return ROOT_RESPONSE_EXAMPLE