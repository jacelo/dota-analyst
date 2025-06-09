import json
from typing import Dict, Any, List
from data_manager import DataManager
from team_analyzer import TeamAnalyzer
from vector_store import VectorStore
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema.runnable import RunnablePassthrough
import multiprocessing as mp
from functools import partial
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
import time
import re

def _extract_json_from_response(response_text: str, part: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response with robust error handling.

    Args:
        response_text (str): The raw response text from the LLM
        part (str): The part of the analysis being processed (for error messages)

    Returns:
        Dict[str, Any]: Parsed JSON or error information
    """
    try:
        # First try to find JSON between markers
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
        if not json_match:
            # Try to find any JSON-like content
            json_match = re.search(r'({[\s\S]*})', response_text)

        if json_match:
            json_str = json_match.group(1)
            # Clean up JSON string
            json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove single-line comments
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # Remove multi-line comments
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r'}\s*{', '},{', json_str)  # Fix missing commas between objects
            json_str = re.sub(r']\s*{', '],{', json_str)  # Fix missing commas between array and object
            json_str = re.sub(r'}\s*\[', '},{', json_str)  # Fix missing commas between object and array

            # Try to parse the cleaned JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to extract key-value pairs
                key_value_pairs = {}
                # Look for common patterns in the text
                for line in response_text.split('\n'):
                    # Match "key": value or key: value patterns
                    kv_match = re.search(r'["\']?(\w+)["\']?\s*:\s*["\']?([^"\',\n]+)["\']?', line)
                    if kv_match:
                        key, value = kv_match.groups()
                        key_value_pairs[key] = value.strip()

                if key_value_pairs:
                    return key_value_pairs

                return {
                    "error": f"Failed to parse {part} JSON: {str(e)}",
                    "raw_response": response_text
                }

        # For overall_assessment, if no JSON found, try to extract the paragraph
        if part == "overall_assessment":
            # Look for a substantial paragraph (more than 50 chars)
            paragraphs = [p.strip() for p in response_text.split('\n\n') if len(p.strip()) > 50]
            if paragraphs:
                return {"assessment": paragraphs[0]}

        return {
            "error": f"No valid JSON or content found in {part} response",
            "raw_response": response_text
        }
    except Exception as e:
        return {
            "error": f"Error processing {part} response: {str(e)}",
            "raw_response": response_text
        }

def generate_insight_part(part: str, context: str, team_analysis: str, **kwargs) -> Dict[str, Any]:
    """Generate a specific part of the insights in a separate process."""
    process_start_time = time.time()
    try:
        # Initialize LLM in the new process
        llm = ChatOllama(model="qwen3:0.6b")

        # Create prompt template for this part
        templates = {
            "overall_assessment": """
            You are a Dota 2 expert analyst. Provide a comprehensive summary of the 5v5 matchup.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Provide a summary of the matchup, highlighting key factors that will influence the game outcome.
            Only reference heroes that are actually in the teams.
            Format your response as a single paragraph.
            """,

            "team_analysis": """
            You are a Dota 2 expert analyst. Analyze the {team_side} team ({team_heroes}) and provide a comprehensive analysis on the following:
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Analyze the {team_side} team ({team_heroes}) and provide a comprehensive analysis on the following:
            1. List of specific strengths
            2. List of specific weaknesses
            3. List of key heroes and why they are crucial
            4. Specific strategic recommendations

            Format your response in JSON:
            {{
                "strengths": ["list of strengths"],
                "weaknesses": ["list of weaknesses"],
                "key_heroes": [
                    {{
                        "name": "hero name",
                        "role": "hero role",
                        "strength": "why this hero is crucial"
                    }}
                ],
                "strategy": "strategic recommendations"
            }}
            """,

            "matchup_insights": """
            You are a Dota 2 expert analyst. Analyze the matchup dynamics between the teams.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Provide detailed and comprehensive analysis of:
            1. Early game objectives, lane matchups, and key timings
            2. Mid-game power spikes, objectives, and team fight strategies
            3. Late game scaling, win conditions, and team fight execution
            4. Key hero counters between the teams
            5. Team fight analysis

            Format your response in JSON:
            {{
                "early_game": "early game analysis",
                "mid_game": "mid game analysis",
                "late_game": "late game analysis",
                "key_counters": ["list of key counters"],
                "team_fight_analysis": "team fight analysis"
            }}
            """,

            "prediction": """
            You are a Dota 2 expert analyst. Predict the outcome of the matchup.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Make a prediction about the match outcome, considering all factors analyzed. Explain in detail your reasoning focusing on the team compositions, synergy, counters, win rates and previous matches.
            Format your response in JSON:
            {{
                "expected_winner": "Radiant or Dire",
                "reasoning": "detailed reasoning for the prediction"
            }}
            """
        }

        prompt = PromptTemplate(
            input_variables=["context", "team_analysis"] + (["team_side", "team_heroes"] if part == "team_analysis" else []),
            template=templates[part]
        )

        # Generate response
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "team_analysis": team_analysis,
            **kwargs
        })
        response_text = response.text()

        # Extract and parse JSON using the new robust parser
        result = _extract_json_from_response(response_text, part)

        process_end_time = time.time()
        result["process_time"] = process_end_time - process_start_time
        return result
    except Exception as e:
        process_end_time = time.time()
        return {
            "error": f"Error generating {part} insights: {str(e)}",
            "process_time": process_end_time - process_start_time
        }

class InsightGenerator:
    """
    A class that generates insights about Dota 2 team matchups using RAG-enhanced LLM.
    Analyzes team compositions, synergies, and counter picks to provide strategic insights.
    """

    def __init__(self):
        """Initialize the InsightGenerator with required data managers."""
        self.data_manager = DataManager()
        self.team_analyzer = TeamAnalyzer()
        self.vector_store = VectorStore()
        self.hero_directory = self.data_manager.get_hero_directory()

        # Initialize Ollama LLM
        self.llm = ChatOllama(model="qwen3:0.6b")

        # Initialize RAG chains for different aspects
        self.rag_chains = self._create_rag_chains()

        # Initialize contextual compression retriever
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.hero_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vector_store.hero_store.as_retriever(search_kwargs={"k": 3})
        )
        self.matchup_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vector_store.matchup_store.as_retriever(search_kwargs={"k": 3})
        )

        # Set up multiprocessing pool
        self.num_processes = min(mp.cpu_count(), 5)  # Use up to 5 processes
        self.pool = mp.Pool(processes=self.num_processes)

    def __del__(self):
        """Clean up the multiprocessing pool."""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

    def _create_rag_chains(self) -> Dict[str, Any]:
        """Create separate RAG chains for different aspects of the analysis."""
        templates = {
            "overall_assessment": """
            You are a Dota 2 expert analyst. Provide a comprehensive summary of the 5v5 matchup.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Provide a summary of the matchup, highlighting key factors that will influence the game outcome.
            Only reference heroes that are actually in the teams.
            Format your response as a single paragraph.
            """,

            "team_analysis": """
            You are a Dota 2 expert analyst. Analyze the strengths, weaknesses, key heroes, and strategy for {team_side} team.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Analyze the {team_side} team ({team_heroes}) and provide a comprehensive analysis on the following:
            1. List of specific strengths
            2. List of specific weaknesses
            3. List of key heroes and why they are crucial
            4. Specific strategic recommendations

            Format your response in JSON:
            {{
                "strengths": ["list of strengths"],
                "weaknesses": ["list of weaknesses"],
                "key_heroes": ["list of key heroes with explanations"],
                "strategy": "strategic recommendations"
            }}
            """,

            "matchup_insights": """
            You are a Dota 2 expert analyst. Analyze the matchup dynamics between the teams.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Provide detailed and comprehensive analysis of:
            1. Early game objectives, lane matchups, and key timings
            2. Mid-game power spikes, objectives, and team fight strategies
            3. Late game scaling, win conditions, and team fight execution
            4. Key hero counters between the teams
            5. Team fight analysis

            Format your response in JSON:
            {{
                "early_game": "early game analysis",
                "mid_game": "mid game analysis",
                "late_game": "late game analysis",
                "key_counters": ["list of key counters"],
                "team_fight_analysis": "team fight analysis"
            }}
            """,

            "prediction": """
            You are a Dota 2 expert analyst. Predict the outcome of the matchup.
            Consider the team analysis and context provided.

            Context from Vector Store:
            {context}

            Team Analysis:
            {team_analysis}

            Make a prediction about the match outcome, considering all factors analyzed. Explain in detail your reasoning focusing on the team compositions, synergy, counters, win rates and previous matches.
            Format your response in JSON:
            {{
                "expected_winner": "Radiant or Dire",
                "reasoning": "detailed reasoning for the prediction"
            }}
            """
        }

        return {
            key: PromptTemplate(
                input_variables=["context", "team_analysis"] + (["team_side", "team_heroes"] if key == "team_analysis" else []),
                template=template
            ) | self.llm
            for key, template in templates.items()
        }

    def _get_hero_name(self, hero_id: int) -> str:
        """
        Get the display name of a hero from their ID.

        Args:
            hero_id (int): The ID of the hero

        Returns:
            str: The display name of the hero
        """
        hero = next((h for h in self.hero_directory if h["id"] == hero_id), None)
        return hero["displayName"] if hero else f"Hero_{hero_id}"

    def _format_team_data(self, team: List[int], analysis: Dict[str, Any], is_radiant: bool) -> Dict[str, Any]:
        """
        Format team data for the prompt.

        Args:
            team (List[int]): List of hero IDs
            analysis (Dict[str, Any]): Analysis results
            is_radiant (bool): Whether this is the Radiant team

        Returns:
            Dict[str, Any]: Formatted team data
        """
        prefix = "radiant" if is_radiant else "dire"
        return {
            "team": analysis[f"{prefix}_team"],
            "win_probability": analysis[f"{prefix}_win_probability"],
            "synergy": analysis[f"{prefix}_synergy"],
            "counters": analysis[f"{prefix}_counters"],
            "role_score": analysis[f"{prefix}_role_score"]
        }

    def _get_relevant_context(self, radiant_team: List[int], dire_team: List[int]) -> str:
        """
        Get relevant context from the vector store for the teams.
        Optimized version that batches queries and caches results.

        Args:
            radiant_team (List[int]): List of Radiant hero IDs
            dire_team (List[int]): List of Dire hero IDs

        Returns:
            str: Formatted context from vector store
        """
        context_parts = []
        hero_cache = {}  # Cache for hero information
        matchup_cache = {}  # Cache for matchup information

        # Batch hero queries
        all_heroes = set(radiant_team + dire_team)
        hero_names = {hero_id: self._get_hero_name(hero_id) for hero_id in all_heroes}

        # Get hero information in parallel using the process pool
        hero_tasks = []
        for hero_id, hero_name in hero_names.items():
            if hero_id not in hero_cache:
                hero_tasks.append((hero_id, hero_name))

        if hero_tasks:
            # Create a batch query for all heroes
            batch_query = " OR ".join(f'"{name}"' for _, name in hero_tasks)
            hero_docs = self.hero_retriever.invoke(batch_query)

            # Process and cache results
            for doc in hero_docs:
                hero_id = next((h_id for h_id, name in hero_names.items()
                              if name in doc.page_content), None)
                if hero_id:
                    hero_cache[hero_id] = doc.page_content

        # Get matchup information in parallel
        matchup_tasks = []
        for r_hero in radiant_team:
            for d_hero in dire_team:
                if (r_hero, d_hero) not in matchup_cache:
                    r_name = hero_names[r_hero]
                    d_name = hero_names[d_hero]
                    matchup_tasks.append((r_hero, d_hero, r_name, d_name))

        if matchup_tasks:
            # Create a batch query for all matchups
            batch_query = " OR ".join(f'matchup between "{r_name}" and "{d_name}"'
                                    for _, _, r_name, d_name in matchup_tasks)
            matchup_docs = self.matchup_retriever.invoke(batch_query)

            # Process and cache results
            for doc in matchup_docs:
                for r_hero, d_hero, r_name, d_name in matchup_tasks:
                    if r_name in doc.page_content and d_name in doc.page_content:
                        matchup_cache[(r_hero, d_hero)] = doc.page_content

        # Format hero information
        for hero_id in all_heroes:
            if hero_id in hero_cache:
                context_parts.append(f"Hero Information for {hero_names[hero_id]}:")
                context_parts.append(hero_cache[hero_id])

        # Format matchup information
        for r_hero in radiant_team:
            for d_hero in dire_team:
                if (r_hero, d_hero) in matchup_cache:
                    r_name = hero_names[r_hero]
                    d_name = hero_names[d_hero]
                    context_parts.append(f"Matchup Information for {r_name} vs {d_name}:")
                    context_parts.append(matchup_cache[(r_hero, d_hero)])

        return "\n\n".join(context_parts)

    def generate_insights(self, radiant_team: List[int], dire_team: List[int]) -> Dict[str, Any]:
        """Generate insights about a team matchup using parallel processing."""
        timing_info = {
            "analysis_start": time.time(),
            "team_analysis_time": None,
            "context_time": None,
            "process_start": None,
            "process_end": None,
            "total_time": None
        }

        # Get team analysis
        analysis_start = time.time()
        analysis = self.team_analyzer.analyze_teams(radiant_team, dire_team)
        timing_info["team_analysis_time"] = time.time() - analysis_start

        # Format team data
        radiant_data = self._format_team_data(radiant_team, analysis, True)
        dire_data = self._format_team_data(dire_team, analysis, False)

        # Get relevant context
        context_start = time.time()
        context = self._get_relevant_context(radiant_team, dire_team)
        timing_info["context_time"] = time.time() - context_start

        # Format team analysis
        team_analysis = f"""
        Radiant Team: {', '.join(radiant_data['team'])}
        - Win Probability: {radiant_data['win_probability']:.1%}
        - Team Synergy: {radiant_data['synergy']:.1%}
        - Counter Advantage: {radiant_data['counters']:.1%}
        - Role Distribution: {radiant_data['role_score']:.1%}

        Dire Team: {', '.join(dire_data['team'])}
        - Win Probability: {dire_data['win_probability']:.1%}
        - Team Synergy: {dire_data['synergy']:.1%}
        - Counter Advantage: {dire_data['counters']:.1%}
        - Role Distribution: {dire_data['role_score']:.1%}
        """

        # Prepare tasks for parallel processing
        tasks = [
            ("overall_assessment", context, team_analysis),
            ("team_analysis", context, team_analysis, "Radiant", ', '.join(radiant_data['team'])),
            ("team_analysis", context, team_analysis, "Dire", ', '.join(dire_data['team'])),
            ("matchup_insights", context, team_analysis),
            ("prediction", context, team_analysis)
        ]

        # Generate insights in parallel using multiprocessing
        timing_info["process_start"] = time.time()
        results = []
        for task in tasks:
            if len(task) == 3:
                part, ctx, analysis = task
                result = self.pool.apply_async(generate_insight_part, (part, ctx, analysis))
            else:
                part, ctx, analysis, side, heroes = task
                result = self.pool.apply_async(generate_insight_part, (part, ctx, analysis),
                                             kwds={"team_side": side, "team_heroes": heroes})
            results.append((task[0], result))

        # Collect results and timing information
        processed_results = []
        process_times = {}
        for part, result in results:
            processed_result = result.get()  # This is where we wait for each process
            processed_results.append(processed_result)
            process_times[part] = processed_result.pop("process_time", None)

        timing_info["process_end"] = time.time()
        timing_info["total_time"] = time.time() - timing_info["analysis_start"]

        # Calculate timing breakdowns
        timing_breakdown = {
            "total_execution": timing_info["total_time"],
            "team_analysis": timing_info["team_analysis_time"],
            "context_generation": timing_info["context_time"],
            "parallel_processing": timing_info["process_end"] - timing_info["process_start"],
            "per_process": process_times,
            "overhead": timing_info["total_time"] - (
                timing_info["team_analysis_time"] +
                timing_info["context_time"] +
                (timing_info["process_end"] - timing_info["process_start"])
            )
        }

        # Combine results
        insights = {
            "overall_assessment": processed_results[0].get("overall_assessment", processed_results[0].get("error", "Failed to generate overall assessment")),
            "radiant_analysis": processed_results[1],
            "dire_analysis": processed_results[2],
            "matchup_insights": processed_results[3],
            "prediction": processed_results[4],
            "timing": timing_breakdown
        }

        return insights

if __name__ == "__main__":
    # Example usage
    generator = InsightGenerator()
    radiant_team = [10, 20, 30, 40, 50]  # Example hero IDs
    dire_team = [60, 70, 80, 90, 100]    # Example hero IDs

    insights = generator.generate_insights(radiant_team, dire_team)

    print("Team Analysis")
    print("================================================")
    print(json.dumps(insights, indent=2))
    print("\nTiming Information:")
    print("================================================")
    timing = insights['timing']
    print(f"Total execution time: {timing['total_execution']:.2f} seconds")
    print("\nBreakdown:")
    print(f"- Team analysis: {timing['team_analysis']:.2f} seconds")
    print(f"- Context generation: {timing['context_generation']:.2f} seconds")
    print(f"- Parallel processing: {timing['parallel_processing']:.2f} seconds")
    print(f"- Overhead: {timing['overhead']:.2f} seconds")
    print("\nPer-process timing:")
    for part, time_taken in timing['per_process'].items():
        print(f"- {part}: {time_taken:.2f} seconds")
    print(f"\nUsing {generator.num_processes} processes")
