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

        # Initialize RAG chain
        self.rag_chain = self._create_rag_chain()

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

    def _create_rag_chain(self):
        """
        Create a RAG chain for generating insights with relevant context.

        Returns:
            RunnableSequence: Configured RAG chain
        """
        template = """
        You are a Dota 2 expert analyst. Your task is to analyze 5v5 Dota 2 drafts and predict win probabilities and provide in-depth insights on the matchup.
        Use the following context and team analysis to provide a detailed analysis.

        Context from Vector Store:
        {context}

        Team Analysis:
        {team_analysis}

        IMPORTANT: Only analyze the heroes that are actually present in the teams. Do not make up or hallucinate heroes that aren't in the team data.
        The Radiant and Dire teams are explicitly listed in the Team Analysis section above.

        Based on the above information, provide a detailed analysis of the matchup. Consider:
        1. Hero synergies and counter-picks between the ACTUAL heroes in each team
        2. Team composition strengths and weaknesses based on the REAL heroes present
        3. Early, mid, and late game strategies for the SPECIFIC heroes in each team
        4. Key objectives and team fight potential based on the ACTUAL team compositions
        5. Win probability factors derived from the provided team analysis data

        Format your response in JSON format with the following structure:
        JSON_RESPONSE={{
            "overall_assessment": "Provide a concise but comprehensive summary of the matchup, highlighting key factors that will influence the game outcome. Only reference heroes that are actually in the teams.",
            "radiant_analysis": {{
                "strengths": ["List specific strengths based on the ACTUAL heroes in the Radiant team"],
                "weaknesses": ["List specific weaknesses that the enemy team can exploit, based on the REAL Radiant heroes"],
                "key_heroes": ["Identify the most impactful heroes from the ACTUAL Radiant team and explain why they are crucial"],
                "strategy": "Provide specific strategic recommendations based on the REAL Radiant team composition"
            }},
            "dire_analysis": {{
                "strengths": ["List specific strengths based on the ACTUAL heroes in the Dire team"],
                "weaknesses": ["List specific weaknesses that the enemy team can exploit, based on the REAL Dire heroes"],
                "key_heroes": ["Identify the most impactful heroes from the ACTUAL Dire team and explain why they are crucial"],
                "strategy": "Provide specific strategic recommendations based on the REAL Dire team composition"
            }},
            "matchup_insights": {{
                "early_game": "Detail specific early game objectives, lane matchups, and key timings for the ACTUAL heroes",
                "mid_game": "Explain mid-game power spikes, key objectives, and team fight strategies for the REAL team compositions",
                "late_game": "Analyze late game scaling, win conditions, and team fight execution for the SPECIFIC heroes present",
                "key_counters": ["List specific hero counters between the ACTUAL heroes in both teams"],
                "team_fight_analysis": "Analyze team fight potential, positioning requirements, and key abilities to watch for the REAL team compositions"
            }},
            "prediction": {{
                "expected_winner": "Radiant or Dire",
                "confidence": "High/Medium/Low",
                "reasoning": "Provide detailed reasoning for the prediction, considering all factors analyzed above and the ACTUAL team compositions"
            }}
        }}

        Remember to:
        - ONLY analyze heroes that are actually present in the teams
        - NEVER make up or hallucinate heroes that aren't in the team data
        - Be specific and detailed in your analysis
        - Use the context from the vector store to support your insights
        - Consider hero synergies and counter-picks between the REAL heroes
        - Provide actionable strategic recommendations
        - Base your prediction on concrete factors from the analysis and context data
        """

        prompt = PromptTemplate(
            input_variables=["context", "team_analysis"],
            template=template
        )

        return prompt | self.llm

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

        Args:
            radiant_team (List[int]): List of Radiant hero IDs
            dire_team (List[int]): List of Dire hero IDs

        Returns:
            str: Formatted context from vector store
        """
        context_parts = []

        # Get hero information
        for hero_id in radiant_team + dire_team:
            hero_name = self._get_hero_name(hero_id)
            hero_docs = self.hero_retriever.invoke(hero_name)
            if hero_docs:
                context_parts.append(f"Hero Information for {hero_name}:")
                context_parts.extend([doc.page_content for doc in hero_docs])

        # Get matchup information
        for r_hero in radiant_team:
            for d_hero in dire_team:
                r_name = self._get_hero_name(r_hero)
                d_name = self._get_hero_name(d_hero)
                matchup_query = f"matchup between {r_name} and {d_name}"
                matchup_docs = self.matchup_retriever.invoke(matchup_query)
                if matchup_docs:
                    context_parts.append(f"Matchup Information for {r_name} vs {d_name}:")
                    context_parts.extend([doc.page_content for doc in matchup_docs])

        return "\n\n".join(context_parts)

    def generate_insights(self, radiant_team: List[int], dire_team: List[int]) -> Dict[str, Any]:
        """
        Generate insights about a team matchup using RAG-enhanced LLM.

        Args:
            radiant_team (List[int]): List of Radiant hero IDs
            dire_team (List[int]): List of Dire hero IDs

        Returns:
            Dict[str, Any]: Generated insights in JSON format
        """
        # Get team analysis
        analysis = self.team_analyzer.analyze_teams(radiant_team, dire_team)

        # Format team data
        radiant_data = self._format_team_data(radiant_team, analysis, True)
        dire_data = self._format_team_data(dire_team, analysis, False)

        # Get relevant context from vector store
        context = self._get_relevant_context(radiant_team, dire_team)

        # Format team analysis for the prompt
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

        Overall Confidence: {analysis['confidence']:.1%}
        """

        # Generate insights using RAG chain
        response = self.rag_chain.invoke({
            "context": context,
            "team_analysis": team_analysis
        })

        try:
            response_text = response.text()

            # Find JSON content between ```json and ``` markers
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON markers found, try to find any JSON-like content
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    raise json.JSONDecodeError("No JSON content found", response_text, 0)

            # Clean up the JSON string
            json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove single-line comments
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # Remove multi-line comments
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r'}\s*{', '},{', json_str)  # Fix missing commas between objects
            json_str = re.sub(r'}\s*]', '}]', json_str)  # Fix missing commas before array end
            json_str = re.sub(r']\s*}', ']}', json_str)  # Fix missing commas before object end

            insights = json.loads(json_str)
            return insights
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return {
                "error": f"Failed to parse insights: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else str(response)
            }

if __name__ == "__main__":
    # Example usage
    import time

    generator = InsightGenerator()
    radiant_team = [10, 20, 30, 40, 50]  # Example hero IDs
    dire_team = [60, 70, 80, 90, 100]    # Example hero IDs

    start_time = time.time()
    insights = generator.generate_insights(radiant_team, dire_team)
    end_time = time.time()

    print("Team Analysis")
    print("================================================")
    print(json.dumps(insights, indent=2))
    print(f"\nInsight generation took {end_time - start_time:.2f} seconds")
