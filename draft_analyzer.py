import requests
import time
import os
from itertools import combinations, product
from math import exp
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any

load_dotenv()

class DotaDraftAnalyzer:
    """
    A class for analyzing Dota 2 draft compositions and calculating win probabilities.

    This class uses the STRATZ API to fetch hero matchup data and calculates
    win probabilities based on individual hero winrates, hero synergies, and counter picks.
    """

    def __init__(self):
        """Initialize the DotaDraftAnalyzer with API configuration and hero data."""
        self.STRATZ_API_URL = os.getenv("STRATZ_API_URL")
        self.STRATZ_API_KEY = os.getenv("STRATZ_API_KEY")

        # Convert environment variables to floats with default values
        self.WEIGHT_INDIVIDUAL = float(os.getenv("WEIGHT_INDIVIDUAL", "1.0"))
        self.WEIGHT_SYNERGY = float(os.getenv("WEIGHT_SYNERGY", "1.0"))
        self.WEIGHT_COUNTER = float(os.getenv("WEIGHT_COUNTER", "1.0"))
        self.FACTOR = float(os.getenv("FACTOR", "2.0"))

        # Initialize hero directory
        self.hero_directory = self.fetch_hero_constants()

    def fetch_hero_constants(self):
        """
        Fetch hero ID to display name mapping from STRATZ API.

        Returns:
            dict: A dictionary mapping hero IDs to their display names.
        """
        query = """
        {
          constants {
            heroes {
              id
              displayName
              shortName
            }
          }
        }
        """
        headers = {
            "Authorization": f"Bearer {self.STRATZ_API_KEY}",
            "User-Agent": "STRATZ_API",
            "Accept": "application/json",
        }

        res = requests.post(self.STRATZ_API_URL, json={"query": query}, headers=headers)
        if res.status_code != 200:
            raise Exception(f"Query failed: {res.status_code} - {res.text}")

        data = res.json()["data"]["constants"]["heroes"]
        return {hero["id"]: hero["displayName"] for hero in data}

    def get_hero_vs_hero_matchup(self, hero_id: int):
        """
        Fetch hero matchup stats from STRATZ API for a specific hero.

        Args:
            hero_id (int): The ID of the hero to fetch matchup data for.

        Returns:
            dict: Raw matchup data from the API.
        """
        query = f"""
        {{
          heroStats {{
            heroVsHeroMatchup(heroId: {hero_id}) {{
              advantage {{
                with {{
                  heroId1
                  heroId2
                  winRateHeroId1
                  winRateHeroId2
                }}
                vs {{
                  heroId1
                  heroId2
                  winRateHeroId1
                  winRateHeroId2
                }}
              }}
              disadvantage {{
                with {{
                  heroId1
                  heroId2
                  winRateHeroId1
                  winRateHeroId2
                }}
                vs {{
                  heroId1
                  heroId2
                  winRateHeroId1
                  winRateHeroId2
                }}
              }}
            }}
          }}
        }}
        """
        headers = {
            "Authorization": f"Bearer {self.STRATZ_API_KEY}",
            "User-Agent": "STRATZ_API",
            "Accept": "application/json",
        }
        res = requests.post(self.STRATZ_API_URL, json={"query": query}, headers=headers)

        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(f"Query failed: {res.status_code} - {res.text}")

    def parse_matchup_data(self, raw_data, hero_id):
        """
        Parse raw matchup data into synergy and counter dictionaries.

        Args:
            raw_data (dict): Raw matchup data from the API.
            hero_id (int): The ID of the hero being analyzed.

        Returns:
            tuple: (synergy_dict, counter_dict, individual_winrate) containing hero data.
        """
        synergy = {}
        counter = {}
        individual_winrate = 0.5  # Default value
        matchup = raw_data["data"]["heroStats"]["heroVsHeroMatchup"]

        # Calculate individual winrate from matchup data
        total_matches = 0
        total_wins = 0

        for section in ["advantage", "disadvantage"]:
            for matchup_item in matchup[section]:
                for context in ["with", "vs"]:
                    for item in matchup_item[context]:
                        h1, h2 = item["heroId1"], item["heroId2"]
                        winrate = item["winRateHeroId1"] if h1 == hero_id else item["winRateHeroId2"]

                        if context == "with":
                            key = tuple(sorted((h1, h2)))
                            synergy[key] = winrate
                        else:  # vs
                            key = (h1, h2)
                            counter[key] = winrate

                        # Update individual winrate calculation
                        if h1 == hero_id or h2 == hero_id:
                            total_matches += 1
                            total_wins += winrate

        if total_matches > 0:
            individual_winrate = total_wins / total_matches

        return synergy, counter, individual_winrate

    def calculate_team_score(self, team, enemy, winrates, synergies, counters):
        """
        Calculate a team's overall score based on individual winrates, synergies, and counters.

        Args:
            team (list): List of hero IDs in the team.
            enemy (list): List of hero IDs in the enemy team.
            winrates (dict): Dictionary of individual hero winrates.
            synergies (dict): Dictionary of hero synergy data.
            counters (dict): Dictionary of hero counter data.

        Returns:
            tuple: (team_score, synergy_score, counter_score)
        """
        score = 0.0
        synergy_score = 0.0
        counter_score = 0.0

        # Individual hero winrate contributions
        for h in team:
            score += self.WEIGHT_INDIVIDUAL * (winrates.get(h, 0.5) - 0.5)

        # Synergy (within same team)
        synergy_count = 0
        for h1, h2 in combinations(team, 2):
            key = tuple(sorted((h1, h2)))
            if key in synergies:
                synergy_value = synergies[key] - 0.5
                score += self.WEIGHT_SYNERGY * synergy_value
                synergy_score += synergy_value
                synergy_count += 1

        if synergy_count > 0:
            synergy_score /= synergy_count

        # Countering enemy heroes
        counter_count = 0
        for h1, h2 in product(team, enemy):
            key = (h1, h2)
            if key in counters:
                counter_value = counters[key] - 0.5
                score += self.WEIGHT_COUNTER * counter_value
                counter_score += counter_value
                counter_count += 1

        if counter_count > 0:
            counter_score /= counter_count

        return score, synergy_score, counter_score

    def win_probability(self, score_radiant, score_dire):
        """
        Calculate win probability based on team scores.

        Args:
            score_radiant (float): Radiant team's score.
            score_dire (float): Dire team's score.

        Returns:
            float: Probability of Radiant winning (between 0 and 1).
        """
        diff = score_radiant - score_dire
        return 1 / (1 + exp(-diff * self.FACTOR))

    def build_full_matchup_matrix(self, hero_ids):
        """
        Build winrate, synergy, and counter dictionaries for all given heroes.

        Args:
            hero_ids (list): List of hero IDs to analyze.

        Returns:
            tuple: (winrates, synergies, counters) dictionaries for all heroes.
        """
        winrates = {}
        synergies = {}
        counters = {}

        for hero_id in hero_ids:
            print(f"Fetching data for hero {hero_id}...")
            raw = self.get_hero_vs_hero_matchup(hero_id)
            sy, co, wr = self.parse_matchup_data(raw, hero_id)
            winrates[hero_id] = wr
            synergies.update(sy)
            counters.update(co)
            time.sleep(0.5)  # Avoid throttling

        return winrates, synergies, counters

    def get_hero_analysis(self, hero_id: int, team_heroes: List[int], enemy_heroes: List[int],
                         winrates: Dict[int, float], synergies: Dict[Tuple[int, int], float],
                         counters: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """
        Get detailed analysis for a single hero.

        Args:
            hero_id (int): The hero ID to analyze
            team_heroes (List[int]): List of hero IDs in the same team
            enemy_heroes (List[int]): List of hero IDs in the enemy team
            winrates (Dict[int, float]): Dictionary of hero winrates
            synergies (Dict[Tuple[int, int], float]): Dictionary of hero synergies
            counters (Dict[Tuple[int, int], float]): Dictionary of hero counters

        Returns:
            Dict[str, Any]: Detailed hero analysis
        """
        # Calculate synergy score
        synergy_score = 0.0
        hero_synergies = []
        for teammate in team_heroes:
            if teammate != hero_id:
                key = tuple(sorted((hero_id, teammate)))
                if key in synergies:
                    win_rate = synergies[key]
                    synergy_score += win_rate - 0.5
                    hero_synergies.append({
                        "hero_id": teammate,
                        "win_rate": win_rate
                    })

        if len(hero_synergies) > 0:
            synergy_score /= len(hero_synergies)

        # Calculate counter score
        counter_score = 0.0
        hero_counters = []
        for enemy in enemy_heroes:
            key = (hero_id, enemy)
            if key in counters:
                win_rate = counters[key]
                counter_score += win_rate - 0.5
                hero_counters.append({
                    "hero_id": enemy,
                    "win_rate": win_rate
                })

        if len(hero_counters) > 0:
            counter_score /= len(hero_counters)

        return {
            "id": hero_id,
            "name": self.hero_directory[hero_id],
            "win_rate": winrates.get(hero_id, 0.5),
            "synergy_score": synergy_score + 0.5,  # Normalize to 0-1 range
            "counter_score": counter_score + 0.5,  # Normalize to 0-1 range
            "synergies": hero_synergies,
            "counters": hero_counters
        }

    def get_team_analysis(self, team_heroes: List[int], enemy_heroes: List[int],
                         winrates: Dict[int, float], synergies: Dict[Tuple[int, int], float],
                         counters: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        """
        Get detailed analysis for a team.

        Args:
            team_heroes (List[int]): List of hero IDs in the team
            enemy_heroes (List[int]): List of hero IDs in the enemy team
            winrates (Dict[int, float]): Dictionary of hero winrates
            synergies (Dict[Tuple[int, int], float]): Dictionary of hero synergies
            counters (Dict[Tuple[int, int], float]): Dictionary of hero counters

        Returns:
            Dict[str, Any]: Detailed team analysis
        """
        # Calculate team scores
        team_score, team_synergy_score, team_counter_score = self.calculate_team_score(
            team_heroes, enemy_heroes, winrates, synergies, counters
        )

        # Get individual hero analysis
        heroes = [
            self.get_hero_analysis(hero_id, team_heroes, enemy_heroes, winrates, synergies, counters)
            for hero_id in team_heroes
        ]

        # Generate synergy description
        synergy_pairs = []
        for h1, h2 in combinations(team_heroes, 2):
            key = tuple(sorted((h1, h2)))
            if key in synergies and synergies[key] > 0.6:  # Strong synergy threshold
                synergy_pairs.append(
                    f"{self.hero_directory[h1]} and {self.hero_directory[h2]}"
                )

        synergy_description = " and ".join(synergy_pairs) if synergy_pairs else "No strong synergies found"

        # Generate counter description
        counter_pairs = []
        for h1 in team_heroes:
            for h2 in enemy_heroes:
                key = (h1, h2)
                if key in counters and counters[key] > 0.6:  # Strong counter threshold
                    counter_pairs.append(
                        f"{self.hero_directory[h1]} counters {self.hero_directory[h2]}"
                    )

        counter_description = " and ".join(counter_pairs) if counter_pairs else "No strong counters found"

        # Generate timing and strategy
        timing_and_strategy = {
            "early_game": "Focus on securing farm and objectives",
            "mid_game": "Look for team fights and map control",
            "late_game": "Push for high ground and end the game"
        }

        # Generate conclusion
        conclusion = f"Team composition {'has strong synergies' if synergy_pairs else 'lacks strong synergies'} " \
                    f"and {'has good counters' if counter_pairs else 'lacks strong counters'}."

        return {
            "heroes": heroes,
            "team_win_rate": team_score + 0.5,  # Normalize to 0-1 range
            "team_synergy_score": team_synergy_score + 0.5,  # Normalize to 0-1 range
            "team_counter_score": team_counter_score + 0.5,  # Normalize to 0-1 range
            "synergy_description": synergy_description,
            "counter_description": counter_description,
            "timing_and_strategy": timing_and_strategy,
            "conclusion": conclusion
        }

    def evaluate_draft(self, radiant_ids, dire_ids):
        """
        Evaluate a complete draft and calculate win probabilities.

        Args:
            radiant_ids (list): List of hero IDs in the Radiant team.
            dire_ids (list): List of hero IDs in the Dire team.

        Returns:
            tuple: (radiant_analysis, dire_analysis) containing detailed team analysis
        """
        all_hero_ids = set(radiant_ids + dire_ids)
        winrates, synergies, counters = self.build_full_matchup_matrix(all_hero_ids)

        # Get detailed analysis for both teams
        radiant_analysis = self.get_team_analysis(radiant_ids, dire_ids, winrates, synergies, counters)
        dire_analysis = self.get_team_analysis(dire_ids, radiant_ids, winrates, synergies, counters)

        return radiant_analysis, dire_analysis

    def print_draft_analysis(self, radiant_ids, dire_ids):
        """
        Print a formatted analysis of the draft.

        Args:
            radiant_ids (list): List of hero IDs in the Radiant team.
            dire_ids (list): List of hero IDs in the Dire team.
        """
        radiant_analysis, dire_analysis = self.evaluate_draft(radiant_ids, dire_ids)

        print("\nRadiant Team Analysis:")
        print(f"Win Rate: {round(radiant_analysis['team_win_rate'] * 100)}%")
        print(f"Synergy Score: {round(radiant_analysis['team_synergy_score'] * 100)}%")
        print(f"Counter Score: {round(radiant_analysis['team_counter_score'] * 100)}%")
        print(f"\nSynergies: {radiant_analysis['synergy_description']}")
        print(f"Counters: {radiant_analysis['counter_description']}")
        print(f"\nConclusion: {radiant_analysis['conclusion']}")

        print("\nDire Team Analysis:")
        print(f"Win Rate: {round(dire_analysis['team_win_rate'] * 100)}%")
        print(f"Synergy Score: {round(dire_analysis['team_synergy_score'] * 100)}%")
        print(f"Counter Score: {round(dire_analysis['team_counter_score'] * 100)}%")
        print(f"\nSynergies: {dire_analysis['synergy_description']}")
        print(f"Counters: {dire_analysis['counter_description']}")
        print(f"\nConclusion: {dire_analysis['conclusion']}")


if __name__ == "__main__":
    analyzer = DotaDraftAnalyzer()
    radiant = [1, 2, 3, 4, 5]   # e.g., Anti-Mage, Axe, Bane, Bloodseeker, Crystal Maiden
    dire = [6, 7, 8, 9, 10]     # e.g., Drow, Earthshaker, Ember, Invoker, Juggernaut

    analyzer.print_draft_analysis(radiant, dire)
