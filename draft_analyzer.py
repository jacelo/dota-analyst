import requests
import time
import os
from itertools import combinations, product
from math import exp
from dotenv import load_dotenv

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
        self.REMOVED = os.getenv("REMOVED")

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
            "Authorization": f"Bearer {self.REMOVED}",
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
            "Authorization": f"Bearer {self.REMOVED}",
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
            tuple: (synergy_dict, counter_dict) containing hero synergy and counter data.
        """
        synergy = {}
        counter = {}
        matchup = raw_data["data"]["heroStats"]["heroVsHeroMatchup"]

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
        return synergy, counter

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
            float: The calculated team score.
        """
        score = 0.0

        # Individual hero winrate contributions
        for h in team:
            score += self.WEIGHT_INDIVIDUAL * (winrates.get(h, 0.5) - 0.5)

        # Synergy (within same team)
        for h1, h2 in combinations(team, 2):
            key = tuple(sorted((h1, h2)))
            if key in synergies:
                score += self.WEIGHT_SYNERGY * (synergies[key] - 0.5)

        # Countering enemy heroes
        for h1, h2 in product(team, enemy):
            key = (h1, h2)
            if key in counters:
                score += self.WEIGHT_COUNTER * (counters[key] - 0.5)

        return score

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
            sy, co = self.parse_matchup_data(raw, hero_id)

            # Cache individual winrate as avg of with/against (fallback)
            winrates[hero_id] = 0.5 + (
                sum((w - 0.5) for w in sy.values()) +
                sum((w - 0.5) for w in co.values())
            ) / (len(sy) + len(co) + 1e-5)

            synergies.update(sy)
            counters.update(co)
            time.sleep(0.5)  # Avoid throttling

        return winrates, synergies, counters

    def evaluate_draft(self, radiant_ids, dire_ids):
        """
        Evaluate a complete draft and calculate win probabilities.

        Args:
            radiant_ids (list): List of hero IDs in the Radiant team.
            dire_ids (list): List of hero IDs in the Dire team.

        Returns:
            tuple: (radiant_win_probability, dire_win_probability)
        """
        all_hero_ids = set(radiant_ids + dire_ids)
        winrates, synergies, counters = self.build_full_matchup_matrix(all_hero_ids)

        radiant_score = self.calculate_team_score(radiant_ids, dire_ids, winrates, synergies, counters)
        dire_score = self.calculate_team_score(dire_ids, radiant_ids, winrates, synergies, counters)

        prob_radiant = self.win_probability(radiant_score, dire_score)
        prob_dire = 1 - prob_radiant

        return prob_radiant, prob_dire

    def print_draft_analysis(self, radiant_ids, dire_ids):
        """
        Print a formatted analysis of the draft.

        Args:
            radiant_ids (list): List of hero IDs in the Radiant team.
            dire_ids (list): List of hero IDs in the Dire team.
        """
        prob_radiant, prob_dire = self.evaluate_draft(radiant_ids, dire_ids)

        print(f"\nRadiant: {', '.join(self.hero_directory[hero_id] for hero_id in radiant_ids)}")
        print(f"Dire: {', '.join(self.hero_directory[hero_id] for hero_id in dire_ids)}")
        print(f"\nRadiant win probability: {round(prob_radiant * 100)}%")
        print(f"Dire win probability:    {round(prob_dire * 100)}%")


if __name__ == "__main__":
    analyzer = DotaDraftAnalyzer()
    radiant = [1, 2, 3, 4, 5]   # e.g., Anti-Mage, Axe, Bane, Bloodseeker, Crystal Maiden
    dire = [6, 7, 8, 9, 10]     # e.g., Drow, Earthshaker, Ember, Invoker, Juggernaut

    analyzer.print_draft_analysis(radiant, dire)
