from typing import List, Dict, Any, Tuple
from data_manager import DataManager
import numpy as np
import argparse
from itertools import permutations

# Role balance weights for different hero roles. Total should be 1.0
ROLE_BALANCE_WEIGHTS = {
    "core": 1.0,
    "offlane": 0.9,
    "support": 0.8,
    "roamer": 0.7,
    "carry": 1.0
}

# Expected role distribution for a standard team
EXPECTED_ROLES = ["carry", "core", "offlane", "support", "roamer"]

# Weights for win probability calculation. Total should be 1.0
WIN_PROBABILITY_WEIGHTS = {
    "hero_scores": 0.2,
    "synergy": 0.4,
    "counters": 0.1,
    "roles": 0.3
}

# Amplification factors for different calculations
SCORE_AMPLIFICATION_FACTOR = 2.0
TEAM_SYNERGY_AMPLIFICATION_FACTOR = 3.0
TEAM_COUNTERS_AMPLIFICATION_FACTOR = 2.0


class TeamAnalyzer:
    """
    A class for analyzing Dota 2 team compositions and predicting match outcomes.

    This analyzer uses hero matchup data, team synergy, counter picks, and role distribution
    to calculate win probabilities and provide detailed team analysis.
    """

    def __init__(self):
        """
        Initialize the TeamAnalyzer with required data from DataManager.

        Loads hero directory, matchups, and builds a lookup table for quick access
        to matchup data.
        """
        self.data_manager = DataManager()
        self.hero_directory = self.data_manager.get_hero_directory()
        self.matchups = self.data_manager.get_matchups()
        self.matchup_lookup = self._build_matchup_lookup()
        self.hero_baselines = self._compute_hero_baselines()
        self.missing_matchup_count = 0
        self.total_matchup_count = 0

    def _build_matchup_lookup(self) -> Dict[Tuple[int, int], float]:
        """
        Build a lookup table for hero matchups from the raw matchup data.

        Returns:
            Dict[Tuple[int, int], float]: A dictionary mapping hero ID pairs to their win rates.
            The key is a tuple of (hero1_id, hero2_id) and the value is hero1's win rate against hero2.
        """
        lookup = {}
        for hero_id_str, hero_data in self.matchups.items():
            hero_id = int(hero_id_str)

            for section in ['advantage', 'disadvantage']:
                for group in hero_data.get(section, []):
                    for kind in ['with', 'vs']:
                        for matchup in group.get(kind, []):
                            h1 = matchup.get('heroId1')
                            h2 = matchup.get('heroId2')
                            win_rate = matchup.get('winRateHeroId1', 0.5)
                            if h1 is not None and h2 is not None:
                                lookup[(h1, h2)] = win_rate
        return lookup

    def _compute_hero_baselines(self) -> Dict[int, float]:
        """
        Compute baseline win rates for each hero based on their overall performance.

        Returns:
            Dict[int, float]: A dictionary mapping hero IDs to their baseline win rates.
        """
        hero_win_sums = {}
        hero_win_counts = {}
        for (h1, h2), win_rate in self.matchup_lookup.items():
            hero_win_sums[h1] = hero_win_sums.get(h1, 0) + win_rate
            hero_win_counts[h1] = hero_win_counts.get(h1, 0) + 1
        return {
            hero: hero_win_sums[hero] / hero_win_counts[hero] for hero in hero_win_sums
        }

    def calculate_hero_win_rate(self, hero_id: int, against_hero_id: int) -> float:
        """
        Calculate the win rate of a hero against another hero.

        Args:
            hero_id (int): The ID of the hero to calculate win rate for
            against_hero_id (int): The ID of the opposing hero

        Returns:
            float: The win rate of hero_id against against_hero_id, between 0 and 1
        """
        self.total_matchup_count += 1
        win_rate = self.matchup_lookup.get((hero_id, against_hero_id))
        if win_rate is not None:
            return win_rate

        win_rate = self.matchup_lookup.get((against_hero_id, hero_id))
        if win_rate is not None:
            return 1.0 - win_rate

        self.missing_matchup_count += 1
        return self.hero_baselines.get(hero_id, 0.5)

    def amplify_score(self, score: float, factor: float = SCORE_AMPLIFICATION_FACTOR) -> float:
        """
        Amplify a score to make differences more pronounced.

        Args:
            score (float): The input score between 0 and 1
            factor (float): The amplification factor to apply

        Returns:
            float: The amplified score, still between 0 and 1
        """
        deviation = score - 0.5
        amplified = 0.5 + factor * deviation
        return max(0.0, min(1.0, amplified))

    def calculate_team_synergy(self, team: List[int]) -> float:
        """
        Calculate the synergy score for a team of heroes.

        Args:
            team (List[int]): List of hero IDs in the team

        Returns:
            float: A synergy score between 0 and 1, where 1 indicates perfect synergy
        """
        synergy_scores = []
        count = 0
        for i, hero1 in enumerate(team):
            for hero2 in team[i+1:]:
                synergy = self.matchup_lookup.get((hero1, hero2))
                if synergy is None:
                    synergy = self.matchup_lookup.get((hero2, hero1))
                if synergy is not None:
                    synergy_scores.append(synergy)
                    count += 1
        if not synergy_scores:
            return 0.5

        avg = np.mean(synergy_scores)
        coverage_factor = count / 10.0  # 10 is max unique pairs in team of 5
        return self.amplify_score(avg * coverage_factor + 0.5 * (1 - coverage_factor),
                                factor=TEAM_SYNERGY_AMPLIFICATION_FACTOR)

    def calculate_team_counters(self, team: List[int], enemy_team: List[int]) -> float:
        """
        Calculate how well a team counters the enemy team.

        Args:
            team (List[int]): List of hero IDs in the team
            enemy_team (List[int]): List of hero IDs in the enemy team

        Returns:
            float: A counter score between 0 and 1, where 1 indicates perfect countering
        """
        counter_scores = []
        matchups_found = 0
        for hero in team:
            has_counter = False
            for enemy in enemy_team:
                rate = self.matchup_lookup.get((hero, enemy))
                if rate is None:
                    rate = 1.0 - self.matchup_lookup.get((enemy, hero), 0.5)
                else:
                    has_counter = True
                counter_scores.append(rate)
            if has_counter:
                matchups_found += 1

        if not counter_scores:
            return 0.5

        avg = np.mean(counter_scores)
        coverage_factor = matchups_found / len(team)
        return self.amplify_score(avg * coverage_factor + 0.5 * (1 - coverage_factor),
                                factor=TEAM_COUNTERS_AMPLIFICATION_FACTOR)

    def calculate_role_score(self, team: List[int]) -> float:
        """
        Calculate how well a team's heroes fit their expected roles.

        Args:
            team (List[int]): List of hero IDs in the team

        Returns:
            float: A role score between 0 and 1, where 1 indicates perfect role distribution
        """
        role_data = self.data_manager.get_team_roles(team)

        # Build hero-role score matrix
        hero_role_matrix = {
            hero: {
                role: role_data.get(hero, {}).get(role, 0.0) * ROLE_BALANCE_WEIGHTS.get(role, 1.0)
                for role in EXPECTED_ROLES
            }
            for hero in team
        }

        best_score = 0.0
        # Try all permutations of roles among team
        for role_order in permutations(EXPECTED_ROLES):
            used_heroes = set()
            total_score = 0.0
            for role, hero in zip(role_order, team):
                score = hero_role_matrix.get(hero, {}).get(role, 0.0)
                total_score += score
                used_heroes.add(hero)
            best_score = max(best_score, total_score)

        return best_score / len(EXPECTED_ROLES)

    def calculate_win_probability(self, radiant_team: List[int], dire_team: List[int]) -> Tuple[float, float, float]:
        """
        Calculate win probabilities for both teams and confidence in the prediction.

        Args:
            radiant_team (List[int]): List of hero IDs in the Radiant team
            dire_team (List[int]): List of hero IDs in the Dire team

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - Radiant team's win probability
                - Dire team's win probability
                - Confidence in the prediction
        """
        radiant_hero_scores = [
            sum(self.calculate_hero_win_rate(r_hero, d_hero) for d_hero in dire_team) / len(dire_team)
            for r_hero in radiant_team
        ]
        dire_hero_scores = [
            sum(self.calculate_hero_win_rate(d_hero, r_hero) for r_hero in radiant_team) / len(radiant_team)
            for d_hero in dire_team
        ]

        radiant_synergy = self.calculate_team_synergy(radiant_team)
        dire_synergy = self.calculate_team_synergy(dire_team)

        radiant_counters = self.calculate_team_counters(radiant_team, dire_team)
        dire_counters = self.calculate_team_counters(dire_team, radiant_team)

        radiant_roles = self.calculate_role_score(radiant_team)
        dire_roles = self.calculate_role_score(dire_team)

        confidence = 1.0 - (self.missing_matchup_count / max(self.total_matchup_count, 1))

        radiant_score = (
            np.mean(radiant_hero_scores) * WIN_PROBABILITY_WEIGHTS["hero_scores"] +
            radiant_synergy * WIN_PROBABILITY_WEIGHTS["synergy"] * confidence +
            radiant_counters * WIN_PROBABILITY_WEIGHTS["counters"] * confidence +
            radiant_roles * WIN_PROBABILITY_WEIGHTS["roles"]
        )
        dire_score = (
            np.mean(dire_hero_scores) * WIN_PROBABILITY_WEIGHTS["hero_scores"] +
            dire_synergy * WIN_PROBABILITY_WEIGHTS["synergy"] * confidence +
            dire_counters * WIN_PROBABILITY_WEIGHTS["counters"] * confidence +
            dire_roles * WIN_PROBABILITY_WEIGHTS["roles"]
        )

        total_score = radiant_score + dire_score
        if total_score == 0:
            radiant_prob = 0.5
        else:
            radiant_prob = radiant_score / total_score

        radiant_prob = 0.5 + (radiant_prob - 0.5) * confidence
        dire_prob = 1.0 - radiant_prob

        return radiant_prob, dire_prob, confidence

    def format_team_with_roles(self, team: List[int]) -> List[Dict[str, Any]]:
        """
        Assigns each hero in the team to one of the EXPECTED_ROLES based on their role levels.
        Returns a list of dictionaries containing hero information and their assigned role.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing:
                - hero_id: int
                - display_name: str
                - role: str
        """
        formatted_team = []
        assigned_heroes = set()
        assigned_roles = set()

        # Gather and sort roles per hero
        hero_roles = []
        for hero_id in team:
            hero_data = next((h for h in self.hero_directory if h["id"] == hero_id), None)
            if hero_data:
                hero_roles.append({
                    'hero': hero_data,
                    'roles': {r["roleId"]: r["level"] for r in hero_data["roles"]}
                })
            else:
                formatted_team.append({
                    "hero_id": hero_id,
                    "display_name": f"Hero_{hero_id}",
                    "role": "unknown"
                })

        # Build candidate list of (hero, role, level)
        role_candidates = []
        for hr in hero_roles:
            hero = hr["hero"]
            for role in EXPECTED_ROLES:
                level = hr["roles"].get(role, 0)
                if level > 0:
                    role_candidates.append((hero, role, level))

        # Sort all possible assignments by role level descending
        role_candidates.sort(key=lambda x: x[2], reverse=True)

        hero_role_map = {}
        for hero, role, level in role_candidates:
            if hero['id'] not in assigned_heroes and role not in assigned_roles:
                hero_role_map[hero['id']] = role
                assigned_heroes.add(hero['id'])
                assigned_roles.add(role)

        # Assign remaining heroes to their highest available role
        for hr in hero_roles:
            hero = hr["hero"]
            if hero['id'] in hero_role_map:
                assigned_role = hero_role_map[hero['id']]
            else:
                # Pick highest level unassigned role
                sorted_roles = sorted(hr["roles"].items(), key=lambda x: x[1], reverse=True)
                assigned_role = next((r for r, _ in sorted_roles if r not in assigned_roles), None)
                if assigned_role:
                    assigned_roles.add(assigned_role)
                else:
                    assigned_role = sorted_roles[0][0] if sorted_roles else "unknown"

            formatted_team.append({
                "hero_id": hero['id'],
                "display_name": hero['displayName'],
                "role": assigned_role
            })

        return formatted_team

    def analyze_teams(self, radiant_team: List[int], dire_team: List[int]) -> Dict[str, Any]:
        """
        Perform a complete analysis of both teams.

        Args:
            radiant_team (List[int]): List of hero IDs in the Radiant team
            dire_team (List[int]): List of hero IDs in the Dire team

        Returns:
            Dict[str, Any]: A dictionary containing:
                - Win probabilities for both teams
                - Synergy scores
                - Counter scores
                - Role scores
                - Overall confidence in the analysis
                - Team information with hero names and best roles
        """
        self.missing_matchup_count = 0
        self.total_matchup_count = 0

        radiant_prob, dire_prob, confidence = self.calculate_win_probability(radiant_team, dire_team)

        return {
            "radiant_win_probability": radiant_prob,
            "radiant_synergy": self.calculate_team_synergy(radiant_team),
            "radiant_counters": self.calculate_team_counters(radiant_team, dire_team),
            "radiant_role_score": self.calculate_role_score(radiant_team),
            "dire_win_probability": dire_prob,
            "dire_synergy": self.calculate_team_synergy(dire_team),
            "dire_counters": self.calculate_team_counters(dire_team, radiant_team),
            "dire_role_score": self.calculate_role_score(dire_team),
            "confidence": confidence,
            "radiant_team": self.format_team_with_roles(radiant_team),
            "dire_team": self.format_team_with_roles(dire_team)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Dota 2 team compositions')
    parser.add_argument('--radiant', nargs=5, type=int, required=True, help='List of 5 hero IDs for Radiant team')
    parser.add_argument('--dire', nargs=5, type=int, required=True, help='List of 5 hero IDs for Dire team')
    args = parser.parse_args()

    analyzer = TeamAnalyzer()
    results = analyzer.analyze_teams(args.radiant, args.dire)

    print("\nTeam Analysis Results:")
    print("=" * 50)
    print(f"Radiant Team: {results['radiant_team']}")
    print(f"Win Probability: {results['radiant_win_probability']:.2%}")
    print(f"Synergy: {results['radiant_synergy']:.2%}")
    print(f"Counters: {results['radiant_counters']:.2%}")
    print(f"Role Score: {results['radiant_role_score']:.2%}")

    print(f"\nDire Team: {results['dire_team']}")
    print(f"Win Probability: {results['dire_win_probability']:.2%}")
    print(f"Synergy: {results['dire_synergy']:.2%}")
    print(f"Counters: {results['dire_counters']:.2%}")
    print(f"Role Score: {results['dire_role_score']:.2%}")

    print(f"\nConfidence Score: {results['confidence']:.2%}")
