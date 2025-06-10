from typing import List, Dict, Any, Tuple
from data_manager import DataManager
import numpy as np
import argparse
from itertools import permutations

# Position to role mapping for team composition analysis
POSITION_ROLE_TRAITS = {
    1: [  # Carry / Safe Lane
        "CARRY",
        "ESCAPE",
        "DURABLE",
        "PUSHER",
        "NUKER",  # Occasionally for snowball carries
    ],
    2: [  # Mid Lane
        "CARRY",
        "NUKER",
        "ESCAPE",
        "DISABLER",
        "PUSHER",
    ],
    3: [  # Offlane
        "DURABLE",
        "INITIATOR",
        "DISABLER",
        "PUSHER",
        "NUKER",  # Occasionally
    ],
    4: [  # Soft Support
        "INITIATOR",
        "DISABLER",
        "NUKER",
        "ESCAPE",
        "SUPPORT",
    ],
    5: [  # Hard Support
        "SUPPORT",
        "DISABLER",
        "INITIATOR",
        "PUSHER",  # Less common, for aura/creep heroes
    ]
}

# Weights for different aspects of team analysis
ANALYSIS_WEIGHTS = {
    "hero_scores": 0.25,      # Individual hero performance
    "synergy": 0.20,          # Team synergy
    "counters_for": 0.15,     # How well team counters enemy
    "counters_against": 0.10, # How well team handles enemy counters
    "role_fit": 0.30          # How well heroes fit their positions
}

# Thresholds for synergy and counter calculations
SYNERGY_THRESHOLD = 0.55  # Only count synergies above this threshold
COUNTER_THRESHOLD_FOR = 0.60  # Hero A counters hero B if A's win rate > 60%
COUNTER_THRESHOLD_AGAINST = 0.40  # Hero A is countered by hero B if A's win rate < 40%
MIN_SAMPLE_SIZE = 100    # Minimum sample size for reliable data

class TeamAnalyzer:
    """
    A class for analyzing Dota 2 team compositions and predicting match outcomes.

    This analyzer uses hero matchup data, team synergy, counter picks, and role distribution
    to calculate win probabilities and provide detailed team analysis.
    """

    @classmethod
    def as_string(cls, results: Dict[str, Any], hero_directory: List[Dict[str, Any]]) -> str:
        """
        Format team analysis results as a string.

        Args:
            results: The analysis results dictionary from analyze_teams()
            hero_directory: The hero directory for looking up hero names

        Returns:
            str: Formatted string containing the analysis results
        """
        output = []
        output.append("\nTeam Analysis Results:")
        output.append("=" * 50)

        # Format Radiant team analysis
        output.append("\nRadiant Team Analysis:")
        output.append("-" * 30)
        output.append(f"Win Probability: {results['radiant_win_probability']:.2%}")
        output.append(f"Synergy Score: {results['radiant_synergy']:.2%}")
        output.append(f"Counter Analysis:")
        output.append(f"  - Can counter: {results['radiant_counters_for']:.2%} of enemy heroes")
        output.append(f"  - Countered by: {results['radiant_counters_against']:.2%} of enemy heroes")
        output.append(f"Role Fit Score: {results['radiant_role_fit']:.2%}")
        output.append("\nRadiant Team Role Assignments:")
        for assignment in results['radiant_assignments']:
            hero = next((h for h in hero_directory if h["id"] == assignment["hero_id"]), None)
            hero_name = hero["displayName"] if hero else f"Hero_{assignment['hero_id']}"
            position_names = {1: "Carry", 2: "Mid", 3: "Offlane", 4: "Soft Support", 5: "Hard Support"}
            output.append(f"  - {hero_name}: {position_names[assignment['position']]} (Score: {assignment['score']:.2%})")
            if assignment['roles']:
                output.append(f"    Roles: {', '.join(assignment['roles'])}")

        # Format Dire team analysis
        output.append("\nDire Team Analysis:")
        output.append("-" * 30)
        output.append(f"Win Probability: {results['dire_win_probability']:.2%}")
        output.append(f"Synergy Score: {results['dire_synergy']:.2%}")
        output.append(f"Counter Analysis:")
        output.append(f"  - Can counter: {results['dire_counters_for']:.2%} of enemy heroes")
        output.append(f"  - Countered by: {results['dire_counters_against']:.2%} of enemy heroes")
        output.append(f"Role Fit Score: {results['dire_role_fit']:.2%}")
        output.append("\nDire Team Role Assignments:")
        for assignment in results['dire_assignments']:
            hero = next((h for h in hero_directory if h["id"] == assignment["hero_id"]), None)
            hero_name = hero["displayName"] if hero else f"Hero_{assignment['hero_id']}"
            position_names = {1: "Carry", 2: "Mid", 3: "Offlane", 4: "Soft Support", 5: "Hard Support"}
            output.append(f"  - {hero_name}: {position_names[assignment['position']]} (Score: {assignment['score']:.2%})")
            if assignment['roles']:
                output.append(f"    Roles: {', '.join(assignment['roles'])}")

        # output.append(f"\nOverall Confidence: {results['confidence']:.2%}")

        return "\n".join(output)

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
        self.matchup_sample_sizes = self._compute_matchup_sample_sizes()
        self.hero_roles = self._build_hero_role_lookup()
        self.missing_matchup_count = 0
        self.total_matchup_count = 0
        self.hero_versatility = self._compute_hero_versatility()

        # Debug: Print matchup data stats
        total_matchups = len(self.matchup_lookup)
        total_samples = sum(self.matchup_sample_sizes.values())
        print(f"\nMatchup Data Stats:")
        print(f"Total unique matchups: {total_matchups}")
        print(f"Total sample size: {total_samples}")
        print(f"Average samples per matchup: {total_samples/total_matchups if total_matchups > 0 else 0:.1f}")

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

    def _build_hero_role_lookup(self) -> Dict[int, Dict[str, int]]:
        """Build a lookup table for hero roles and their levels."""
        role_lookup = {}
        for hero in self.hero_directory:
            role_lookup[hero["id"]] = {
                role["roleId"]: role["level"]
                for role in hero.get("roles", [])
            }
        return role_lookup

    def calculate_hero_scores(self, team: List[int], enemy_team: List[int]) -> float:
        """Calculate average win rates of heroes in the team against enemy team."""
        hero_scores = []
        for hero in team:
            # Calculate average win rate against all enemy heroes
            win_rates = []
            for enemy in enemy_team:
                rate = self.matchup_lookup.get((hero, enemy))
                if rate is None:
                    rate = self.matchup_lookup.get((enemy, hero))
                    if rate is not None:
                        rate = 1.0 - rate
                if rate is not None:
                    win_rates.append(rate)

            if win_rates:
                hero_scores.append(sum(win_rates) / len(win_rates))

        return sum(hero_scores) / len(hero_scores) if hero_scores else 0.5

    def calculate_team_synergy(self, team: List[int]) -> float:
        """Calculate percentage of heroes that have synergies within the team."""
        heroes_with_synergy = set()
        total_possible_pairs = 0

        for i, hero1 in enumerate(team):
            for hero2 in team[i+1:]:
                total_possible_pairs += 1
                # Try both directions for the matchup
                synergy1 = self.matchup_lookup.get((hero1, hero2))
                synergy2 = self.matchup_lookup.get((hero2, hero1))

                # Use the better synergy value if available
                synergy = None
                if synergy1 is not None:
                    synergy = synergy1
                if synergy2 is not None and (synergy is None or synergy2 > synergy):
                    synergy = synergy2

                if synergy is not None and synergy > SYNERGY_THRESHOLD:
                    heroes_with_synergy.add(hero1)
                    heroes_with_synergy.add(hero2)
                    # Debug: Print found synergy
                    hero1_name = next((h["displayName"] for h in self.hero_directory if h["id"] == hero1), f"Hero_{hero1}")
                    hero2_name = next((h["displayName"] for h in self.hero_directory if h["id"] == hero2), f"Hero_{hero2}")
                    print(f"Found synergy: {hero1_name} + {hero2_name} ({synergy:.2%})")

        return len(heroes_with_synergy) / len(team) if team else 0.0

    def calculate_counters_for(self, team: List[int], enemy_team: List[int]) -> float:
        """Calculate how well the team counters enemy heroes using a continuous score."""
        counter_scores = []
        print("\nCalculating counters_for:")
        print(f"Team: {[next((h['displayName'] for h in self.hero_directory if h['id'] == hero), f'Hero_{hero}') for hero in team]}")
        print(f"Enemy: {[next((h['displayName'] for h in self.hero_directory if h['id'] == hero), f'Hero_{hero}') for hero in enemy_team]}")

        for hero in team:
            hero_counter_scores = []
            hero_name = next((h["displayName"] for h in self.hero_directory if h["id"] == hero), f"Hero_{hero}")
            print(f"\nChecking counters for {hero_name}:")

            for enemy in enemy_team:
                # Try both directions for the matchup
                rate1 = self.matchup_lookup.get((hero, enemy))
                rate2 = self.matchup_lookup.get((enemy, hero))

                # Use the better counter value if available
                rate = None
                if rate1 is not None:
                    rate = rate1
                if rate2 is not None and (rate is None or (1.0 - rate2) > rate):
                    rate = 1.0 - rate2

                enemy_name = next((h["displayName"] for h in self.hero_directory if h["id"] == enemy), f"Hero_{enemy}")
                if rate is not None:
                    # Calculate a continuous counter score
                    # If rate > COUNTER_THRESHOLD_FOR, hero counters enemy
                    # Score is 0 at COUNTER_THRESHOLD_FOR and 1 at 1.0
                    counter_score = max(0.0, (rate - COUNTER_THRESHOLD_FOR) / (1.0 - COUNTER_THRESHOLD_FOR))
                    hero_counter_scores.append(counter_score)
                    print(f"  vs {enemy_name}: {rate:.2%} win rate -> {counter_score:.2%} counter score")
                    # Debug: Print found counter
                    if counter_score > 0:
                        print(f"  Found counter: {hero_name} counters {enemy_name} ({rate:.2%}, score: {counter_score:.2%})")
                else:
                    print(f"  vs {enemy_name}: No matchup data found")

            if hero_counter_scores:
                # Average the counter scores for this hero against all enemies
                avg_score = sum(hero_counter_scores) / len(hero_counter_scores)
                counter_scores.append(avg_score)
                print(f"  Average counter score for {hero_name}: {avg_score:.2%}")
            else:
                print(f"  No counter scores for {hero_name}")

        final_score = sum(counter_scores) / len(counter_scores) if counter_scores else 0.0
        print(f"\nFinal counters_for score: {final_score:.2%}")
        return final_score

    def calculate_counters_against(self, team: List[int], enemy_team: List[int]) -> float:
        """Calculate how well the team handles enemy counters using a continuous score."""
        counter_scores = []
        print("\nCalculating counters_against:")
        print(f"Team: {[next((h['displayName'] for h in self.hero_directory if h['id'] == hero), f'Hero_{hero}') for hero in team]}")
        print(f"Enemy: {[next((h['displayName'] for h in self.hero_directory if h['id'] == hero), f'Hero_{hero}') for hero in enemy_team]}")

        for hero in team:
            hero_counter_scores = []
            hero_name = next((h["displayName"] for h in self.hero_directory if h["id"] == hero), f"Hero_{hero}")
            print(f"\nChecking counters against {hero_name}:")

            for enemy in enemy_team:
                # Try both directions for the matchup
                rate1 = self.matchup_lookup.get((enemy, hero))
                rate2 = self.matchup_lookup.get((hero, enemy))

                # Use the better counter value if available
                rate = None
                if rate1 is not None:
                    rate = rate1
                if rate2 is not None and (rate is None or (1.0 - rate2) > rate):
                    rate = 1.0 - rate2

                enemy_name = next((h["displayName"] for h in self.hero_directory if h["id"] == enemy), f"Hero_{enemy}")
                if rate is not None:
                    # Calculate a continuous counter score
                    # If rate < COUNTER_THRESHOLD_AGAINST, hero is countered by enemy
                    # Score is 0 at COUNTER_THRESHOLD_AGAINST and 1 at 0.0
                    counter_score = max(0.0, (COUNTER_THRESHOLD_AGAINST - rate) / COUNTER_THRESHOLD_AGAINST)
                    hero_counter_scores.append(counter_score)
                    print(f"  vs {enemy_name}: {rate:.2%} win rate -> {counter_score:.2%} counter score")
                    # Debug: Print found counter
                    if counter_score > 0:
                        print(f"  Found counter: {enemy_name} counters {hero_name} ({rate:.2%}, score: {counter_score:.2%})")
                else:
                    print(f"  vs {enemy_name}: No matchup data found")

            if hero_counter_scores:
                # Average the counter scores for this hero against all enemies
                avg_score = sum(hero_counter_scores) / len(hero_counter_scores)
                counter_scores.append(avg_score)
                print(f"  Average counter score for {hero_name}: {avg_score:.2%}")
            else:
                print(f"  No counter scores for {hero_name}")

        final_score = sum(counter_scores) / len(counter_scores) if counter_scores else 0.0
        print(f"\nFinal counters_against score: {final_score:.2%}")
        return final_score

    def calculate_role_fit(self, team: List[int]) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate how well heroes fit their positions based on role traits."""
        best_score = 0.0
        best_assignment = None

        # Try all possible hero-to-position assignments
        for position_order in permutations(range(1, 6)):  # 5 positions
            position_scores = []
            hero_assignments = []
            used_heroes = set()

            for pos, hero_id in zip(position_order, team):
                if hero_id in used_heroes:
                    continue

                hero_roles = self.hero_roles.get(hero_id, {})
                position_traits = POSITION_ROLE_TRAITS[pos]

                # Calculate how well hero fits this position
                role_scores = []
                for trait in position_traits:
                    role_level = hero_roles.get(trait, 0)
                    role_scores.append(role_level / 3.0)  # Normalize by max level

                position_score = max(role_scores) if role_scores else 0.0
                position_scores.append(position_score)

                hero_assignments.append({
                    "hero_id": hero_id,
                    "position": pos,
                    "score": position_score,
                    "roles": [trait for trait in position_traits if hero_roles.get(trait, 0) > 0]
                })
                used_heroes.add(hero_id)

            if len(position_scores) == 5:  # All positions filled
                total_score = sum(position_scores) / 5.0
                if total_score > best_score:
                    best_score = total_score
                    best_assignment = hero_assignments

        return (best_score, best_assignment) if best_assignment else (0.0, [])

    def calculate_data_quality_score(self, team1: List[int], team2: List[int]) -> float:
        """Calculate a quality score for the matchup data with improved reliability assessment."""
        total_samples = 0
        valid_matchups = 0
        reliable_matchups = 0

        # Check sample sizes for all hero matchups
        for h1 in team1:
            for h2 in team2:
                # Check both directions of the matchup
                sample_size1 = self.matchup_sample_sizes.get((h1, h2), 0)
                sample_size2 = self.matchup_sample_sizes.get((h2, h1), 0)
                sample_size = max(sample_size1, sample_size2)

                if sample_size > 0:
                    total_samples += sample_size
                    valid_matchups += 1
                    if sample_size >= MIN_SAMPLE_SIZE:
                        reliable_matchups += 1

        # Calculate quality score based on:
        # 1. Number of valid matchups (should be 25 for full team vs team)
        # 2. Number of reliable matchups (with sufficient sample size)
        # 3. Average sample size per matchup
        matchup_coverage = valid_matchups / 25.0  # 5x5 = 25 possible matchups
        reliable_coverage = reliable_matchups / 25.0
        avg_sample_size = total_samples / max(valid_matchups, 1)

        # Normalize sample size (assuming 1000 is a good sample size)
        sample_size_score = min(1.0, avg_sample_size / 1000.0)

        # Combine factors with more weight on reliable matchups
        # Ensure minimum confidence of 0.3 even with poor data
        base_confidence = (matchup_coverage * 0.3 + reliable_coverage * 0.5 + sample_size_score * 0.2)
        return max(0.3, base_confidence)

    def calculate_win_probability(self, radiant_team: List[int], dire_team: List[int]) -> Tuple[float, float, float]:
        """Calculate win probabilities for both teams based on all factors."""
        # Calculate individual scores
        radiant_scores = {
            "hero_scores": self.calculate_hero_scores(radiant_team, dire_team),
            "synergy": self.calculate_team_synergy(radiant_team),
            "counters_for": self.calculate_counters_for(radiant_team, dire_team),
            "counters_against": self.calculate_counters_against(radiant_team, dire_team),
            "role_fit": self.calculate_role_fit(radiant_team)[0]
        }

        dire_scores = {
            "hero_scores": self.calculate_hero_scores(dire_team, radiant_team),
            "synergy": self.calculate_team_synergy(dire_team),
            "counters_for": self.calculate_counters_for(dire_team, radiant_team),
            "counters_against": self.calculate_counters_against(dire_team, radiant_team),
            "role_fit": self.calculate_role_fit(dire_team)[0]
        }

        # Print debug information
        print("\nDebug Scores:")
        print("Radiant Scores:", {k: f"{v:.2%}" for k, v in radiant_scores.items()})
        print("Dire Scores:", {k: f"{v:.2%}" for k, v in dire_scores.items()})

        # Calculate weighted scores
        radiant_total = sum(score * ANALYSIS_WEIGHTS[factor]
                          for factor, score in radiant_scores.items())
        dire_total = sum(score * ANALYSIS_WEIGHTS[factor]
                        for factor, score in dire_scores.items())

        print(f"Weighted Totals - Radiant: {radiant_total:.2%}, Dire: {dire_total:.2%}")

        # Calculate data quality for confidence
        data_quality = self.calculate_data_quality_score(radiant_team, dire_team)
        print(f"Data Quality Score: {data_quality:.2%}")

        # Normalize to probabilities
        total_score = radiant_total + dire_total
        if total_score == 0:
            radiant_prob = 0.5
        else:
            # Calculate base probability
            radiant_prob = radiant_total / total_score

            # Apply confidence scaling
            # This ensures that even with low confidence, we still reflect the team differences
            # but scale them based on our confidence in the data
            deviation = radiant_prob - 0.5
            radiant_prob = 0.5 + (deviation * data_quality)

            # Ensure probabilities stay within reasonable bounds
            radiant_prob = max(0.4, min(0.6, radiant_prob))

        dire_prob = 1.0 - radiant_prob

        return radiant_prob, dire_prob, data_quality

    def calculate_meta_fit_score(self, team: List[int]) -> float:
        """Calculate how well the team composition fits the current meta."""
        # This is a simplified version - in reality, this would use current meta data
        # For now, we'll use hero versatility as a proxy for meta fit
        versatility_scores = [self.hero_versatility.get(hero_id, 0.5) for hero_id in team]
        return sum(versatility_scores) / len(team)

    def analyze_teams(self, radiant_team: List[int], dire_team: List[int]) -> Dict[str, Any]:
        """Perform complete team analysis."""
        # Calculate win probabilities
        radiant_prob, dire_prob, confidence = self.calculate_win_probability(radiant_team, dire_team)

        # Get role assignments
        radiant_role_fit, radiant_assignments = self.calculate_role_fit(radiant_team)
        dire_role_fit, dire_assignments = self.calculate_role_fit(dire_team)

        return {
            "radiant_win_probability": radiant_prob,
            "radiant_synergy": self.calculate_team_synergy(radiant_team),
            "radiant_counters_for": self.calculate_counters_for(radiant_team, dire_team),
            "radiant_counters_against": self.calculate_counters_against(radiant_team, dire_team),
            "radiant_role_fit": radiant_role_fit,
            "radiant_assignments": radiant_assignments,
            "dire_win_probability": dire_prob,
            "dire_synergy": self.calculate_team_synergy(dire_team),
            "dire_counters_for": self.calculate_counters_for(dire_team, radiant_team),
            "dire_counters_against": self.calculate_counters_against(dire_team, radiant_team),
            "dire_role_fit": dire_role_fit,
            "dire_assignments": dire_assignments,
            # "confidence": confidence
        }

    def _compute_matchup_sample_sizes(self) -> Dict[Tuple[int, int], int]:
        """Compute sample sizes for each matchup to assess data reliability."""
        sample_sizes = {}
        for hero_id_str, hero_data in self.matchups.items():
            for section in ['advantage', 'disadvantage']:
                for group in hero_data.get(section, []):
                    for kind in ['with', 'vs']:
                        for matchup in group.get(kind, []):
                            h1 = matchup.get('heroId1')
                            h2 = matchup.get('heroId2')
                            sample_size = matchup.get('sampleSize', 0)
                            if h1 is not None and h2 is not None:
                                sample_sizes[(h1, h2)] = sample_size
        return sample_sizes

    def _compute_hero_versatility(self) -> Dict[int, float]:
        """Compute how versatile each hero is based on their roles."""
        versatility = {}
        for hero in self.hero_directory:
            roles = hero.get("roles", [])
            if roles:
                # Count number of roles with level >= 2
                high_level_roles = sum(1 for r in roles if r.get("level", 0) >= 2)
                # Calculate versatility score (0-1)
                versatility[hero["id"]] = min(1.0, high_level_roles / 3.0)
        return versatility

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Dota 2 team compositions')
    parser.add_argument('--radiant', nargs=5, type=int, required=True, help='List of 5 hero IDs for Radiant team')
    parser.add_argument('--dire', nargs=5, type=int, required=True, help='List of 5 hero IDs for Dire team')
    args = parser.parse_args()

    analyzer = TeamAnalyzer()
    results = analyzer.analyze_teams(args.radiant, args.dire)

    # Use the new as_string method instead of printing directly
    print(TeamAnalyzer.as_string(results, analyzer.hero_directory))
