import os
import json
import requests
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import argparse

class DataManager:
    """
    A class for managing Dota 2 hero data and matchups.
    This version focuses on role-based analysis and simplified data structures.
    """
    def __init__(self):
        """Initialize the DataManager with configuration."""
        load_dotenv()
        self.STRATZ_API_URL = os.getenv("STRATZ_API_URL")
        self.STRATZ_API_KEY = os.getenv("STRATZ_API_KEY")
        self.data_dir = "data"
        self.hero_directory_file = os.path.join(self.data_dir, "hero_directory.json")
        self.hero_matchups_file = os.path.join(self.data_dir, "hero_matchups.json")

    def get_hero_directory(self) -> Dict[str, Any]:
        """
        Get the hero directory data from the API or local file.

        Returns:
            Dict[str, Any]: Hero directory data containing hero information and roles.
        """
        if not os.path.exists(self.hero_directory_file):
            self.fetch_and_save_hero_directory()

        with open(self.hero_directory_file, 'r') as f:
            return json.load(f)

    def get_matchups(self) -> Dict[str, Any]:
        """
        Get the hero matchup data from the API or local file.

        Returns:
            Dict[str, Any]: Hero matchup data containing win rates and synergies.
        """
        if not os.path.exists(self.hero_matchups_file):
            self.fetch_and_save_matchups()

        with open(self.hero_matchups_file, 'r') as f:
            return json.load(f)

    def fetch_and_save_hero_directory(self) -> None:
        """Fetch hero directory data from the STRATZ API and save it locally."""
        query = """
        {
            constants {
                heroes {
                    id
                    displayName
                    shortName
                    roles {
                        roleId
                        level
                    }
                }
            }
        }
        """

        response = requests.post(
            self.STRATZ_API_URL,
            json={"query": query},
            headers={
                "Authorization": f"Bearer {self.STRATZ_API_KEY}",
                "User-Agent": "STRATZ_API",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )

        if response.status_code != 200:
            raise Exception(f"Failed to fetch hero directory: {response.text}")

        data = response.json()
        heroes = data["data"]["constants"]["heroes"]

        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.hero_directory_file, 'w') as f:
            json.dump(heroes, f, indent=2)

    def fetch_and_save_matchups(self) -> None:
        """Fetch hero matchup data from the STRATZ API and save it locally."""
        # First get the hero directory to know which heroes to fetch
        hero_directory = self.get_hero_directory()
        all_matchups = {}

        print("Fetching matchup data for each hero...")
        for hero in hero_directory:
            hero_id = hero["id"]
            print(f"Fetching matchups for {hero['displayName']} (ID: {hero_id})...")

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
                                winsAverage
                            }}
                            vs {{
                                heroId1
                                heroId2
                                winRateHeroId1
                                winRateHeroId2
                                winsAverage
                            }}
                        }}
                        disadvantage {{
                            with {{
                                heroId1
                                heroId2
                                winRateHeroId1
                                winRateHeroId2
                                winsAverage
                            }}
                            vs {{
                                heroId1
                                heroId2
                                winRateHeroId1
                                winRateHeroId2
                                winsAverage
                            }}
                        }}
                    }}
                }}
            }}
            """

            response = requests.post(
                self.STRATZ_API_URL,
                json={"query": query},
                headers={
                    "Authorization": f"Bearer {self.STRATZ_API_KEY}",
                    "User-Agent": "STRATZ_API",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )

            if response.status_code != 200:
                print(f"Warning: Failed to fetch matchups for hero {hero_id}: {response.text}")
                continue

            data = response.json()
            if "data" in data and "heroStats" in data["data"]:
                all_matchups[str(hero_id)] = data["data"]["heroStats"]["heroVsHeroMatchup"]

        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.hero_matchups_file, 'w') as f:
            json.dump(all_matchups, f, indent=2)

    def get_hero_roles(self, hero_id: int) -> List[Dict[str, Any]]:
        """
        Get the roles for a specific hero.

        Args:
            hero_id (int): The ID of the hero.

        Returns:
            List[Dict[str, Any]]: List of roles with their levels.
        """
        hero_directory = self.get_hero_directory()
        hero_data = next(
            (h for h in hero_directory if h["id"] == hero_id),
            None
        )
        return hero_data.get("roles", []) if hero_data else []

    def get_team_roles(self, team_heroes: List[int]) -> Dict[str, int]:
        """
        Get the role distribution for a team.

        Args:
            team_heroes (List[int]): List of hero IDs in the team.

        Returns:
            Dict[str, int]: Dictionary mapping role IDs to their total level in the team.
        """
        role_distribution = {}
        for hero_id in team_heroes:
            roles = self.get_hero_roles(hero_id)
            for role in roles:
                role_id = role["roleId"]
                role_distribution[role_id] = role_distribution.get(role_id, 0) + role["level"]
        return role_distribution

    def is_data_loaded(self) -> bool:
        """
        Check if all required data files exist.

        Returns:
            bool: True if all required data files exist, False otherwise.
        """
        return (os.path.exists(self.hero_directory_file) and
                os.path.exists(self.hero_matchups_file))

    def ensure_data_loaded(self) -> None:
        """Ensure all required data files exist, fetching them if necessary."""
        if not self.is_data_loaded():
            self.fetch_and_save_hero_directory()
            self.fetch_and_save_matchups()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dota 2 Data Manager')
    parser.add_argument('--force', action='store_true', help='Force reload all data from API')
    args = parser.parse_args()

    data_manager = DataManager()

    if args.force:
        print("Forcing data reload...")
        data_manager.fetch_and_save_hero_directory()
        data_manager.fetch_and_save_matchups()
    else:
        print("Checking if data is loaded...")
        data_manager.ensure_data_loaded()

    print("Data loaded successfully!")
