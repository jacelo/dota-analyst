import requests
import time
import os
from itertools import combinations, product
from math import exp
from dotenv import load_dotenv

load_dotenv()

STRATZ_API_URL = os.getenv("STRATZ_API_URL")
REMOVED = os.getenv("REMOVED")

# Convert environment variables to floats with default values
WEIGHT_INDIVIDUAL = float(os.getenv("WEIGHT_INDIVIDUAL", "1.0"))
WEIGHT_SYNERGY = float(os.getenv("WEIGHT_SYNERGY", "1.0"))
WEIGHT_COUNTER = float(os.getenv("WEIGHT_COUNTER", "1.0"))
FACTOR = float(os.getenv("FACTOR", "2.0"))

def fetch_hero_constants():
    """Fetch hero ID to display name map"""
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
        "Authorization": f"Bearer {REMOVED}",
        "User-Agent": "STRATZ_API",
        "Accept": "application/json",
    }

    res = requests.post(STRATZ_API_URL, json={"query": query}, headers=headers)
    if res.status_code != 200:
        raise Exception(f"Query failed: {res.status_code} - {res.text}")

    data = res.json()["data"]["constants"]["heroes"]
    return {hero["id"]: hero["displayName"] for hero in data}

def get_hero_vs_hero_matchup(hero_id: int):
    """Fetch hero matchup stats from STRATZ API for a specific hero."""
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
        "Authorization": f"Bearer {REMOVED}",
        "User-Agent": "STRATZ_API",
        "Accept": "application/json",
    }
    res = requests.post(STRATZ_API_URL, json={"query": query}, headers=headers)

    if res.status_code == 200:
        return res.json()
    else:
        raise Exception(f"Query failed: {res.status_code} - {res.text}")

def parse_matchup_data(raw_data, hero_id):
    synergy = {}
    counter = {}
    matchup = raw_data["data"]["heroStats"]["heroVsHeroMatchup"]

    # The API returns a list of matchups for each section
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

def calculate_team_score(team, enemy, winrates, synergies, counters,
                         weight_individual=WEIGHT_INDIVIDUAL, weight_synergy=WEIGHT_SYNERGY, weight_counter=WEIGHT_COUNTER):
    score = 0.0

    # Individual hero winrate contributions
    for h in team:
        score += weight_individual * (winrates.get(h, 0.5) - 0.5)

    # Synergy (within same team)
    for h1, h2 in combinations(team, 2):
        key = tuple(sorted((h1, h2)))
        if key in synergies:
            score += weight_synergy * (synergies[key] - 0.5)

    # Countering enemy heroes
    for h1, h2 in product(team, enemy):
        key = (h1, h2)
        if key in counters:
            score += weight_counter * (counters[key] - 0.5)

    return score

def win_probability(score_radiant, score_dire, factor=FACTOR):
    diff = score_radiant - score_dire
    return 1 / (1 + exp(-diff * factor))

def build_full_matchup_matrix(hero_ids):
    """Builds winrate, synergy, and counter dictionaries for all given heroes"""
    winrates = {}
    synergies = {}
    counters = {}

    for hero_id in hero_ids:
        print(f"Fetching data for hero {hero_id}...")
        raw = get_hero_vs_hero_matchup(hero_id)
        sy, co = parse_matchup_data(raw, hero_id)

        # Cache individual winrate as avg of with/against (fallback)
        winrates[hero_id] = 0.5 + (
            sum((w - 0.5) for w in sy.values()) +
            sum((w - 0.5) for w in co.values())
        ) / (len(sy) + len(co) + 1e-5)

        synergies.update(sy)
        counters.update(co)
        time.sleep(0.5)  # Avoid throttling

    return winrates, synergies, counters

def evaluate_draft(radiant_ids, dire_ids):
    all_hero_ids = set(radiant_ids + dire_ids)
    winrates, synergies, counters = build_full_matchup_matrix(all_hero_ids)

    radiant_score = calculate_team_score(radiant_ids, dire_ids, winrates, synergies, counters)
    dire_score = calculate_team_score(dire_ids, radiant_ids, winrates, synergies, counters)

    prob_radiant = win_probability(radiant_score, dire_score)
    prob_dire = 1 - prob_radiant

    return prob_radiant, prob_dire

if __name__ == "__main__":
    hero_directory = fetch_hero_constants()
    radiant = [1, 2, 3, 4, 5]   # e.g., Anti-Mage, Axe, Bane, Bloodseeker, Crystal Maiden
    dire = [6, 7, 8, 9, 10]     # e.g., Drow, Earthshaker, Ember, Invoker, Juggernaut

    prob_radiant, prob_dire = evaluate_draft(radiant, dire)

    # Display output as:
    # Radiant: Hero1, Hero2, Hero3, Hero4, Hero5
    # Dire: Hero6, Hero7, Hero8, Hero9, Hero10
    #
    # Radiant win probability: 50%
    # Dire win probability: 50%

    print(f"\nRadiant: {', '.join(hero_directory[hero_id] for hero_id in radiant)}")
    print(f"Dire: {', '.join(hero_directory[hero_id] for hero_id in dire)}")
    print(f"\nRadiant win probability: {prob_radiant:.2%}")
    print(f"Dire win probability:    {prob_dire:.2%}")
