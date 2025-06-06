Given 2 lists of heroes, with 5 heroes in each list, calculate the win rate of each of the teams
Connect to Stratz API to get each hero's information and statistics

How to calculate
1. Pull individual hero win rates
```
individual_score = (win_rate - 0.5) * weight
```
2. Aggregate team synergy using with win rates
```
synergy_score = average(winRateHeroId1, winRateHeroId2) - 0.5
```
3. Calculate counter advantage using vs win rates
```
counter_score = winRateHeroId1 - 0.5 (if HeroId1 is in Radiant)
counter_score = 0.5 - winRateHeroId1 (if HeroId1 is in Dire)
```
4. Combine all scores and convert to probability
```
team_draft_score = avg_individual_win_rate + total_team_synergy_score + total_counter_advantage_score
```

Stratz API notes:
Ensure all requests have a User-Agent header: User-Agent: STRATZ_API
API endpoint: https://api.stratz.com/graphql
Graphql query:
```
query {
  heroStats{
    heroVsHeroMatchup(heroId:86){
      advantage{
        with{
          heroId1
          heroId2
					winRateHeroId1
          winRateHeroId2
          winsAverage
        }
        vs {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
      }
      disadvantage{
        with{
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
        vs {
          heroId1
          heroId2
          winRateHeroId1
          winRateHeroId2
          winsAverage
        }
      }
    }
  }
}
```

# Key Terms Used in the Calculation
**1. Weight**

Definition:
A scalar multiplier applied to specific components of the draft evaluation (e.g., individual hero win rate, synergy score, counter score) to control their relative influence on the final score.

Usage:
If you assign higher weight to individual win rates, you are saying that individual performance matters more than synergy or counters. You could also assign different weights to different score components, e.g.:
```
total_score = (weight_individual * individual_score
             + weight_synergy * synergy_score
             + weight_counter * counter_score)
```
Example:
```
weight_individual = 1.0
weight_synergy = 0.5
weight_counter = 0.8
```
These weights allow you to "tune" the model based on what data you trust most or find most predictive.

**2. Factor**

Definition:
A scalar used inside the sigmoid function to control the sensitivity of the win probability output to the difference in draft scores between the two teams.

Formula Context:
```
from math import exp

def win_probability(score_radiant, score_dire):
    diff = score_radiant - score_dire
    return 1 / (1 + exp(-diff * factor))

    A higher factor means even small score differences result in strong win probabilities (e.g., 80%+).

    A lower factor makes the output closer to 50/50, reducing confidence in small differences.
```

Example:
```
Score Difference	Factor	Win Probability
0.5	1	62%
0.5	2	73%
0.5	5	92%
```