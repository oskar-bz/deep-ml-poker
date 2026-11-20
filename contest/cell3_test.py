# CELL 3 (keep or delete; play_local lines are stripped on submit)
# Strength + think-time vs the house bots. Worst-case think time must stay well under 1 ms.
from pokerbot_sdk import play_local

play_local(MyBot(), tables=20)
