import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "steam-200k.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "steam_event_log.csv"


def generate_log():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE, header=None, names=['user_id', 'game', 'behavior', 'hours', 'zero'])

    top_users = df['user_id'].unique()[:2000]
    df = df[df['user_id'].isin(top_users)]

    events = []
    base_date = datetime(2024, 1, 1)

    print(f"Generating events for {len(top_users)} users...")
    for user, group in df.groupby('user_id'):
        current_time = base_date + timedelta(days=random.randint(0, 300))
        games = group['game'].unique()

        for game in games:
            current_time += timedelta(hours=random.randint(1, 48))

            events.append({
                'case_id': str(user),
                'activity': 'Purchase Game',
                'timestamp': current_time,
                'game': game,
                'hours_played': 0
            })

            play_row = group[(group['game'] == game) & (group['behavior'] == 'play')]

            if not play_row.empty:
                hours = play_row.iloc[0]['hours']
                play_time = current_time + timedelta(hours=random.randint(1, 72))

                events.append({
                    'case_id': str(user),
                    'activity': 'Start Playing',
                    'timestamp': play_time,
                    'game': game,
                    'hours_played': hours
                })

                if hours > 50:
                    dlc_time = play_time + timedelta(days=random.randint(5, 20))
                    events.append({
                        'case_id': str(user),
                        'activity': 'Purchase DLC/Season Pass',
                        'timestamp': dlc_time,
                        'game': game,
                        'hours_played': hours
                    })

                elif hours < 2.0:
                    churn_time = play_time + timedelta(hours=2)
                    events.append({
                        'case_id': str(user),
                        'activity': 'Abandon Game (Refund Risk)',
                        'timestamp': churn_time,
                        'game': game,
                        'hours_played': hours
                    })

    log_df = pd.DataFrame(events)
    log_df = log_df.sort_values(by=['case_id', 'timestamp'])
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

    print(f"Saving {len(log_df)} events to {OUTPUT_FILE}...")
    log_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    generate_log()
