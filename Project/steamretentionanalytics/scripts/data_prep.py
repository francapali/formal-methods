import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

# CONFIGURAZIONE
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "steam-200k.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "steam_event_log.csv"


def generate_log():
    print("Caricamento dataset...")
    df = pd.read_csv(INPUT_FILE, header=None, names=['user_id', 'game', 'behavior', 'hours', 'zero'])

    top_users = df['user_id'].unique()[:2000]
    df = df[df['user_id'].isin(top_users)]

    events = []
    base_date = datetime(2024, 1, 1) # Data di partenza fittizia

    print(f"Generazione eventi per {len(top_users)} utenti...")

    # Raggruppiamo per utente
    for user, group in df.groupby('user_id'):
        # Ogni utente inizia la sua storia in un giorno casuale dell'anno
        current_time = base_date + timedelta(days=random.randint(0, 300))

        # Lista dei giochi interagiti dall'utente
        games = group['game'].unique()

        for game in games:
            # --- EVENTO 1: PURCHASE ---
            # Assumiamo che ogni gioco in lista sia stato comprato
            # Aggiungiamo un po' di tempo casuale tra un acquisto e l'altro
            current_time += timedelta(hours=random.randint(1, 48))

            events.append({
                'case_id': str(user),      # L'utente è il "Caso"
                'activity': 'Purchase Game',
                'timestamp': current_time,
                'game': game,
                'hours_played': 0
            })

            # --- EVENTO 2: PLAY (Se esiste) ---
            # Cerchiamo se c'è una riga 'play' per questo gioco
            play_row = group[(group['game'] == game) & (group['behavior'] == 'play')]

            if not play_row.empty:
                hours = play_row.iloc[0]['hours']

                # Regola: Se ha giocato, l'evento avviene DOPO l'acquisto
                play_time = current_time + timedelta(hours=random.randint(1, 72))

                events.append({
                    'case_id': str(user),
                    'activity': 'Start Playing',
                    'timestamp': play_time,
                    'game': game,
                    'hours_played': hours
                })

                # --- EVENTO 3: HIGH ENGAGEMENT / DLC (Regola di Business) ---
                # Se ha giocato più di 50 ore, simuliamo un evento di fidelizzazione
                if hours > 50:
                    dlc_time = play_time + timedelta(days=random.randint(5, 20))
                    events.append({
                        'case_id': str(user),
                        'activity': 'Purchase DLC/Season Pass',
                        'timestamp': dlc_time,
                        'game': game,
                        'hours_played': hours
                    })

                # --- EVENTO 4: CHURN / ABANDON (Regola di Business) ---
                # Se ha giocato meno di 2 ore (Refund window)
                elif hours < 2.0:
                    churn_time = play_time + timedelta(hours=2)
                    events.append({
                        'case_id': str(user),
                        'activity': 'Abandon Game (Refund Risk)',
                        'timestamp': churn_time,
                        'game': game,
                        'hours_played': hours
                    })

    # Creazione DataFrame finale
    log_df = pd.DataFrame(events)

    # Ordiniamo per data (FONDAMENTALE per il Process Mining)
    log_df = log_df.sort_values(by=['case_id', 'timestamp'])

    # Formattazione Timestamp per pm4py
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

    print(f"Salvataggio di {len(log_df)} eventi in {OUTPUT_FILE}...")
    log_df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    generate_log()
