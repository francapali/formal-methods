import pandas as pd
import pm4py
import os
import random
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "steam_event_log.csv"
OUTPUT_PREPROCESSED = BASE_DIR / "output" / "preprocessed_data"
OUTPUT_NETS = BASE_DIR / "output" / "petri_nets"

# If Graphviz was not added to PATH, change this line to add the path to the Graphviz/bin directory.
# os.environ["PATH"] += os.pathsep + "...Graphviz/bin"


def augment_data(df, num_new_cases):
    """Generate new cases by cloning existing users with varied timestamps."""
    new_rows = []
    last_id = df['case_id'].max()
    unique_ids = df['case_id'].unique()

    for i in range(1, num_new_cases + 1):
        random_user = random.choice(unique_ids)
        user_data = df[df['case_id'] == random_user].copy()
        new_case_id = last_id + i
        time_offset = timedelta(days=random.randint(1, 60))
        user_data['case_id'] = new_case_id
        user_data['timestamp'] = user_data['timestamp'] + time_offset
        new_rows.append(user_data)

    return pd.concat([df] + new_rows).reset_index(drop=True)


def get_variants_for_llm(event_log):
    """Extract most frequent process variants for LLM analysis."""
    from pm4py.algo.filtering.log.variants import variants_filter
    variants = variants_filter.get_variants(event_log)
    sorted_variants = sorted(variants.items(), key=lambda x: len(x[1]), reverse=True)

    summary = "\n--- LLM REASONING EXTRACT ---\n"
    summary += f"Dataset: Steam Event Log. Total cases analyzed: {len(event_log)}\n"
    summary += "Most common variants:\n"
    for i, (variant, occurrences) in enumerate(sorted_variants[:10]):
        summary += f"{i+1}. Path: {variant} - Occurrences: {len(occurrences)}\n"
    summary += "------------------------------------\n"
    return summary


def evaluate_model(event_log, net, im, fm, name):
    """Calculate quality metrics for the model (Fitness and Precision)."""
    fitness = pm4py.fitness_token_based_replay(event_log, net, im, fm)
    precision = pm4py.precision_token_based_replay(event_log, net, im, fm)
    print(f"[{name}] Fitness: {fitness['log_fitness']:.3f} | Precision: {precision:.3f}")


def run_steam_mining_complete():
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found at {INPUT_FILE}")
        return
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for folder in [OUTPUT_PREPROCESSED, OUTPUT_NETS]:
        folder.mkdir(parents=True, exist_ok=True)

    df_enriched = augment_data(df, num_new_cases=100)
    df_enriched.to_csv(OUTPUT_PREPROCESSED / "steam_enriched_log.csv", index=False)

    event_log = pm4py.format_dataframe(
        df_enriched, case_id='case_id', activity_key='activity', timestamp_key='timestamp'
    )
    log_formal = pm4py.convert_to_event_log(event_log)

    print(f"Analyzing {df_enriched['case_id'].nunique()} cases...")

    # ALPHA MINER
    net_a, im_a, fm_a = pm4py.discover_petri_net_alpha(log_formal)
    pm4py.save_vis_petri_net(net_a, im_a, fm_a, str(OUTPUT_NETS / "alpha_steam.png"))
    evaluate_model(log_formal, net_a, im_a, fm_a, "Alpha")

    # HEURISTIC MINER
    net_h, im_h, fm_h = pm4py.discover_petri_net_heuristics(log_formal)
    pm4py.save_vis_petri_net(net_h, im_h, fm_h, str(OUTPUT_NETS / "heuristic_steam.png"))
    evaluate_model(log_formal, net_h, im_h, fm_h, "Heuristic")

    # INDUCTIVE MINER
    net_i, im_i, fm_i = pm4py.discover_petri_net_inductive(log_formal)
    pm4py.save_vis_petri_net(net_i, im_i, fm_i, str(OUTPUT_NETS / "inductive_steam.png"))
    evaluate_model(log_formal, net_i, im_i, fm_i, "Inductive")

    llm_report = get_variants_for_llm(log_formal)
    print(llm_report)
    print("Models saved in 'petri_nets/'. Process completed!")


if __name__ == "__main__":
    run_steam_mining_complete()
