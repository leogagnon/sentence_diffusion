import wandb
import argparse

api = wandb.Api()

parser = argparse.ArgumentParser(description="List W&B run IDs for a given sweep_id")
# ...existing code...
parser.add_argument("sweep_id", help="Value for config.sweep_id to filter runs")
args = parser.parse_args()
filters = {
    "config.sweep_id": args.sweep_id,
}
runs = api.runs(f"guillaume-lajoie/sentence_diffusion", filters=filters)
for run in runs:
    print(run.id, end=" ")
