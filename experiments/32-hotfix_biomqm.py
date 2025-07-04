import argparse
import csv

args = argparse.ArgumentParser()
args.add_argument("files", nargs="+")
args = args.parse_args()

for file in args.files:
  print("Converting", file)
  with open(file, "r") as f:
    data = list(csv.DictReader(f))
    for line in data:
      for key in ["score", "score2", "score3", "score4", "score5", "score6"]:
        if key in line:
          score = float(line[key])
          line[key] = score if score > 0 else 100 + score

  with open(file, "w") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

"""
Sample usage (dangerous, operates in place!)

python3 32-hotfix_biomqm.py data/csv/*.csv
"""
