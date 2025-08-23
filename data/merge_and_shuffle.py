from argparse import ArgumentParser
import pandas as pd
import random

parser = ArgumentParser()
parser.add_argument("--path", type=str,nargs='+', default=[])
args = parser.parse_args()

paths = args.path

all_data = []

for path in paths:
    data = pd.read_parquet(path)
    data = data.to_dict(orient="records")
    all_data.extend(data)

random.seed(42)
random.shuffle(all_data)

df = pd.DataFrame(all_data)
df.to_parquet("./data/merged/0822-QASPER.parquet")