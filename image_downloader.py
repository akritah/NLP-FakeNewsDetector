import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import urllib.error
import time
import numpy as np
import sys

parser = argparse.ArgumentParser(description='r/Fakeddit image downloader')

parser.add_argument('type', type=str, help='train, validate, or test')
parser.add_argument('--output-dir', default='images', help='directory for downloaded images')
parser.add_argument('--max-retries', type=int, default=5)
parser.add_argument('--delay', type=float, default=0.2, help='delay between requests in seconds')

args = parser.parse_args()

df = pd.read_csv(args.type, sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)
downloaded = 0
skipped = 0
failed = 0
for index, row in df.iterrows():
  has_image = str(row["hasImage"]).strip().lower() == "true"
  if has_image and row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    image_path = os.path.join(args.output_dir, row["id"] + ".jpg")
    if os.path.exists(image_path):
      skipped += 1
      pbar.update(1)
      continue
    request = urllib.request.Request(
      image_url,
      headers={"User-Agent": "Mozilla/5.0 (Fakeddit dataset downloader)"}
    )
    for attempt in range(args.max_retries):
      try:
        with urllib.request.urlopen(request, timeout=30) as response, open(image_path, "wb") as output:
          output.write(response.read())
        downloaded += 1
        break
      except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
        if attempt == args.max_retries - 1:
          failed += 1
        else:
          time.sleep(max(args.delay, 2 ** attempt))
    time.sleep(args.delay)
  pbar.update(1)
print(f"done: downloaded={downloaded}, skipped={skipped}, failed={failed}")
