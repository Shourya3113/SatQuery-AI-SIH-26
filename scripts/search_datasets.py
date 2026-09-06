"""
Search Hugging Face datasets API for BigEarthNet, VRSBench, RSVQA, and CDVQA.
"""

import urllib.request
import json

queries = ["BigEarthNet", "VRSBench", "RSVQA", "CDVQA"]

for q in queries:
    url = f"https://huggingface.co/api/datasets?search={q}&limit=5"
    req = urllib.request.Request(url, headers={"User-Agent": "SatQuery-AI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"=== Query: {q} (Found: {len(data)}) ===")
            for item in data:
                print(f"  - ID: {item.get('id')} | Downloads: {item.get('downloads', 0)} | Likes: {item.get('likes', 0)}")
    except Exception as e:
        print(f"Error querying {q}: {e}")
