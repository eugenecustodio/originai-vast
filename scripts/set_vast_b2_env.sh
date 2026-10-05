#!/usr/bin/env bash
# Stores your read-only Backblaze key as Vast.ai account environment variables
# (B2_KEY_ID, B2_APP_KEY). Vast injects them into your workers, so the worker
# template no longer needs to contain the key and can be updated without it.
# The key is typed at hidden prompts; it goes only to Backblaze (to test it) and Vast.
set -euo pipefail

# The Vast CLI (pip install vastai). Override with VAST=/path/to/vastai if it is not on PATH.
VAST="${VAST:-vastai}"

read -rp  "Backblaze keyID: " B2_KEY_ID
read -rsp "Backblaze applicationKey (hidden, just paste and press Enter): " B2_APP_KEY; echo
if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then echo "Both values are required."; exit 1; fi

echo "Checking that the key can read the model checkpoints..."
B2_KEY_ID="$B2_KEY_ID" B2_APP_KEY="$B2_APP_KEY" python3 - <<'EOF'
import base64, json, os, sys, urllib.request
token = base64.b64encode(f"{os.environ['B2_KEY_ID']}:{os.environ['B2_APP_KEY']}".encode()).decode()
try:
    auth = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://api.backblazeb2.com/b2api/v3/b2_authorize_account", headers={"Authorization": "Basic " + token})))
    api = auth["apiInfo"]["storageApi"]
    body = json.dumps({"bucketId": "ef8db658a4e30906a30c0818", "maxFileCount": 5,
                       "prefix": "Thesis_Code/results/within__ffpp_c23__video_swin_b__fixed_v002__seed3109/checkpoints/"}).encode()
    files = json.load(urllib.request.urlopen(urllib.request.Request(
        api["apiUrl"] + "/b2api/v3/b2_list_file_names", data=body, headers={"Authorization": auth["authorizationToken"]})))["files"]
except Exception as exc:
    sys.exit(f"The key did not work: {exc}")
if not any(f["fileName"].endswith("selected.pt") for f in files):
    sys.exit("The key works, but it cannot see the checkpoints. Check the bucket (salik-digma) and prefix (Thesis_Code/results/).")
print("Key OK: it can read the checkpoints.")
EOF

echo "Saving it to your Vast account..."
existing="$("$VAST" show env-vars 2>/dev/null || true)"
for name in B2_KEY_ID B2_APP_KEY; do
  value="${!name}"
  if grep -q "$name" <<<"$existing"; then
    "$VAST" update env-var "$name" "$value" > /dev/null
  else
    "$VAST" create env-var "$name" "$value" > /dev/null
  fi
done
echo "Done. B2_KEY_ID and B2_APP_KEY are saved in your Vast account. Tell Claude \"done\"."
