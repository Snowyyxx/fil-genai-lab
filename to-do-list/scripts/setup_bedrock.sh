#!/usr/bin/env bash
#
# setup_bedrock.sh — get an OPEN-WEIGHT model on Amazon Bedrock working for
# Meowstermind, entirely with `aws` CLI commands.
#
# Targets Llama / Mistral / DeepSeek / Qwen / gpt-oss. Unlike Anthropic models,
# these need no vendor use-case form — just the standard model agreement.
#
#   ./scripts/setup_bedrock.sh                     # inspect only (safe, read-only)
#   ./scripts/setup_bedrock.sh --accept-terms      # also accept the model agreement
#   ./scripts/setup_bedrock.sh --region us-east-1 --model meta.llama3-3-70b-instruct-v1:0
#   ./scripts/setup_bedrock.sh --iam-policy        # print the IAM policy for the app
#
# Read-only by default. Nothing that changes your account runs unless you pass
# --accept-terms (accepting a model agreement is a legal acceptance on your
# account, so it is opt-in on purpose).
#
set -euo pipefail

ACCEPT_TERMS=0
MAKE_POLICY=0
WANT_MODEL=""
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-terms) ACCEPT_TERMS=1; shift ;;
    --iam-policy)   MAKE_POLICY=1; shift ;;
    --region)       REGION="$2"; shift 2 ;;
    --model)        WANT_MODEL="$2"; shift 2 ;;
    -h|--help)      sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

bold=$'\e[1m'; dim=$'\e[2m'; green=$'\e[32m'; yellow=$'\e[33m'; red=$'\e[31m'; off=$'\e[0m'
step() { printf '\n%s▸ %s%s\n' "$bold" "$1" "$off"; }
ok()   { printf '  %s✓%s %s\n' "$green" "$off" "$1"; }
warn() { printf '  %s!%s %s\n' "$yellow" "$off" "$1"; }
die()  { printf '\n%s✗ %s%s\n' "$red" "$1" "$off" >&2; exit 1; }

# --iam-policy is pure output — answer it before touching AWS at all, so it works
# with no credentials and no CLI.
if [[ "$MAKE_POLICY" -eq 1 ]]; then
  cat <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MeowstermindInvokeOpenWeightModels",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/meta.*",
        "arn:aws:bedrock:*::foundation-model/mistral.*",
        "arn:aws:bedrock:*::foundation-model/deepseek.*",
        "arn:aws:bedrock:*::foundation-model/qwen.*",
        "arn:aws:bedrock:*::foundation-model/openai.*",
        "arn:aws:bedrock:*:*:inference-profile/*"
      ]
    },
    {
      "Sid": "MeowstermindDiscoverModels",
      "Effect": "Allow",
      "Action": ["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
      "Resource": "*"
    }
  ]
}
JSON
  cat <<EOF

  ${dim}Save as policy.json, then:${off}
  aws iam create-policy --policy-name MeowstermindBedrock --policy-document file://policy.json
  ${dim}(In Phase 2 this attaches to the ECS task role — no keys in the image.)${off}
EOF
  exit 0
fi

command -v aws >/dev/null || die "aws CLI not found. Install it, or add ~/.local/bin to PATH."

# Keep in sync with MODEL_PREFERENCE / OPEN_WEIGHT_PROVIDERS in backend/zen_cat.py
PY_PREF='PREF = ("gpt-oss-120b","llama4-maverick","llama3-3-70b","llama4-scout","llama3-1-70b",
        "qwen3-235b","mixtral-8x7b","qwen3-32b","deepseek","gpt-oss-20b","mistral-small",
        "llama3-1-8b","llama3-8b","mistral-7b")
OPEN = {"meta","mistral","deepseek","qwen","openai"}
def provider(mid):
    p = mid.split(".")
    if len(p) > 2 and p[0] in ("us","eu","apac"): p = p[1:]
    return p[0] if p else ""
def rank(mid):
    for n, name in enumerate(PREF):
        if name in mid: return (n, mid)
    return (len(PREF), mid)'

# ── 1. who am I ──────────────────────────────────────────────────────────────
step "1/6  Checking credentials"
if ! IDENTITY=$(aws sts get-caller-identity --output json 2>&1); then
  cat <<EOF
$red✗ No usable AWS credentials.$off

Pick one and re-run this script:

  ${bold}IAM user access keys${off}
    aws configure                       # prompts for key, secret, region

  ${bold}IAM Identity Center (SSO)${off}
    aws configure sso                   # then: aws sso login --profile <name>
    export AWS_PROFILE=<name>

  ${bold}Temporary credentials${off}
    export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_SESSION_TOKEN=...

aws said: $(echo "$IDENTITY" | tail -1)
EOF
  exit 1
fi
ACCOUNT=$(echo "$IDENTITY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')
ARN=$(echo "$IDENTITY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')
ok "account $ACCOUNT"
ok "identity $ARN"

REGION="${REGION:-$(aws configure get region 2>/dev/null || true)}"
REGION="${REGION:-us-east-1}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
ok "region  $REGION"
printf '  %sOpen-weight choice is widest in us-east-1 / us-west-2; EU accounts use eu.* profiles.%s\n' "$dim" "$off"

# ── 2. open-weight models offered here ───────────────────────────────────────
step "2/6  Open-weight models offered in $REGION"
MODELS_JSON=$(aws bedrock list-foundation-models --output json 2>&1) \
  || die "list-foundation-models failed — missing bedrock:ListFoundationModels? aws said: $(echo "$MODELS_JSON" | tail -1)"

echo "$MODELS_JSON" | python3 -c "
import json, sys
$PY_PREF
models = json.load(sys.stdin).get('modelSummaries', [])
rows = [m for m in models if provider(m['modelId']) in OPEN
        and 'TEXT' in m.get('outputModalities', ['TEXT'])]
if not rows:
    print('  (none offered here — try --region us-east-1)'); raise SystemExit
for m in sorted(rows, key=lambda m: rank(m['modelId'])):
    kinds = ','.join(m.get('inferenceTypesSupported', [])) or 'INFERENCE_PROFILE only'
    print(f\"  {m['modelId']:<52} {kinds}\")
"

# ── 3. inference profiles (several models are profile-only) ──────────────────
step "3/6  Open-weight inference profiles"
PROFILES_JSON=$(aws bedrock list-inference-profiles --output json 2>&1) || PROFILES_JSON='{}'
echo "$PROFILES_JSON" | python3 -c "
import json, sys
$PY_PREF
try: items = json.load(sys.stdin).get('inferenceProfileSummaries', [])
except Exception: items = []
rows = [p for p in items if provider(p.get('inferenceProfileId','')) in OPEN]
if not rows:
    print('  (none visible — needs bedrock:ListInferenceProfiles, or none exist here)')
for p in sorted(rows, key=lambda p: rank(p['inferenceProfileId'])):
    print(f\"  {p['inferenceProfileId']:<52} {p.get('status','')}\")
"

# ── 4. pick a model and check access ─────────────────────────────────────────
step "4/6  Access status"
if [[ -z "$WANT_MODEL" ]]; then
  WANT_MODEL=$(echo "$MODELS_JSON" | python3 -c "
import json, sys
$PY_PREF
ids = [m['modelId'] for m in json.load(sys.stdin).get('modelSummaries', [])
       if provider(m['modelId']) in OPEN and 'TEXT' in m.get('outputModalities', ['TEXT'])]
print(sorted(ids, key=rank)[0] if ids else '')
")
fi
[[ -n "$WANT_MODEL" ]] || die "No open-weight model offered in $REGION. Try --region us-east-1."
ok "target model: $WANT_MODEL"

availability() {
  aws bedrock get-foundation-model-availability --model-id "$1" --output json 2>/dev/null || echo '{}'
}
AVAIL=$(availability "$WANT_MODEL")
echo "$AVAIL" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for key in ("agreementAvailability", "authorizationStatus", "entitlementAvailability", "regionAvailability"):
    v = d.get(key)
    if isinstance(v, dict): v = v.get("status", v)
    if v: print(f"  {key:<24} {v}")
' || true

AGREED=$(echo "$AVAIL" | python3 -c '
import json,sys
d=json.load(sys.stdin); a=d.get("agreementAvailability") or {}
print("yes" if (a.get("status") if isinstance(a,dict) else a) == "AVAILABLE" else "no")
' 2>/dev/null || echo "no")

# ── 5. accept the model agreement (opt-in) ───────────────────────────────────
step "5/6  Model agreement"
if [[ "$AGREED" == "yes" ]]; then
  ok "already granted — nothing to do"
elif [[ "$ACCEPT_TERMS" -eq 0 ]]; then
  warn "access not granted yet. Re-run with --accept-terms to accept it:"
  printf '     %s%s --accept-terms --model %s%s\n' "$dim" "$0" "$WANT_MODEL" "$off"
else
  OFFERS=$(aws bedrock list-foundation-model-agreement-offers --model-id "$WANT_MODEL" --output json 2>&1) \
    || die "list-foundation-model-agreement-offers failed: $(echo "$OFFERS" | tail -1)"
  TOKEN=$(echo "$OFFERS" | python3 -c '
import json,sys
offers=json.load(sys.stdin).get("offers",[])
print(offers[0]["offerToken"] if offers else "")
' 2>/dev/null || echo "")
  [[ -n "$TOKEN" ]] || die "No agreement offer returned for $WANT_MODEL."
  printf '  accepting terms for %s …\n' "$WANT_MODEL"
  aws bedrock create-foundation-model-agreement --model-id "$WANT_MODEL" --offer-token "$TOKEN" >/dev/null \
    || die "create-foundation-model-agreement failed."
  for _ in $(seq 1 12); do
    sleep 5
    [[ "$(availability "$WANT_MODEL" | python3 -c '
import json,sys
a=(json.load(sys.stdin).get("agreementAvailability") or {})
print((a.get("status") if isinstance(a,dict) else a) or "")' 2>/dev/null)" == "AVAILABLE" ]] && { AGREED=yes; break; }
    printf '  %swaiting for access to propagate…%s\n' "$dim" "$off"
  done
  [[ "$AGREED" == "yes" ]] && ok "access granted" || warn "still pending — re-run in a minute"
fi

# ── 6. prove it works, through the same Converse API the app uses ────────────
step "6/6  Test call (Converse API)"
INVOKE_ID="$WANT_MODEL"
# Models without ON_DEMAND must be called through a regional inference profile.
if ! echo "$MODELS_JSON" | python3 -c "
import json,sys
mid=sys.argv[1]
for m in json.load(sys.stdin).get('modelSummaries',[]):
    if m['modelId']==mid:
        sys.exit(0 if 'ON_DEMAND' in m.get('inferenceTypesSupported',[]) else 1)
sys.exit(1)" "$WANT_MODEL"; then
  PROFILE_ID=$(echo "$PROFILES_JSON" | python3 -c "
import json,sys
mid=sys.argv[1]
try: items=json.load(sys.stdin).get('inferenceProfileSummaries',[])
except Exception: items=[]
print(next((p['inferenceProfileId'] for p in items if p['inferenceProfileId'].endswith(mid)), ''))
" "$WANT_MODEL")
  [[ -n "$PROFILE_ID" ]] && INVOKE_ID="$PROFILE_ID"
fi
ok "calling $INVOKE_ID"

if REPLY=$(aws bedrock-runtime converse \
      --model-id "$INVOKE_ID" \
      --messages '[{"role":"user","content":[{"text":"Reply with exactly one word: purr"}]}]' \
      --inference-config '{"maxTokens":32,"temperature":0.2}' \
      --query 'output.message.content[0].text' --output text 2>&1); then
  printf '  %sthe cat says:%s %s\n' "$green" "$off" "$REPLY"
  cat <<EOF

${green}${bold}Bedrock is ready.${off} Point Meowstermind at it:

  export AWS_REGION=$REGION
  export BEDROCK_MODEL_ID=$INVOKE_ID
  cd backend && uvicorn main:app --reload --port 8000

Then confirm:  curl -s localhost:8000/health   → "detail": null means live.
EOF
else
  warn "call failed:"
  echo "$REPLY" | sed 's/^/     /' | tail -4
  printf '     %sAccessDenied      → IAM lacks bedrock:InvokeModel, or terms not accepted.%s\n' "$dim" "$off"
  printf '     %sValidationException → use the us.*/eu.* inference-profile id, not the bare model id.%s\n' "$dim" "$off"
fi
