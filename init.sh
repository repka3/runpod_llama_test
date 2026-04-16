#!/usr/bin/env bash
set -euo pipefail

LLAMA_REPO_URL="${LLAMA_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_DIR="${LLAMA_DIR:-$PWD/llama.cpp}"
MODELS_DIR="${MODELS_DIR:-$PWD/models}"
MODEL_INPUT="${1:-${MODEL_URL:-}}"
JOBS="${JOBS:-}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
LLAMA_CMAKE_EXTRA_ARGS="${LLAMA_CMAKE_EXTRA_ARGS:-}"

usage() {
  cat <<'EOF'
Usage:
  ./init.sh <huggingface-url-or-repo-id>

Examples:
  ./init.sh https://huggingface.co/mradermacher/Gemma-4-31B-Cognitive-Unshackled-GGUF/tree/main
  ./init.sh mradermacher/Gemma-4-31B-Cognitive-Unshackled-GGUF

Optional environment variables:
  LLAMA_DIR                Where llama.cpp will be cloned/built.
  MODELS_DIR               Where downloaded model files will be stored.
  JOBS                     Parallel compile jobs. Defaults to all CPUs.
  HF_TOKEN                 Hugging Face token for gated/private repos.
  MODEL_CHOICE             Non-interactive main model choice number.
  VISION_CHOICE            Non-interactive vision choice number.
  AUTO_DOWNLOAD_VISION     Set to 1 to auto-download a detected vision file.
  LLAMA_CMAKE_EXTRA_ARGS   Extra CMake flags for llama.cpp.
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

detect_jobs() {
  if [ -n "$JOBS" ]; then
    printf '%s\n' "$JOBS"
    return
  fi

  if need_cmd nproc; then
    nproc
    return
  fi

  getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n'
}

ensure_system_packages() {
  local -a missing=()

  need_cmd git || missing+=(git)
  need_cmd cmake || missing+=(cmake)
  need_cmd python3 || missing+=(python3 python3-pip)
  need_cmd curl || missing+=(curl)
  need_cmd g++ || missing+=(build-essential)
  need_cmd make || missing+=(build-essential)

  if [ "${#missing[@]}" -eq 0 ]; then
    return
  fi

  if need_cmd apt-get; then
    log "Installing missing packages: ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y "${missing[@]}"
    return
  fi

  fail "Missing required commands (${missing[*]}), and no supported package manager was found."
}

clone_or_update_llama() {
  if [ -d "$LLAMA_DIR/.git" ]; then
    log "Updating llama.cpp in $LLAMA_DIR"
    git -C "$LLAMA_DIR" fetch --all --tags
    git -C "$LLAMA_DIR" pull --ff-only
  elif [ -e "$LLAMA_DIR" ]; then
    fail "$LLAMA_DIR exists but is not a git repository."
  else
    log "Cloning llama.cpp into $LLAMA_DIR"
    git clone "$LLAMA_REPO_URL" "$LLAMA_DIR"
  fi
}

build_llama() {
  local jobs
  jobs="$(detect_jobs)"

  log "Building llama.cpp with CUDA using $jobs parallel jobs"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ${LLAMA_CMAKE_EXTRA_ARGS}
  cmake --build "$LLAMA_DIR/build" --config Release -j "$jobs"
}

prompt_for_model_input() {
  if [ -n "$MODEL_INPUT" ]; then
    return
  fi

  read -r -p "Hugging Face model URL or repo id: " MODEL_INPUT
  [ -n "$MODEL_INPUT" ] || fail "A Hugging Face URL or repo id is required."
}

download_model_files() {
  mkdir -p "$MODELS_DIR"

  MODEL_INPUT="$MODEL_INPUT" MODELS_DIR="$MODELS_DIR" HF_TOKEN="$HF_TOKEN" python3 - <<'PY'
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MODEL_INPUT = os.environ["MODEL_INPUT"].strip()
MODELS_DIR = Path(os.environ["MODELS_DIR"]).expanduser().resolve()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "").strip()
VISION_CHOICE = os.environ.get("VISION_CHOICE", "").strip()
AUTO_DOWNLOAD_VISION = os.environ.get("AUTO_DOWNLOAD_VISION", "").strip().lower() in {"1", "true", "yes", "y"}


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def build_headers() -> dict[str, str]:
    headers = {"User-Agent": "runpod-llama-init/1.0"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


def http_json(url: str) -> object:
    request = urllib.request.Request(url, headers=build_headers())
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def head_size(url: str) -> int | None:
    request = urllib.request.Request(url, headers=build_headers(), method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def parse_model_input(model_input: str) -> tuple[str, str, str]:
    if re.fullmatch(r"[^/\s]+/[^/\s]+", model_input):
        return model_input, "main", ""

    parsed = urllib.parse.urlparse(model_input)
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        fail("Input must be a Hugging Face repo id like owner/repo or a huggingface.co URL.")

    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        fail("Could not parse the Hugging Face repository from the URL.")

    repo_id = "/".join(parts[:2])
    revision = "main"
    subdir = ""

    if len(parts) >= 4 and parts[2] in {"tree", "resolve"}:
        revision = parts[3]
        if len(parts) > 4:
            subdir = "/".join(parts[4:])
    elif len(parts) > 2:
        subdir = "/".join(parts[2:])

    return repo_id, revision, subdir


def fetch_repo_files(repo_id: str, revision: str, subdir: str) -> list[dict]:
    encoded_repo = repo_id.replace("/", "%2F")
    encoded_revision = urllib.parse.quote(revision, safe="")
    api_url = f"https://huggingface.co/api/models/{encoded_repo}/revision/{encoded_revision}?full=true"

    try:
        payload = http_json(api_url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            fail(f"Model repo not found: {repo_id}@{revision}")
        if exc.code == 401:
            fail("The repo is gated or private. Export HF_TOKEN with access to it.")
        raise

    siblings = payload.get("siblings") or []
    prefix = f"{subdir.rstrip('/')}/" if subdir else ""
    files: list[dict] = []
    for entry in siblings:
        path = entry.get("rfilename") or entry.get("path") or ""
        if not path:
            continue
        if subdir:
            if path == subdir:
                continue
            if not path.startswith(prefix):
                continue
        files.append(
            {
                "path": path,
                "name": path.rsplit("/", 1)[-1],
                "size": entry.get("size"),
            }
        )

    if not files:
        fail("No files were found in the selected repo or subdirectory.")
    return files


def model_candidates(files: list[dict]) -> list[dict]:
    model_exts = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx")
    primary = [
        item for item in files
        if item["name"].lower().endswith(model_exts)
        and not re.search(r"(mmproj|vision|projector)", item["name"], re.IGNORECASE)
    ]

    gguf_only = [item for item in primary if item["name"].lower().endswith(".gguf")]
    if gguf_only:
        primary = gguf_only

    primary.sort(key=lambda item: (item["size"] is None, -(item["size"] or 0), item["name"].lower()))
    return primary


def vision_candidates(files: list[dict]) -> list[dict]:
    candidates = []
    for item in files:
        name = item["name"]
        lower = name.lower()
        if not lower.endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth")):
            continue
        if not re.search(r"(mmproj|vision|projector)", lower):
            continue
        if not re.search(r"(f16|fp16|float16)", lower):
            continue
        candidates.append(item)

    candidates.sort(key=lambda item: (item["size"] is None, -(item["size"] or 0), item["name"].lower()))
    return candidates


def print_choices(title: str, items: list[dict]) -> None:
    print(f"\n{title}")
    for idx, item in enumerate(items, start=1):
        size = item["size"]
        if size is None:
            size = head_size(item["download_url"])
            item["size"] = size
        print(f"  [{idx}] {item['path']}  ({human_size(size)})")


def choose_item(items: list[dict], env_choice: str, prompt: str, allow_skip: bool = False) -> dict | None:
    if not items:
        return None

    if env_choice:
        if allow_skip and env_choice.lower() in {"skip", "none", "0"}:
            return None
        if env_choice.isdigit() and 1 <= int(env_choice) <= len(items):
            return items[int(env_choice) - 1]
        fail(f"Invalid choice '{env_choice}' for {prompt}.")

    while True:
        answer = input(prompt).strip()
        if allow_skip and answer.lower() in {"", "skip", "none", "n", "no", "0"}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]
        print("Please enter a valid number.")


def select_vision_file(items: list[dict]) -> dict | None:
    if not items:
        return None

    if AUTO_DOWNLOAD_VISION and len(items) == 1:
        return items[0]

    if len(items) == 1 and not VISION_CHOICE:
        answer = input(f"\nDownload companion vision FP16 file '{items[0]['path']}'? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return items[0]
        return None

    print_choices("Available FP16 vision companion files:", items)
    return choose_item(items, VISION_CHOICE, "Pick a vision file number, or press Enter to skip: ", allow_skip=True)


def ensure_download_size(items: list[dict]) -> None:
    for item in items:
        if item["size"] is None:
            item["size"] = head_size(item["download_url"])


def add_download_urls(repo_id: str, revision: str, files: list[dict]) -> None:
    encoded_revision = urllib.parse.quote(revision, safe="/")
    for item in files:
        encoded_path = urllib.parse.quote(item["path"], safe="/")
        item["download_url"] = f"https://huggingface.co/{repo_id}/resolve/{encoded_revision}/{encoded_path}?download=true"


def download_with_system_tool(repo_id: str, revision: str, file_item: dict) -> None:
    destination = MODELS_DIR / file_item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    downloader = shutil.which("wget") or shutil.which("curl")
    if not downloader:
        fail("Neither wget nor curl is installed.")

    print(f"\nDownloading {file_item['path']} -> {destination}")
    if downloader.endswith("wget"):
        command = [downloader, "-c", "-O", str(destination)]
        if HF_TOKEN:
            command.extend(["--header", f"Authorization: Bearer {HF_TOKEN}"])
        command.append(file_item["download_url"])
    else:
        command = [downloader, "-L", "-C", "-", "-o", str(destination)]
        if HF_TOKEN:
            command.extend(["-H", f"Authorization: Bearer {HF_TOKEN}"])
        command.append(file_item["download_url"])

    subprocess.run(command, check=True)


repo_id, revision, subdir = parse_model_input(MODEL_INPUT)
print(f"Repo: {repo_id}")
print(f"Revision: {revision}")
if subdir:
    print(f"Subdirectory: {subdir}")

all_files = fetch_repo_files(repo_id, revision, subdir)
add_download_urls(repo_id, revision, all_files)

models = model_candidates(all_files)
if not models:
    fail("No downloadable model files were found.")

print_choices("Available model files:", models)
selected_model = choose_item(models, MODEL_CHOICE, "Pick a model number: ")

vision_files = vision_candidates(all_files)
selected_vision = select_vision_file(vision_files)

ensure_download_size([selected_model] + ([selected_vision] if selected_vision else []))
download_with_system_tool(repo_id, revision, selected_model)
if selected_vision:
    download_with_system_tool(repo_id, revision, selected_vision)
else:
    print("\nNo FP16 vision companion file selected.")
PY
}

main() {
  if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
  fi

  ensure_system_packages
  prompt_for_model_input
  clone_or_update_llama
  build_llama
  download_model_files

  log "Done"
  printf 'llama.cpp build: %s\n' "$LLAMA_DIR/build"
  printf 'models dir: %s\n' "$MODELS_DIR"
}

main "$@"
