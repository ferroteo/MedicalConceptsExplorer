# Set up virtual environment and install dependencies
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate || { echo "error"; exit 1; }
    echo "Virtual environment 'venv' already exists. Skipping setup."
else
    python -m venv venv || { echo "error"; exit 1; }
    source venv/bin/activate || { echo "error"; exit 1; }
    pip install --upgrade pip || { echo "error"; exit 1; }
    pip install -r requirements.txt || { echo "error"; exit 1; }
    echo "Virtual environment 'venv' created and dependencies installed."
fi

# Pre-compute embeddings and log output with timing
LOG_FILE="Initialize.log"
(
    set -o pipefail
    { time -p python -u PreComputeEmbeddings.py; } 2>&1 | tee -a "$LOG_FILE"
) || { echo "error"; exit 1; }
echo "Pre-computation of embeddings complete. Check '$LOG_FILE' for details."
