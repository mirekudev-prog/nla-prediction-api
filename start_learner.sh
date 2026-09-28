#!/bin/bash
# Start script for Continuous Learner
# DATABASE_URL must be set in environment

if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable not set"
    exit 1
fi

cd /home/user/nla-prediction  # Adjust path
source venv/bin/activate

export PYTHONPATH=/home/user/nla-prediction/src:$PYTHONPATH

python continuous_learner.py