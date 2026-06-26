python run_glan.py --samples 5

python run_glan.py --concurrency 48 --batch 64 --samples 5

nohup python run_glan.py --runs 10 --samples 60 --concurrency 58 --batch 64 --output-dir v1 > glan.log 2>&1 &