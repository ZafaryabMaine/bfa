# Usage

## Run SST-2 (default)
```bash
python -u run.py --max_examples=512

python -u run.py \
  --dataset=yelp_review_full \
  --split=test \
  --model_name=rttl-ai/bert-base-uncased-yelp-reviews \
  --max_examples=512
