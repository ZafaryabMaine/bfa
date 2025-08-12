   ## Project Structure
   - `src/` - Core implementation modules
   - `log/` - Experiment logs  
   - `plots/` - Generated visualizations
   - `metrics/` - Experiment metrics
   - `dataset/` - Downloaded datasets
   - `model/` - Model checkpoints

# Usage
## Run SST-2 (default)
<!-- sst2_170021.log -->
```bash
python -u run.py --max_examples=512
```

## Run yelp_review_full (5 class)
<!-- yelp5_170106.log -->
```bash
python -u run.py --dataset=yelp_review_full --split=test --model_name=rttl-ai/bert-base-uncased-yelp-reviews --max_examples=512
```