# Assignment 2: Model Training with Azure ML

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 80.87% |
| Test F1 Score | 89.09% |
| Test AUC | 74.01% |

## Model
- **Algorithm**: Random Forest
- **Training Job**: `jolly_seal_25vgwz88kz`
- **Registered Model**: `amazon_review_sentiment_model:1`

## How to Reproduce
```bash
az ml job create --file jobs/train_raw_final.yml
```

## Author
Nouresham Katrmiz - DSAI3202 - Winter 2026
