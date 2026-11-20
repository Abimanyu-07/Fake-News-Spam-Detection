# Dataset Notes

The raw datasets are **not** committed to GitHub. Download them yourself and
place them in `data/raw/`.

## 1. Fake and Real News dataset

- Commonly published on Kaggle as "Fake and Real News Dataset".
- Expected files:
  - `data/raw/Fake.csv`: fake articles
  - `data/raw/True.csv`: real articles
- Expected columns: `title`, `text`, `subject`, `date`.
- Encoding: UTF-8.

## 2. SMS Spam Collection dataset (UCI)

- UCI Machine Learning Repository, "SMS Spam Collection".
- Expected file: `data/raw/spam.csv`
- Columns: `v1` (label: `ham` or `spam`), `v2` (message), plus unnamed extra
  columns that are dropped during cleaning.
  
## Known limitations of both datasets

- News: labels describe the source or the collection process. They are not a
  verified fact-check of each article.
- SMS: the messages come from a limited number of contributors and time
  periods, so the model may not generalise to other regions or languages.

Record the actual row counts you observe in the EDA and preprocessing
notebooks. Do not copy numbers from anywhere else.
