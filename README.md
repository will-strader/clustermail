# ClusterMail – Email Insight Engine

Turn thousands of raw e‑mails into actionable insights with **Sentence‑BERT
embeddings, HDBSCAN clustering, and a Streamlit explorer** – plus a
Microsoft Teams bot for chat‑based search.

---

## Key features
| Module | What it does |
|--------|--------------|
| `bert_hdbscan.py` | Parses >9 k Enron e‑mails, creates SBERT embeddings (`all‑MiniLM‑L6‑v2`), clusters with UMAP → HDBSCAN, and saves artifacts. |
| `streamlit.py` | Interactive UI – free‑text search, cluster filter, CSV download – deployable on Streamlit Cloud or HF Spaces. |
| `semantic_search.py` | Re‑usable search CLI + `search_api()` helper (cosine similarity on embeddings). |
| `teams_bot.py` | FastAPI bot adapter: query the corpus directly from Microsoft Teams (`query | top | cluster`). |



## Quick start

```bash
# Clone & set up Python env (3.12 recommended)
git clone https://github.com/will-strader/clustermail.git
cd clustermail
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate clusters & embeddings (one‑time, ~1 min on CPU)
python bert_hdbscan.py

# Launch the Streamlit demo
streamlit run streamlit.py      # then open http://localhost:8501
```

---

## Data

* **Dataset** – 9 913 cleaned e‑mails from the public [Enron corpus](https://www.cs.cmu.edu/~enron/).  
* Outputs saved to `data/`:
  * `email_embeddings.npy` – 9913 × 384 SBERT vectors  
  * `emails_with_clusters.csv` – parsed fields + `cluster` id

---

## Deployment

### Streamlit Cloud

1. Push the repo to GitHub.  
2. Add `runtime.txt` with `python-3.12.3`.  
3. New app ▸ select `streamlit.py` ▸ Deploy.  
4. First build (~5 min) downloads model; subsequent reloads are ≈ 5 s.

### Microsoft Teams bot (optional)

1. Create an **Azure Bot (Bot Channel Registration)** → copy `APP_ID` & `PASSWORD`.  
2. Deploy with Docker (Render / Fly / Azure App Service):

   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   ENV PORT=8080
   CMD uvicorn teams_bot:app --host 0.0.0.0 --port $PORT
   ```

3. Set env vars: `MICROSOFT_APP_ID`, `MICROSOFT_APP_PASSWORD`.  
4. In the Azure Bot blade: **Messaging endpoint** → `https://<domain>/api/messages`.  
5. Add the bot to Teams → `@EmailInsightBot invoice overdue | 20 | 7`.



## Architecture

```
CSV  →  email_preprocessing  →  SBERT embeddings (npy)
                      ↘              ↘
                       bert_hdbscan   semantic_search
                                         ↘
             Streamlit UI  ←  search_api  →  Teams Bot
```



## Example insight

*Cluster 7 (≈ 80 e‑mails) surfaces “VaR / Market Risk” discussions – traders requesting
VaR reports, debating penalties, and planning methodology changes.*



## License

MIT


## Author

Will Strader 2025