import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from helpers import extract_email_fields

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "emails_with_clusters.csv"
NPY_PATH = DATA_DIR / "email_embeddings.npy"

def fetch_new_messages() -> list[str]:
    """
    Your company-specific hook:
        - IMAP search SINCE last_date
        - Gmail API
        - Outlook Graph API
        - Etc.
    Return list of raw e-mail strings.
    """
    return []   # Replace with real retrieval code for your system

def main():
    raw_new = fetch_new_messages()
    if not raw_new:
        print("No new mail.")
        return

    df_old = pd.read_csv(CSV_PATH)
    last_id = df_old.index.max()

    df_new = pd.DataFrame(extract_email_fields(raw_new))
    df_new["cluster"] = -1         # placeholder until re-clustered
    df_new.index = range(last_id + 1, last_id + 1 + len(df_new))

    # Encode
    emb_new = MODEL.encode(df_new.body.tolist(), batch_size=64, show_progress_bar=True)

    # Append to CSV / NPY
    df_new.to_csv(CSV_PATH, mode="a", header=False, index=False)
    emb_old = np.load(NPY_PATH)
    emb_all = np.vstack([emb_old, emb_new])
    np.save(NPY_PATH, emb_all)
    print(f"Appended {len(df_new)} e-mails.")

if __name__ == "__main__":
    main()