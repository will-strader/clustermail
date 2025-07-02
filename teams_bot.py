"""
teams_bot.py
FastAPI wrapper for a Microsoft Teams bot that uses semantic email search.

HOW TO USE (quick version)
1.  Create an Azure Bot resource (Bot Channel Registration) in the Azure portal.
    - Note the Microsoft APP ID and APP PASSWORD (client secret).
    - Enable the Teams channel.
2.  Set environment variables on your hosting platform (Azure App Service, Render, Fly.io, etc.):
        MICROSOFT_APP_ID = <your bot App ID>
        MICROSOFT_APP_PASSWORD = <your bot Client Secret>
3.  Deploy this app (Dockerfile or gunicorn) at an HTTPS URL.
    Example Docker CMD:
        uvicorn teams_bot:app --host 0.0.0.0 --port 8080
4.  In Azure -> Bot → Settings -> **Messaging endpoint**:
        https://<your-domain>/api/messages
5.  In Teams (same tenant) add the bot (
    “Apps -> Built by your org -> Add”).

Users can now DM the bot or @-mention it in a channel:
    @EmailInsightBot invoice overdue | 20 | 7
The bot replies with the top 20 emails (in cluster 7) sorted by similarity to
“invoice overdue”.
"""

import os
import semantic_search as ss
from fastapi import FastAPI, Request, Response, HTTPException
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity, ActivityTypes

# Load credentials from env
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PW = os.getenv("MICROSOFT_APP_PASSWORD", "")
if not (APP_ID and APP_PW):
    raise RuntimeError(
        "MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD env vars must be set to run the Teams bot."
    )

settings = BotFrameworkAdapterSettings(APP_ID, APP_PW)
adapter = BotFrameworkAdapter(settings)

#  FastAPI app + Bot logic class
app = FastAPI(title="Email Insight Teams Bot")


class EmailInsightBot:
    async def on_turn(self, turn: TurnContext):
        if turn.activity.type != ActivityTypes.message:
            return

        # Parse the user message: "query | top | cluster" (top & cluster optional)
        text = (turn.activity.text or "").strip()
        # If bot was mentioned, strip the mention text
        if turn.activity.entities:
            for ent in turn.activity.entities:
                if ent.get("type") == "mention":
                    mention_text = ent.get("text", "")
                    text = text.replace(mention_text, "").strip()

        parts = [p.strip() for p in text.split("|")]
        query = parts[0] if parts else ""
        top = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
        cluster = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

        if not query and cluster is None:
            await turn.send_activity(
                "Please provide a search query, e.g. `invoice overdue | 15 | 7`"
            )
            return

        # Call the existing semantic search helper
        df = ss.search_api(query=query, top_k=top, cluster_id=cluster)
        if df.empty:
            await turn.send_activity("No results found.")
            return

        # Build reply text (Teams markdown)
        bullets = [
            f"• **{r.similarity:.2f}** — {r.body[:80].replace('\n', ' ')}…"
            for _, r in df.iterrows()
        ]
        cluster_info = f" in cluster {cluster}" if cluster is not None else ""
        header = f"Top {len(df)} results for '{query or '[all]'}'{cluster_info}:\n\n"
        reply = header + "\n".join(bullets)

        await turn.send_activity(reply)


bot = EmailInsightBot()


@app.post("/api/messages")
async def messages(req: Request):
    body = await req.json()
    activity = Activity().deserialize(body)
    auth = req.headers.get("Authorization", "")

    try:
        response = await adapter.process_activity(activity, auth, bot.on_turn)
        if response:
            return Response(status_code=response.status)
        return Response(status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
