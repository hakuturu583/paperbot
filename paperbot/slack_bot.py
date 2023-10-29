import os

# Use the package we installed
from slack_bolt import App
import json
import logging
import re
from slack_bolt import App, Say, BoltContext
from slack_sdk import WebClient
from typing import Callable

# Initializes your app with your bot token and signing secret
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
)

# def get_thread_ts(body: dict):
#     if "thread_ts" in body["event"]:
#         print("thread ts = ")
#         return body["event"]["thread_ts"]
#     else:
#         print("ts = ")
#         return body["event"]["ts"]


@app.event("app_mention")
def mention_handler(
    body: dict,
    client: WebClient,
    context: BoltContext,
    logger: logging.Logger,
    say: Say,
):
    mention = body["event"]
    text = mention["text"]
    channel = mention["channel"]
    thread_ts = 0
    if "thread_ts" in mention:
        thread_ts = mention["thread_ts"]
    else:
        thread_ts = mention["ts"]
    history = client.conversations_replies(
        channel=body["event"]["channel"], ts=thread_ts, limit=1000
    )
    print(history)
    # for message in history["messages"]:
    #     print("=======================================================================================================---")
    #     print(message)
    # print(get_thread_ts(body))
    if "files" in body["event"]:
        client.reactions_add(
            channel=context.channel_id, timestamp=mention["ts"], name="eyes"
        )
    else:
        print(f"メンションされました: {text}")
        print(mention["event_ts"])
        say(text=text, channel=channel, thread_ts=mention["ts"])


# @app.event("file_shared")
# def add_reaction(
#     body: dict, client: WebClient, context: BoltContext, logger: logging.Logger
# ):
#     print(body)
#     api_response = client.reactions_add(
#         channel=context.channel_id,
#         timestamp=body["event_time"],
#         name="eyes",
#     )
#     logger.info(f"api_response: {api_response}")


# Start your app
if __name__ == "__main__":
    app.start(port=3000)
