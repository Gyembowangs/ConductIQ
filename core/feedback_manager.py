import csv
import uuid
from pathlib import Path
from datetime import datetime, timezone


PROJECT_DIR = Path(__file__).resolve().parent.parent

FEEDBACK_DIR = PROJECT_DIR / "data" / "feedback"

FEEDBACK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEEDBACK_FILE = FEEDBACK_DIR / "feedback.csv"


FEEDBACK_FIELDS = [
    "feedback_id",
    "timestamp",
    "conversation_id",
    "question",
    "agent_response",
    "rating",
    "issue_category",
    "comments",
    "review_status"
]


def save_feedback(
    conversation_id,
    question,
    agent_response,
    rating,
    issue_category="",
    comments=""
):
    """
    Save student feedback for a ConductIQ response.
    """

    if rating not in ["positive", "negative"]:
        raise ValueError(
            "Rating must be positive or negative."
        )

    record = {
        "feedback_id": str(uuid.uuid4()),

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "conversation_id": conversation_id,

        "question": question,

        "agent_response": agent_response,

        "rating": rating,

        "issue_category": issue_category,

        "comments": comments,

        "review_status": (
            "review_required"
            if rating == "negative"
            else "recorded"
        )
    }

    file_exists = FEEDBACK_FILE.exists()

    with open(
        FEEDBACK_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FEEDBACK_FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

    return record


def load_feedback():

    if not FEEDBACK_FILE.exists():
        return []

    import pandas as pd

    df = pd.read_csv(FEEDBACK_FILE)

    return df.to_dict("records")


def get_review_queue():

    records = load_feedback()

    return [
        record
        for record in records
        if record.get("review_status") == "review_required"
    ]


def feedback_summary():

    records = load_feedback()

    if not records:
        return {
            "total_feedback": 0,
            "positive": 0,
            "negative": 0,
            "review_required": 0
        }

    return {
        "total_feedback": len(records),

        "positive": sum(
            r["rating"] == "positive"
            for r in records
        ),

        "negative": sum(
            r["rating"] == "negative"
            for r in records
        ),

        "review_required": sum(
            r["review_status"] == "review_required"
            for r in records
        )
    }