import os
import json
import snowflake.connector
from ollama import Client
from dotenv import load_dotenv
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

load_dotenv()

# =========================================================
# Configuration
# =========================================================

MODEL = "qwen3:4b"

SAMPLE_N = int(os.getenv("SAMPLE_N", "5"))

TOPICS = [
    "food quality",
    "delivery",
    "pricing",
    "service",
    "packaging",
    "other"
]

# Local Windows:
#   http://localhost:11434
#
# Airflow Docker -> Ollama on Windows:
#   http://host.docker.internal:11434
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

client = Client(host=OLLAMA_HOST)


# =========================================================
# System Prompt
# =========================================================

SYSTEM_PROMPT = f"""
Classify the customer review for a food delivery app.

Return ONLY valid JSON with exactly these four fields:

- sentiment_label: "positive", "negative", or "neutral"
- sentiment_score: number from -1.0 to 1.0
- topic: exactly one of {TOPICS}
- key_issue: main problem in 6 words or fewer, or null if there is no problem

Rules:
- sentiment_label must be exactly "positive", "negative", or "neutral"
- sentiment_score must be between -1.0 and 1.0
- topic must be exactly one of {TOPICS}
- key_issue must contain 6 words or fewer
- Use null when there is no clear problem
- Do not include explanations
- Do not include markdown
- Do not include extra fields
- Return JSON only

Example:
{{
    "sentiment_label": "positive",
    "sentiment_score": 0.8,
    "topic": "food quality",
    "key_issue": null
}}
"""


# =========================================================
# Snowflake Connection
# =========================================================

def get_connection():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


# =========================================================
# Create Output Table
# =========================================================

def create_output_table(cursor):

    cursor.execute(
        "CREATE SCHEMA IF NOT EXISTS ZOMATO.AI"
    )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ZOMATO.AI.REVIEW_ENRICHED (
            REVIEW_ID STRING,
            SENTIMENT_LABEL STRING,
            SENTIMENT_SCORE FLOAT,
            TOPIC STRING,
            KEY_ISSUE STRING,
            MODEL STRING,
            ENRICHED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)


# =========================================================
# Get Reviews That Have Not Been Enriched
# =========================================================

def get_reviews_to_enrich(cursor):

    cursor.execute(f"""
        SELECT
            REVIEW_ID,
            COMMENT
        FROM ZOMATO.RAW.REVIEWS r
        WHERE NOT EXISTS (
            SELECT 1
            FROM ZOMATO.AI.REVIEW_ENRICHED e
            WHERE e.REVIEW_ID = r.REVIEW_ID
        )
        LIMIT {SAMPLE_N}
    """)

    return cursor.fetchall()


# =========================================================
# Classify Review Using Ollama
# =========================================================

def classify_review(comment):

    print("Sending review to Ollama...")

    response = client.chat(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": comment
            }
        ],

        # Force structured JSON output
        format="json",

        # Qwen3 does not need reasoning for this task
        think=False,

        options={
            "temperature": 0,
            "num_predict": 150
        }
    )

    answer = response["message"]["content"]

    # repr() makes empty responses visible as ''
    print(
        f"Ollama response: {repr(answer)}"
    )

    # -----------------------------------------------------
    # Check for empty response
    # -----------------------------------------------------

    if not answer or not answer.strip():

        raise ValueError(
            "Ollama returned an empty response."
        )

    # -----------------------------------------------------
    # Parse JSON
    # -----------------------------------------------------

    try:

        labels = json.loads(answer)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Ollama returned invalid JSON: {answer}"
        ) from e

    return labels


# =========================================================
# Validate Classification
# =========================================================

def validate_classification(labels):

    # -----------------------------------------------------
    # Check required fields
    # -----------------------------------------------------

    required_fields = [
        "sentiment_label",
        "sentiment_score",
        "topic",
        "key_issue"
    ]

    for field in required_fields:

        if field not in labels:

            raise ValueError(
                f"Missing required field: {field}"
            )

    # -----------------------------------------------------
    # Validate sentiment label
    # -----------------------------------------------------

    sentiment_label = labels["sentiment_label"]

    if sentiment_label not in [
        "positive",
        "negative",
        "neutral"
    ]:

        raise ValueError(
            f"Invalid sentiment_label: "
            f"{sentiment_label}"
        )

    # -----------------------------------------------------
    # Validate sentiment score
    # -----------------------------------------------------

    try:

        sentiment_score = float(
            labels["sentiment_score"]
        )

    except (TypeError, ValueError):

        raise ValueError(
            f"Invalid sentiment_score: "
            f"{labels['sentiment_score']}"
        )

    if not -1.0 <= sentiment_score <= 1.0:

        raise ValueError(
            f"sentiment_score must be between "
            f"-1.0 and 1.0: {sentiment_score}"
        )

    # -----------------------------------------------------
    # Validate topic
    # -----------------------------------------------------

    topic = labels["topic"]

    if topic not in TOPICS:

        raise ValueError(
            f"Invalid topic: {topic}"
        )

    # -----------------------------------------------------
    # Validate key issue
    # -----------------------------------------------------

    key_issue = labels.get("key_issue")

    if key_issue is not None:

        key_issue = str(key_issue).strip()

        if len(key_issue.split()) > 6:

            raise ValueError(
                f"key_issue exceeds 6 words: "
                f"{key_issue}"
            )

    # -----------------------------------------------------
    # Return cleaned values
    # -----------------------------------------------------

    return (
        sentiment_label,
        sentiment_score,
        topic,
        key_issue
    )


# =========================================================
# Save Results to Snowflake
# =========================================================

def save_results(cursor, results):

    if not results:

        print("No results to save.")

        return

    print(
        f"Saving {len(results)} "
        f"enriched reviews to Snowflake..."
    )

    cursor.executemany(
        """
        INSERT INTO ZOMATO.AI.REVIEW_ENRICHED
        (
            REVIEW_ID,
            SENTIMENT_LABEL,
            SENTIMENT_SCORE,
            TOPIC,
            KEY_ISSUE,
            MODEL
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        results
    )


# =========================================================
# Main Pipeline
# =========================================================

def main():

    conn = None
    cursor = None

    try:

        # -------------------------------------------------
        # Startup Information
        # -------------------------------------------------

        print(
            "========================================"
        )
        print(
            "Zomato AI Review Enrichment"
        )
        print(
            "========================================"
        )

        print(
            f"Ollama host : {OLLAMA_HOST}"
        )

        print(
            f"Ollama model: {MODEL}"
        )

        print(
            f"Sample size : {SAMPLE_N}"
        )

        print()

        # -------------------------------------------------
        # Check Ollama
        # -------------------------------------------------

        print("Checking Ollama...")

        models = client.list()

        installed_models = [
            model.model
            for model in models.models
        ]

        if MODEL not in installed_models:

            raise RuntimeError(
                f"Ollama model '{MODEL}' "
                f"is not installed.\n"
                f"Installed models: "
                f"{installed_models}"
            )

        print(
            f"Ollama is ready. "
            f"Model '{MODEL}' found."
        )

        print()

        # -------------------------------------------------
        # Connect to Snowflake
        # -------------------------------------------------

        print(
            "Connecting to Snowflake..."
        )

        conn = get_connection()

        cursor = conn.cursor()

        print(
            "Snowflake connection successful."
        )

        print()

        # -------------------------------------------------
        # Create AI Output Table
        # -------------------------------------------------

        create_output_table(cursor)

        # -------------------------------------------------
        # Get Reviews
        # -------------------------------------------------

        reviews = get_reviews_to_enrich(
            cursor
        )

        if len(reviews) == 0:

            print(
                "No new reviews to enrich."
            )

            return

        print(
            f"Found {len(reviews)} "
            f"reviews to enrich."
        )

        print()

        # -------------------------------------------------
        # Process Reviews
        # -------------------------------------------------

        results = []

        for review_id, comment in reviews:

            print(
                "----------------------------------------"
            )

            print(
                f"Review ID: {review_id}"
            )

            print(
                f"Review   : {comment}"
            )

            try:

                # -----------------------------------------
                # Send review to Ollama
                # -----------------------------------------

                labels = classify_review(
                    comment
                )

                print(
                    f"Classification: {labels}"
                )

                # -----------------------------------------
                # Validate response
                # -----------------------------------------

                (
                    sentiment_label,
                    sentiment_score,
                    topic,
                    key_issue
                ) = validate_classification(
                    labels
                )

                # -----------------------------------------
                # Prepare Snowflake row
                # -----------------------------------------

                results.append(
                    (
                        review_id,
                        sentiment_label,
                        sentiment_score,
                        topic,
                        key_issue,
                        MODEL
                    )
                )

            except Exception as e:

                print(
                    f"Error processing review "
                    f"{review_id}: {e}"
                )

        # -------------------------------------------------
        # Save Results
        # -------------------------------------------------

        save_results(
            cursor,
            results
        )

        # -------------------------------------------------
        # Commit Transaction
        # -------------------------------------------------

        conn.commit()

        print()

        print(
            "========================================"
        )

        print(
            f"Successfully saved "
            f"{len(results)} "
            f"enriched reviews."
        )

        print(
            "========================================"
        )

    except Exception as e:

        print()

        print(
            "========================================"
        )

        print(
            "PIPELINE ERROR"
        )

        print(
            "========================================"
        )

        print(e)

        if conn:

            conn.rollback()

        raise

    finally:

        # -------------------------------------------------
        # Close Snowflake Resources
        # -------------------------------------------------

        if cursor:

            cursor.close()

        if conn:

            conn.close()


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()