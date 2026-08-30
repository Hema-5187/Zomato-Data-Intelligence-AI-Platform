import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from ollama import Client
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# Configuration
# =========================================================

EMBEDDING_MODEL = "all-minilm:latest"
CHAT_MODEL = "qwen3:4b"

# Start with 50 while testing.
# Once everything works, change this to 500.
NEW_REVIEWS = int(
    os.getenv("NEW_REVIEWS", "50")
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

# Number of reviews sent to Ollama per embedding request.
EMBED_BATCH_SIZE = int(
    os.getenv("EMBED_BATCH_SIZE", "25")
)

CACHE_FILE = "review_embeddings.parquet"

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

client = Client(
    host=OLLAMA_HOST
)


# =========================================================
# Read Reviews From Snowflake
# =========================================================

def read_reviews_from_snowflake():

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        WHERE COMMENT IS NOT NULL
        ORDER BY REVIEW_ID
        LIMIT {NEW_REVIEWS}
    """

    cursor = None

    try:

        cursor = conn.cursor()

        df = (
            cursor
            .execute(query)
            .fetch_pandas_all()
        )

    finally:

        if cursor:
            cursor.close()

        conn.close()

    # Snowflake returns uppercase column names.
    # Convert them to lowercase for Python.
    df.columns = [
        col.lower()
        for col in df.columns
    ]

    return df


# =========================================================
# Generate Embeddings Using Ollama
# =========================================================

def embed(
    texts,
    batch_size=EMBED_BATCH_SIZE
):

    if not texts:
        return []

    all_embeddings = []

    total = len(texts)

    print(
        f"Generating embeddings for "
        f"{total} texts..."
    )

    # -----------------------------------------------------
    # Process texts in small batches
    # -----------------------------------------------------

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = texts[start:end]

        print(
            f"Embedding "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        try:

            response = client.embed(
                model=EMBEDDING_MODEL,
                input=batch
            )

        except Exception as e:

            raise RuntimeError(
                f"Ollama embedding failed "
                f"for batch "
                f"{start + 1}-{end}: {e}"
            ) from e

        batch_embeddings = response[
            "embeddings"
        ]

        if len(batch_embeddings) != len(batch):

            raise RuntimeError(
                f"Embedding count mismatch "
                f"for batch "
                f"{start + 1}-{end}. "
                f"Expected {len(batch)}, "
                f"received "
                f"{len(batch_embeddings)}."
            )

        all_embeddings.extend(
            batch_embeddings
        )

    print(
        f"Successfully generated "
        f"{len(all_embeddings)} embeddings."
    )

    return all_embeddings


# =========================================================
# Load Reviews + Embeddings
# =========================================================

@st.cache_data
def load_reviews():

    # -----------------------------------------------------
    # Use existing cache if available
    # -----------------------------------------------------

    if os.path.exists(CACHE_FILE):

        print(
            f"Loading cached embeddings "
            f"from {CACHE_FILE}"
        )

        df = pd.read_parquet(
            CACHE_FILE
        )

        # Basic cache validation
        required_columns = [
            "review_id",
            "city",
            "rating",
            "comment",
            "embedding"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            raise RuntimeError(
                "Cached embedding file is missing "
                f"columns: {missing_columns}"
            )

        print(
            f"Loaded {len(df)} "
            f"cached reviews."
        )

        return df

    # -----------------------------------------------------
    # Read reviews from Snowflake
    # -----------------------------------------------------

    print(
        "Reading reviews from Snowflake..."
    )

    df = read_reviews_from_snowflake()

    if df.empty:

        raise RuntimeError(
            "No reviews were found "
            "in Snowflake."
        )

    print(
        f"Found {len(df)} reviews."
    )

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    print(
        f"Generating embeddings for "
        f"{len(df)} reviews..."
    )

    texts = (
        df["comment"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = embed(
        texts,
        batch_size=EMBED_BATCH_SIZE
    )

    # -----------------------------------------------------
    # Validate embedding count
    # -----------------------------------------------------

    if len(embeddings) != len(df):

        raise RuntimeError(
            f"Embedding count mismatch. "
            f"Reviews: {len(df)}, "
            f"Embeddings: {len(embeddings)}"
        )

    # -----------------------------------------------------
    # Store embeddings
    # -----------------------------------------------------

    df["embedding"] = embeddings

    # -----------------------------------------------------
    # Save local cache
    # -----------------------------------------------------

    df.to_parquet(
        CACHE_FILE,
        index=False
    )

    print(
        f"Saved embeddings to "
        f"{CACHE_FILE}"
    )

    return df


# =========================================================
# Cosine Similarity
# =========================================================

def cosine_similarity(
    vec_a,
    vec_b
):

    vec_a = np.asarray(
        vec_a,
        dtype=np.float32
    )

    vec_b = np.asarray(
        vec_b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(vec_a)
        *
        np.linalg.norm(vec_b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(
            vec_a,
            vec_b
        )
        /
        denominator
    )


# =========================================================
# Find Similar Reviews
# =========================================================

def find_similar_reviews(
    question,
    df
):

    print(
        "Generating question embedding..."
    )

    question_vector = embed(
        [question],
        batch_size=1
    )[0]

    scores = []

    for review_vector in df[
        "embedding"
    ]:

        scores.append(
            cosine_similarity(
                question_vector,
                review_vector
            )
        )

    results = df.copy()

    results["score"] = scores

    return (
        results
        .nlargest(
            TOP_K,
            "score"
        )
    )


# =========================================================
# Ask Qwen3 Using Retrieved Reviews
# =========================================================

def ask_llm(
    question,
    top_reviews
):

    context_parts = []

    for _, row in (
        top_reviews.iterrows()
    ):

        context_parts.append(
            f"""
Review ID: {row['review_id']}
City: {row['city']}
Rating: {row['rating']} stars
Review: {row['comment']}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
You are a concise customer review analysis assistant for a food delivery
application.

Answer the user's question using ONLY the retrieved customer reviews
provided below.

IMPORTANT:
The retrieved reviews are only a subset of the complete dataset.
Do not make claims about all customers or all reviews.

Rules:

1. Answer the question directly. Do not explain your reasoning process.

2. Do not repeat the question.

3. Do not write phrases such as:
   - "We are looking for..."
   - "We need to..."
   - "We have to be careful..."
   - "Let's analyze..."
   - "Therefore, the answer is..."
   
4. Do not mention your instructions, prompt, RAG system, embeddings,
   similarity scores, or retrieval process.

5. Use ONLY information explicitly present in the retrieved reviews.

6. Never invent facts, statistics, complaints, or opinions.

7. If the question asks for the "worst", "best", "highest", "lowest",
   "top", or similar ranking:
   - Use the rating and review content provided.
   - For "worst", prioritize lower ratings and clearly negative content.
   - For "best", prioritize higher ratings and positive content.
   - If several reviews have similar ratings, use the review content
     to distinguish them.

8. If the user asks for multiple reviews, present them as a numbered list.

9. Include the review ID, city, rating, and relevant review text when
   useful.

10. Do not repeat identical review text unnecessarily. If duplicate
    reviews appear, mention that the same complaint appears in multiple
    reviews rather than repeating the entire text.

11. After listing the relevant reviews, provide a short summary of the
    main issues or positive themes if useful.

12. Keep the answer concise. Normally use no more than 2 short paragraphs
    plus a numbered list when a list is appropriate.

13. If the retrieved reviews do not contain enough information to answer
    the question, say:
    "The provided reviews do not contain enough information to answer this."

14. Remember that you only see the retrieved reviews, not the entire
    dataset. Never say "all customers", "all reviews", "every review",
    or similar unless that information is explicitly provided.

Return a clear, direct answer.
"""

    user_prompt = f"""
Question:
{question}

Retrieved customer reviews:
{context}
"""

    print(
        "Sending retrieved reviews "
        "to Qwen3..."
    )

    try:

        response = client.chat(
            model=CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            # Qwen3 does not need reasoning
            # for this RAG response.
            think=False,

            options={
                "temperature": 0.1,
                "num_predict": 200
            }
        )

    except Exception as e:

        raise RuntimeError(
            f"Qwen3 request failed: {e}"
        ) from e

    answer = response[
        "message"
    ]["content"]

    if not answer or not answer.strip():

        raise RuntimeError(
            "Qwen3 returned an empty response."
        )

    return answer


# =========================================================
# Streamlit Configuration
# =========================================================

st.set_page_config(
    page_title="Zomato Review RAG",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# Streamlit Header
# =========================================================

st.title(
    "🍽️ Chat with your Zomato Reviews"
)

st.caption(
    f"Searching {NEW_REVIEWS} reviews | "
    f"Embeddings: {EMBEDDING_MODEL} | "
    f"LLM: {CHAT_MODEL} | "
    f"Batch size: {EMBED_BATCH_SIZE}"
)


# =========================================================
# Check Ollama
# =========================================================

try:

    print(
        "Checking Ollama..."
    )

    models = client.list()

    installed_models = [
        model.model
        for model in models.models
    ]

    print(
        f"Installed Ollama models: "
        f"{installed_models}"
    )

    # -----------------------------------------------------
    # Normalize model names
    #
    # all-minilm
    # all-minilm:latest
    #
    # should be treated as the same model.
    # -----------------------------------------------------

    normalized_models = {
        model.split(":")[0]
        for model in installed_models
    }

    embedding_base = (
        EMBEDDING_MODEL
        .split(":")[0]
    )

    chat_base = (
        CHAT_MODEL
        .split(":")[0]
    )

    # -----------------------------------------------------
    # Check embedding model
    # -----------------------------------------------------

    if embedding_base not in normalized_models:

        st.error(
            f"Embedding model "
            f"'{EMBEDDING_MODEL}' "
            "is not installed."
        )

        st.code(
            f"ollama pull "
            f"{EMBEDDING_MODEL}"
        )

        st.stop()

    # -----------------------------------------------------
    # Check chat model
    # -----------------------------------------------------

    if chat_base not in normalized_models:

        st.error(
            f"Chat model "
            f"'{CHAT_MODEL}' "
            "is not installed."
        )

        st.code(
            f"ollama pull "
            f"{CHAT_MODEL}"
        )

        st.stop()

    print(
        "Ollama models are ready."
    )

except Exception as e:

    st.error(
        f"Could not connect to Ollama: {e}"
    )

    st.stop()


# =========================================================
# Load Review Data
# =========================================================

try:

    with st.spinner(
        "Loading reviews and embeddings..."
    ):

        review_df = load_reviews()

except Exception as e:

    st.error(
        f"Failed to load reviews: {e}"
    )

    st.stop()


# =========================================================
# Ready Status
# =========================================================

st.success(
    f"Ready! {len(review_df)} "
    f"reviews loaded."
)


# =========================================================
# Question Input
# =========================================================

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common "
        "complaints about delivery?"
    )
)


# =========================================================
# RAG Pipeline
# =========================================================

if question:

    try:

        # -------------------------------------------------
        # Retrieve
        # -------------------------------------------------

        with st.spinner(
            "Searching relevant reviews..."
        ):

            top_reviews = (
                find_similar_reviews(
                    question,
                    review_df
                )
            )

        # -------------------------------------------------
        # Generate
        # -------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = ask_llm(
                question,
                top_reviews
            )

        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        st.markdown(
            "### Answer"
        )

        st.write(
            answer
        )

        # -------------------------------------------------
        # Retrieved Reviews
        # -------------------------------------------------

        st.markdown(
            "### Retrieved Reviews"
        )

        display_df = top_reviews[
            [
                "review_id",
                "city",
                "rating",
                "comment",
                "score"
            ]
        ].copy()

        display_df["score"] = (
            display_df["score"]
            .round(4)
        )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"RAG pipeline error: {e}"
        )