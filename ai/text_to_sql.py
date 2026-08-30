import os
import re
import json

import pandas as pd
import streamlit as st
import snowflake.connector

from ollama import Client
from dotenv import load_dotenv


# =========================================================
# Environment
# =========================================================

load_dotenv()


# =========================================================
# Configuration
# =========================================================

MODEL = "qwen3:4b"

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

client = Client(
    host=OLLAMA_HOST
)


# =========================================================
# SQL Safety
# =========================================================

FORBIDDEN_KEYWORDS = {
    "drop",
    "delete",
    "truncate",
    "alter",
    "update",
    "insert",
    "create",
    "replace",
    "grant",
    "revoke",
    "merge",
    "call",
    "execute",
    "copy",
    "put",
    "remove",
}


# =========================================================
# Example Questions
# =========================================================

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method",
    "Top 10 restaurants by revenue",
    "Which city has the highest GMV?",
]


# =========================================================
# Snowflake Schema Available to the LLM
# =========================================================

SCHEMA = """
You are querying a Zomato analytics warehouse in Snowflake.

Use ONLY the following tables and columns.

TABLE: FCT_ORDERS

Columns:
- order_id
- order_date
- customer_id
- restaurant_id
- city
- cuisine
- payment_method
- order_status
- is_delivered
- sales_amount
- discount
- delivery_fee
- gst
- customer_rating
- delivery_time_min


TABLE: DIM_RESTAURANT

Columns:
- restaurant_id
- restaurant_name
- city
- cuisine
- rating
- cost_for_two


TABLE: DIM_CUSTOMER

Columns:
- customer_id
- customer_name
- age
- age_segment
- gender
- city


TABLE: MART_DAILY_CITY_REVENUNE

Columns:
- order_date
- city
- orders
- cancel_rate
- gmv
- aov


TABLE: MART_RESTAURANT_PERFORMANCE

Columns:
- restaurant_id
- restaurant_name
- city
- cuisine
- orders
- revenue
- avg_customer_rating
- cancel_rate


TABLE: MART_DELIVERY_SLA

Columns:
- city
- order_hour
- delivered_orders
- p50
- p90

IMPORTANT BUSINESS DEFINITIONS:

- P50 and P90 are percentile delivery-time metrics.
- Do not invent columns such as p50_delivery_min or late_rate.
- If the user asks for average delivery time, use the appropriate
  available delivery-time metric only when supported by the table.
- If the question requires a metric that does not exist, use another
  available table or explain that the required metric is unavailable.
- GMV means delivered revenue.
- Prefer MART_DAILY_CITY_REVENUNE for city-level GMV,
  orders, cancellation rate, and AOV questions.
- Prefer MART_RESTAURANT_PERFORMANCE for restaurant-level
  revenue, orders, ratings, and cancellation questions.
- Prefer MART_DELIVERY_SLA for delivery-time and late-rate
  analysis.
- Use FCT_ORDERS when the question requires fields not
  available in the MART tables.
- Use DIM_RESTAURANT when restaurant attributes are needed.
- Use DIM_CUSTOMER when customer attributes are needed.

TABLE NAMING:

Use bare table names only.

Correct:
SELECT city, gmv
FROM MART_DAILY_CITY_REVENUNE

Incorrect:
SELECT city, gmv
FROM ZOMATO.MARTS.MART_DAILY_CITY_REVENUNE

Do NOT invent tables or columns.
"""


# =========================================================
# SQL Generation Prompt
# =========================================================

SYSTEM_PROMPT = f"""
You are a Snowflake SQL generation assistant for a Zomato
analytics application.

Your job is to convert the user's natural-language question
into ONE safe Snowflake SELECT query.

{SCHEMA}

STRICT RULES:

1. Generate exactly ONE SQL query.

2. The query must be read-only.

3. Only SELECT statements or SELECT queries beginning with
   WITH are allowed.

4. Never generate:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   TRUNCATE
   CREATE
   REPLACE
   MERGE
   GRANT
   REVOKE
   CALL
   EXECUTE
   COPY
   PUT
   REMOVE

5. Never modify database objects or data.

6. Use ONLY the tables and columns listed above.

7. Use bare table names without database or schema prefixes.

8. Prefer the MART tables when they directly answer the question.

9. If the user asks for "top", sort in descending order.

10. If the user asks for "worst", sort appropriately in
    ascending order for the relevant metric.

11. If the user asks for an average, use AVG().

12. If the user asks for a count of orders, use COUNT(order_id)
    unless a suitable pre-aggregated MART table provides the
    required metric.

13. If calculating GMV from FCT_ORDERS, remember that GMV means
    delivered revenue. Filter to delivered orders when necessary.

14. Use NULL-safe calculations where appropriate.

15. Avoid division by zero using NULLIF().

16. Add LIMIT 100 or less for detailed result lists.

17. A LIMIT is not required for a query returning a single
    aggregate result.

18. Do not use SELECT * unless absolutely necessary.

19. Do not include SQL comments.

20. Do not include markdown code fences.

21. Return ONLY valid JSON.

22. Return JSON in exactly this structure:

{{
    "sql": "SELECT ..."
}}

23. Do not add any explanation outside the JSON.

Examples:

User:
Top 10 cities by GMV

JSON:
{{
    "sql": "SELECT city, SUM(gmv) AS total_gmv FROM MART_DAILY_CITY_REVENUNE GROUP BY city ORDER BY total_gmv DESC LIMIT 10"
}}

User:
Which cuisine has the most orders?

JSON:
{{
    "sql": "SELECT cuisine, SUM(orders) AS total_orders FROM MART_RESTAURANT_PERFORMANCE GROUP BY cuisine ORDER BY total_orders DESC LIMIT 1"
}}

User:
Average delivery time by city, worst first

JSON:
{{
    "sql": "SELECT city, AVG(p50_delivery_min) AS avg_delivery_time_min FROM MART_DELIVERY_SLA GROUP BY city ORDER BY avg_delivery_time_min DESC LIMIT 100"
}}
"""


# =========================================================
# Streamlit Page
# =========================================================

st.set_page_config(
    page_title="Zomato Text-to-SQL",
    page_icon="🗄️",
    layout="wide"
)


# =========================================================
# Header
# =========================================================

st.title("🗄️ Chat with your Zomato Data")

st.caption(
    f"Ask in English → {MODEL} generates SQL → "
    "Snowflake executes it"
)


# =========================================================
# Ollama Model Check
# =========================================================

def check_ollama():

    try:

        models = client.list()

        installed_models = [
            model.model
            for model in models.models
        ]

        normalized_models = {
            model.split(":")[0]
            for model in installed_models
        }

        model_base = MODEL.split(":")[0]

        if model_base not in normalized_models:

            st.error(
                f"Ollama model '{MODEL}' is not installed."
            )

            st.code(
                f"ollama pull {MODEL}"
            )

            st.stop()

        return True

    except Exception as e:

        st.error(
            f"Could not connect to Ollama: {e}"
        )

        st.code(
            "ollama serve"
        )

        st.stop()


check_ollama()


# =========================================================
# Snowflake Connection
# =========================================================

@st.cache_resource
def get_connection():

    return snowflake.connector.connect(

        account=os.getenv(
            "SNOWFLAKE_ACCOUNT"
        ),

        user=os.getenv(
            "SNOWFLAKE_USER"
        ),

        password=os.getenv(
            "SNOWFLAKE_PASSWORD"
        ),

        warehouse=os.getenv(
            "SNOWFLAKE_WAREHOUSE"
        ),

        database=os.getenv(
            "SNOWFLAKE_DATABASE"
        ),

        schema="MARTS",

        role="DBT_ROLE",

        client_session_keep_alive=False,
    )


# =========================================================
# Generate SQL
# =========================================================

def generate_sql(question):

    response = client.chat(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],

        # Qwen does not need extended reasoning
        # for this structured SQL generation task.
        think=False,

        # Ask Ollama for JSON.
        format="json",

        options={
            "temperature": 0,
            "num_predict": 400
        }
    )

    answer = response[
        "message"
    ]["content"]

    if not answer:

        raise ValueError(
            "Ollama returned an empty response."
        )

    try:

        parsed = json.loads(answer)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Ollama returned invalid JSON:\n{answer}"
        ) from e

    if "sql" not in parsed:

        raise ValueError(
            "Ollama JSON response does not contain "
            "'sql'."
        )

    sql = parsed["sql"]

    if not isinstance(sql, str):

        raise ValueError(
            "Generated SQL is not a string."
        )

    # Remove accidental database/schema prefixes.
    sql = re.sub(
        r"(?i)\bZOMATO\.MARTS\.",
        "",
        sql
    )

    sql = re.sub(
        r"(?i)\bZOMATO\.",
        "",
        sql
    )

    return sql.strip().rstrip(";")


# =========================================================
# SQL Safety Validation
# =========================================================

def is_safe(sql):

    if not sql:

        return False, "SQL is empty."

    sql_clean = sql.strip()

    lowered = sql_clean.lower()

    # -----------------------------------------------------
    # Must start with SELECT or WITH
    # -----------------------------------------------------

    if not (
        lowered.startswith("select ")
        or lowered.startswith("select\n")
        or lowered == "select"
        or lowered.startswith("with ")
        or lowered.startswith("with\n")
    ):

        return (
            False,
            "Only SELECT/WITH queries are allowed."
        )

    # -----------------------------------------------------
    # Only one SQL statement
    # -----------------------------------------------------

    if ";" in sql_clean:

        return (
            False,
            "Multiple SQL statements are not allowed."
        )

    # -----------------------------------------------------
    # Block SQL comments
    # -----------------------------------------------------

    if "--" in sql_clean or "/*" in sql_clean:

        return (
            False,
            "SQL comments are not allowed."
        )

    # -----------------------------------------------------
    # Tokenize SQL keywords
    #
    # This is safer than:
    # if "update" in sql
    #
    # because words such as:
    # updated_at
    # should not automatically be rejected.
    # -----------------------------------------------------

    tokens = set(
        re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
            lowered
        )
    )

    forbidden_found = (
        tokens
        &
        FORBIDDEN_KEYWORDS
    )

    if forbidden_found:

        return (
            False,
            "Forbidden SQL operation detected: "
            + ", ".join(
                sorted(forbidden_found)
            )
        )

    # -----------------------------------------------------
    # Block obvious system/object access
    # -----------------------------------------------------

    dangerous_patterns = [
        r"\binformation_schema\b",
        r"\bsnowflake\.account_usage\b",
        r"\bsystem\$",
    ]

    for pattern in dangerous_patterns:

        if re.search(
            pattern,
            lowered
        ):

            return (
                False,
                "Access to system metadata is not allowed."
            )

    return True, "Safe"


# =========================================================
# Run Query
# =========================================================

def run_query(sql):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        # Prevent a runaway query from running forever.
        cursor.execute(
            "ALTER SESSION SET "
            "STATEMENT_TIMEOUT_IN_SECONDS = 60"
        )

        result = (
            cursor
            .execute(sql)
            .fetch_pandas_all()
        )

        return result

    finally:

        cursor.close()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header(
        "Example Questions"
    )

    for q in EXAMPLE_QUESTIONS:

        if st.button(
            q,
            use_container_width=True
        ):

            st.session_state[
                "question"
            ] = q


# =========================================================
# Question Input
# =========================================================

default_question = st.session_state.get(
    "question",
    ""
)

question = st.text_input(

    "Enter your question here",

    value=default_question,

    placeholder=(
        "e.g. Top 10 restaurants by revenue "
        "in Bangalore"
    )
)


# =========================================================
# Process Question
# =========================================================

if question:

    try:

        # -------------------------------------------------
        # Generate SQL
        # -------------------------------------------------

        with st.spinner(
            "Qwen3 is generating SQL..."
        ):

            sql = generate_sql(
                question
            )

        # -------------------------------------------------
        # Validate SQL
        # -------------------------------------------------

        safe, reason = is_safe(
            sql
        )

        # -------------------------------------------------
        # Display SQL
        # -------------------------------------------------

        st.subheader(
            "Generated SQL"
        )

        st.code(
            sql,
            language="sql"
        )

        if not safe:

            st.error(
                "The generated SQL is not safe."
            )

            st.warning(
                reason
            )

            st.stop()

        st.success(
            "SQL passed the read-only safety check."
        )

        # -------------------------------------------------
        # Execute
        # -------------------------------------------------

        with st.spinner(
            "Running query in Snowflake..."
        ):

            df = run_query(
                sql
            )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.subheader(
            "Query Results"
        )

        st.success(
            f"{len(df)} rows returned."
        )

        if df.empty:

            st.info(
                "The query executed successfully "
                "but returned no rows."
            )

        else:

            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True
            )

        # -------------------------------------------------
        # Automatic chart
        # -------------------------------------------------

        if (
            len(df) > 0
            and len(df.columns) == 2
            and pd.api.types.is_numeric_dtype(
                df.iloc[:, 1]
            )
        ):

            st.subheader(
                "Visualization"
            )

            chart_df = df.copy()

            chart_df = chart_df.set_index(
                chart_df.columns[0]
            )

            st.bar_chart(
                chart_df[
                    chart_df.columns[0]
                ]
            )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )