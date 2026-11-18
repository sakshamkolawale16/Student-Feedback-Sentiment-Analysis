import streamlit as st
import pandas as pd
import joblib
import sqlite3
import plotly.express as px
from preprocess import clean_text
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Feedback Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/sentiment_model.pkl")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("feedback.db", check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment TEXT,
    sentiment TEXT
)
""")
conn.commit()


def get_data():
    return pd.read_sql_query(
        "SELECT * FROM feedback ORDER BY id DESC",
        conn
    )


# ---------------- SESSION STATE ----------------
if "portal" not in st.session_state:
    st.session_state.portal = "Home"

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False


# ---------------- FUTURISTIC CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #18265c 0%, transparent 35%),
        radial-gradient(circle at 90% 80%, #35145e 0%, transparent 35%),
        #080d1c;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a35, #111027);
    border-right: 1px solid #26376b;
}

h1, h2, h3 {
    color: #ffffff !important;
}

p, label, span {
    color: #d5def7;
}

.hero {
    padding: 45px 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #142a58, #30205d, #132d4d);
    border: 1px solid #5267bc;
    box-shadow: 0 0 35px rgba(69, 110, 255, 0.18);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    color: #d6e1ff;
}

.glass-card {
    background: rgba(22, 33, 65, 0.78);
    border: 1px solid rgba(111, 143, 255, 0.35);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 15px;
    transition: 0.3s;
}

.glass-card:hover {
    border-color: #65d9ff;
    box-shadow: 0 0 25px rgba(65, 185, 255, 0.25);
    transform: translateY(-3px);
}

.badge {
    display: inline-block;
    background: rgba(0, 225, 255, 0.12);
    color: #62e5ff;
    border: 1px solid #287eaa;
    border-radius: 30px;
    padding: 7px 14px;
    font-size: 13px;
    margin-bottom: 12px;
}

div.stButton > button {
    background: linear-gradient(90deg, #176bff, #8b45ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
    transition: 0.3s;
}

div.stButton > button:hover {
    box-shadow: 0 0 20px #526dff;
    border: 1px solid #8beaff;
    color: white;
}

div[data-testid="stMetric"] {
    background: rgba(27, 42, 78, 0.85);
    border: 1px solid #344c87;
    padding: 18px;
    border-radius: 15px;
}

div[data-testid="stMetricValue"] {
    color: #66e5ff;
}

textarea, input {
    background-color: #111a34 !important;
    color: white !important;
}

hr {
    border-color: #293c70;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HELPER ----------------
def show_hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <div class="badge">✦ AI POWERED SENTIMENT INTELLIGENCE</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HOME ----------------
if st.session_state.portal == "Home":

    show_hero(
        "Student Feedback Intelligence",
        "Understand student experiences with Artificial Intelligence and Natural Language Processing."
    )

    st.markdown("### Explore Your Portal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h2>🎓 Student Portal</h2>
            <p>Share your feedback anonymously and let AI understand your experience.</p>
            <div class="badge">ANONYMOUS FEEDBACK</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Student Portal", use_container_width=True):
            st.session_state.portal = "Student Portal"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h2>🔐 Admin Portal</h2>
            <p>Access feedback analytics, manage responses, and monitor sentiment trends.</p>
            <div class="badge">ADMIN ACCESS</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Admin Portal", use_container_width=True):
            st.session_state.portal = "Admin Login"
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Platform Capabilities")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="glass-card">
            <h3>🤖 AI Analysis</h3>
            <p>Machine learning based sentiment prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Smart Analytics</h3>
            <p>Interactive charts and feedback statistics.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="glass-card">
            <h3>🔒 Privacy</h3>
            <p>The feedback form does not ask for student names or IDs.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- STUDENT PORTAL ----------------
elif st.session_state.portal == "Student Portal":

    if st.button("← Back to Home"):
        st.session_state.portal = "Home"
        st.rerun()

    show_hero(
        "Your Voice Matters",
        "Share your experience. Our AI system will analyze the sentiment of your feedback."
    )

    st.markdown('<div class="badge">🔒 NO NAME OR STUDENT ID REQUIRED</div>',
                unsafe_allow_html=True)

    st.markdown("### ✍️ Submit Your Feedback")

    feedback = st.text_area(
        "Write your feedback below:",
        placeholder="Example: The teaching quality is excellent and the faculty is helpful...",
        height=170
    )

    if st.button("🚀 Analyze My Feedback", use_container_width=True):

        if feedback.strip():

            cleaned = clean_text(feedback)
            prediction = model.predict([cleaned])[0]

            conn.execute(
                "INSERT INTO feedback (comment, sentiment) VALUES (?, ?)",
                (feedback, prediction)
            )
            conn.commit()

            if prediction == "positive":
                st.success("😊 Positive Feedback Detected!")

            elif prediction == "negative":
                st.error("😟 Negative Feedback Detected!")

            else:
                st.info("😐 Neutral Feedback Detected!")

            st.markdown(f"""
            <div class="glass-card">
                <h2>🤖 AI Analysis Result</h2>
                <h1>{prediction.upper()}</h1>
                <p>Your feedback has been successfully recorded.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("Please enter your feedback first.")


# ---------------- ADMIN LOGIN ----------------
elif st.session_state.portal == "Admin Login":

    if st.button("← Back to Home"):
        st.session_state.portal = "Home"
        st.rerun()

    show_hero(
        "Administrator Access",
        "Secure login for feedback monitoring and analytics."
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown("""
        <div class="glass-card">
            <h2>🔐 Admin Login</h2>
            <p>Enter your administrator credentials.</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login to Dashboard", use_container_width=True):

            if username == "admin" and password == "admin123":
                st.session_state.admin_logged_in = True
                st.session_state.portal = "Admin Dashboard"
                st.success("Login successful!")
                st.rerun()

            else:
                st.error("Invalid username or password.")


# ---------------- ADMIN DASHBOARD ----------------
elif st.session_state.portal == "Admin Dashboard":

    if not st.session_state.admin_logged_in:
        st.session_state.portal = "Admin Login"
        st.rerun()

    st.sidebar.title("🤖 AI Admin Panel")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Analyze Feedback",
            "Feedback History",
            "Manage Feedback",
            "Model Performance"
        ]
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.admin_logged_in = False
        st.session_state.portal = "Home"
        st.rerun()

    data = get_data()

    # ---------------- DASHBOARD ----------------
    if page == "Dashboard":

        show_hero(
            "Admin Analytics Dashboard",
            "Monitor student feedback and understand overall student sentiment."
        )

        total = len(data)

        positive = len(data[data["sentiment"] == "positive"])
        negative = len(data[data["sentiment"] == "negative"])
        neutral = len(data[data["sentiment"] == "neutral"])

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Feedback", total)
        c2.metric("Positive", positive)
        c3.metric("Negative", negative)
        c4.metric("Neutral", neutral)
        st.markdown("### 📈 Sentiment Percentages")

        if total > 0:
            positive_percent = (positive / total) * 100
            negative_percent = (negative / total) * 100
            neutral_percent = (neutral / total) * 100
        else:
            positive_percent = 0
            negative_percent = 0
            neutral_percent = 0

        p1, p2, p3 = st.columns(3)

        p1.metric("🟢 Positive", f"{positive_percent:.1f}%")
        p2.metric("🔴 Negative", f"{negative_percent:.1f}%")
        p3.metric("🔵 Neutral", f"{neutral_percent:.1f}%")

        st.markdown("### 📊 Sentiment Analytics")

        if total > 0:

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    data,
                    names="sentiment",
                    title="Sentiment Distribution",
                    hole=0.60,
                    color="sentiment",
                    color_discrete_map={
                        "positive": "#00D9A6",
                        "negative": "#FF5277",
                        "neutral": "#6C9EFF"
                    }
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111A34",
                    plot_bgcolor="#111A34",
                    font=dict(color="white")
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                counts = data["sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]

                fig2 = px.bar(
                    counts,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    title="Feedback Overview",
                    color_discrete_map={
                        "positive": "#00D9A6",
                        "negative": "#FF5277",
                        "neutral": "#6C9EFF"
                    }
                )

                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111A34",
                    plot_bgcolor="#111A34",
                    font=dict(color="white"),
                    showlegend=False
                )

                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("No feedback has been submitted yet.")

    # ---------------- ANALYZE FEEDBACK ----------------

   
    elif page == "Analyze Feedback":

        show_hero(
            "Feedback Analysis",
            "Test the machine learning model with a feedback comment."
        )

        comment = st.text_area(
            "Enter feedback:",
            height=150
        )

        if st.button("Analyze Sentiment"):

            if comment.strip():

                cleaned = clean_text(comment)
                result = model.predict([cleaned])[0]

                st.success(f"Predicted Sentiment: {result.upper()}")

            else:
                st.warning("Please enter feedback.")

    # ---------------- HISTORY ----------------
        elif page == "Feedback History":

            show_hero(
            "Feedback History",
            "View previously submitted student feedback."
        )

        if not data.empty:

            st.dataframe(data, use_container_width=True)

            csv = data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Feedback CSV",
                csv,
                "feedback_history.csv",
                "text/csv"
            )

        else:
            st.info("No feedback available.")

       # ---------------- MANAGE FEEDBACK ----------------
    elif page == "Manage Feedback":

        show_hero(
            "Feedback Management",
            "Search, filter, and manage submitted feedback."
        )

        if not data.empty:

            sentiment_filter = st.selectbox(
                "Filter by sentiment",
                ["All", "positive", "negative", "neutral"]
            )

            filtered = data.copy()

            if sentiment_filter != "All":
                filtered = filtered[
                    filtered["sentiment"] == sentiment_filter
                ]

            search = st.text_input("🔍 Search feedback")

            if search:
                filtered = filtered[
                    filtered["comment"].str.contains(
                        search, case=False, na=False
                    )
                ]

            st.dataframe(filtered, use_container_width=True)

            ids = filtered["id"].tolist()

            selected = st.multiselect(
                "Select feedback IDs to delete",
                ids
            )

            if st.button("🗑️ Delete Selected Feedback"):

                if selected:
                    placeholders = ",".join(["?"] * len(selected))

                    conn.execute(
                        f"DELETE FROM feedback WHERE id IN ({placeholders})",
                        selected
                    )

                    conn.commit()
                    st.success("Selected feedback deleted.")
                    st.rerun()

                else:
                    st.warning("Please select at least one record.")

        else:
            st.info("No feedback records available.")

    # ---------------- MODEL PERFORMANCE ----------------
    elif page == "Model Performance":

        show_hero(
            "AI Model Performance",
            "Evaluate the accuracy and reliability of your sentiment analysis model."
        )

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
            confusion_matrix
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        if st.button("Calculate Model Performance"):

            dataset = pd.read_csv("feedback.csv")

            X = dataset["feedback"].apply(clean_text)
            y = dataset["sentiment"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            evaluation_model = Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("classifier", LogisticRegression(max_iter=1000))
            ])

            evaluation_model.fit(X_train, y_train)
            predictions = evaluation_model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, average="macro", zero_division=0
            )
            recall = recall_score(
                y_test, predictions, average="macro", zero_division=0
            )
            f1 = f1_score(
                y_test, predictions, average="macro", zero_division=0
            )

            st.markdown("### 📊 Model Evaluation Results")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            m2.metric("Precision", f"{precision * 100:.2f}%")
            m3.metric("Recall", f"{recall * 100:.2f}%")
            m4.metric("F1 Score", f"{f1 * 100:.2f}%")

            st.markdown("### 📋 Classification Report")

            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True
            )

            st.markdown("### 🎯 Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=["positive", "negative", "neutral"]
            )

            fig = px.imshow(
                cm,
                x=["Positive", "Negative", "Neutral"],
                y=["Positive", "Negative", "Neutral"],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111A34",
                plot_bgcolor="#111A34"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Click the button above to calculate model performance.")


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>⚡ AI Student Feedback Analyzer | Powered by Machine Learning & NLP</center>",
    unsafe_allow_html=True
)