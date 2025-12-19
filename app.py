import streamlit as st
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from collections import Counter
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



import pandas as pd

# Google Sheets 認証
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("stremlit-voting-dfc8b6ac90cc.json", scope)
client = gspread.authorize(creds)

# スプレッドシートを開く
sheet = client.open("PreferenceVotes").sheet1

# 選択肢
options = ["Option A", "Option B", "Option C"]

st.title("🗳️ Multi-user Voting with Google Sheets")
#投票者の名前とコメント
name = st.text_input("Your name or nickname")
comment = st.text_area("Why did you rank this way?")


ranked = st.multiselect("Rank the options (top = highest)", options, default=options)

if st.button("Submit"):
    if len(ranked) == len(options):
        sheet.append_row([name, comment] + ranked)
        st.success("Your vote has been saved to Google Sheets!")
    else:
        st.warning("Please rank all options before submitting.")


data = sheet.get_all_records()
df = pd.DataFrame(data)



# スコア集計
scores = {opt: 0 for opt in options}
for _, row in df.iterrows():
    for i, opt in enumerate([row["Rank1"], row["Rank2"], row["Rank3"]]):
        scores[opt] += len(options) - i - 1

score_df = pd.DataFrame(scores.items(), columns=["Option", "Score"]).sort_values("Score", ascending=False)
st.subheader("📊 Aggregated Results")
st.bar_chart(score_df.set_index("Option"))
st.subheader("🗣️ Voter Comments")
for _, row in df.iterrows():
    st.markdown(f"**{row['Name']}**: {row['Comment']}")

# --- コメントのキーワード抽出 ---
if not df.empty and "Comment" in df.columns:
    comments = df["Comment"].dropna().tolist()

    tokenizer = Tokenizer()
    words = []
    for comment in comments:
        tokens = tokenizer.tokenize(comment)
        for token in tokens:
            # 名詞だけを抽出
            if token.part_of_speech.startswith("名詞"):
                words.append(token.surface)
# 頻度集計
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)

    if common_words:
        keywords_df = pd.DataFrame(common_words, columns=["Keyword", "Count"])
        st.subheader("🔑 コメントキーワード頻度（上位10件）")
        st.bar_chart(keywords_df.set_index("Keyword"))
    else:
        st.info("まだコメントが少ないため、キーワード抽出はできません。")

# --- ワードクラウド生成 ---
if not df.empty and "Comment" in df.columns:
    comments = df["Comment"].dropna().tolist()
    text = " ".join(comments)

    # WordCloud生成（日本語フォントを指定）
    wordcloud = WordCloud(
        font_path="C:/Windows/Fonts/msgothic.ttc",  # 環境に合わせて変更
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    # Streamlitで表示
    st.subheader("☁️ CommentCloud")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# --- コメントクラスタリング ---
if not df.empty and "Comment" in df.columns:
    comments = df["Comment"].dropna().tolist()

    if len(comments) < 3:
        st.info("コメントが少ないためクラスタリングはスキップしました。")
        for c in comments:
            st.markdown(f"- {c}")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from janome.tokenizer import Tokenizer
    
    tokenizer = Tokenizer()
    def tokenize(text):
        return [token.surface for token in tokenizer.tokenize(text) if token.part_of_speech.startswith("名詞")]

    
    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    X = vectorizer.fit_transform(comments)

    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    
    cluster_df = pd.DataFrame({"Comment": comments, "Cluster": labels})

    st.subheader("🧩 コメントクラスタリング結果")
    for cluster_id in sorted(cluster_df["Cluster"].unique()):
        st.markdown(f"### クラスタ {cluster_id}")
        for comment in cluster_df[cluster_df["Cluster"] == cluster_id]["Comment"]:
            st.markdown(f"- {comment}")





if st.button("🔄 Reset all votes"):
    sheet.resize(rows=1)
    st.warning("All votes have been cleared.")