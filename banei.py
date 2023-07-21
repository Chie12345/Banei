import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

st.title("ばんえい複勝馬の予想アプリ")
st.header("複勝とは・・・選んだ馬が3着以内に入れば勝ち！")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

df = pd.read_csv(uploaded_file)

# 着 順が数字以外のものを消す
df = df.dropna(subset=["着 順"])
df = df[~df["着 順"].str.contains("中止|除外|取消|失格")]

# 出走回数が2以下の馬を削除
df = df[~df["馬名"].str.contains("カツカゲトラ|アキノダイマオー|ヤチヨ|イワフネタロウ|フクイサム")]
df = df[~df["馬名"].str.contains("ホクトセンショウ|アアモンドセブン|マドーシヒカル|カクセンクシロ|□抽 カネタケパワー")]
df = df[~df["馬名"].str.contains("ギンノカミカゼ|ホクトウイン|ウズメレディー|キタノミハル|ワタシモセトヨ")]
df = df[~df["馬名"].str.contains("サクラメイゲツ|ハクノスミカ|ピカイチ|ヤマノボクサー|コマノゴールド")]
df = df[~df["馬名"].str.contains("オイテカナイデ|ラックボックス|メムロビジン|ウチウラタカラ|セミニヨン")]
df = df[~df["馬名"].str.contains("アース|シマノルビー|ビービークイン|ディアレスト|ジェイメイビー")]
df = df[~df["馬名"].str.contains("ジェイミント|サカノテッペン")]

#着順と斤量をint64型にする
df["着 順"] = df["着 順"].astype(int)
df["斤量"] = df["斤量"].astype(int)

# 1-3着は0、それ以外は1
df["victory"] = [0 if i <=3 else 1 for i in df["着 順"].tolist()]

# 天候を数値データにする
wether_mapping = {"晴": 1, "曇": 2, "小": 3, "雨": 4, "雪":5}
df['天候'] = df['天候'].map(wether_mapping)

# 馬名から辞書を作る
name_mapping = dict(zip(df["馬名"].unique().tolist(), range(1, len(df["馬名"].unique().tolist()) + 1)))
# 文字列から数値に変換する
df["馬名"] = df["馬名"].map(name_mapping)

# victory、馬名、斤量、天候以外は削除
df = df.drop(["着 順", "枠", "馬 番", "性齢", "騎手", "タイム", "人 気", "単勝 オッズ", "厩舎", "馬体重 (増減)", "レースID", "Unnamed: 0", "着差", "後3F", "0", "1", "2", "3"], axis=1)

# 目標値
df.loc[df["victory"] == 0, "victory"] = "複勝頑張っちゃうかも！"
df.loc[df["victory"] == 1, "victory"] = "残念また今度頑張ろう"

# 予測モデルの構築
t = df["victory"].values
x = df.drop(["victory"], axis=1).values

# ロジスティック回帰
clf = LogisticRegression(C=1.0)
# モデルの学習
clf.fit(x, t)

# 入力画面
with st.form("my_form", clear_on_submit=False):
    st.write("出走馬情報を入力")
    name = st.text_input("馬名")
    weight = st.number_input("斤量", 0)
    wether = st.selectbox("天候", ["晴", "曇", "小", "雨", "雪"])

    if not name:
        st.warning("出走馬の名前をおしえてください")

    submitted = st.form_submit_button("結果を予想！")
    if submitted:
        with st.spinner("計算中です・・・"):
            time.sleep(3)
    
st.write("結果発表‼‼")
# インプットデータ
df_run = pd.DataFrame({"出走馬情報":"　　", "馬名":name, "斤量":weight, "天候":wether}, index=[0])
df_run.set_index("出走馬情報", inplace=True)
st.write(df_run)

# 予測値のデータフレーム
# カテゴリカル変数のエンコード
df_run["馬名"] = df_run["馬名"].map(name_mapping)
df_run['天候'] = df_run['天候'].map(wether_mapping)

pred_probs = clf.predict_proba(df_run)
pred_df = pd.DataFrame(pred_probs, columns=["複勝頑張っちゃうかも！", "残念また今度頑張ろう"], index=["　　"])

st.write("勝利の行方は・・・")

# 予測結果の出力!!
name = pred_df.idxmax(axis=1).tolist()
st.write("あなたの選んだ出走馬は", str(name[0]))

        




