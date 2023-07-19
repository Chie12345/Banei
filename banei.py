import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

st.title("ばんえい複勝馬の予想アプリ")
st.header("複勝とは・・・選んだ馬が3着以内に入れば勝ち！")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

df = pd.read_csv(uploaded_file)
st.dataframe(df)

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

# 馬名に数値を割り当てる
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df["馬名"])
le.transform(df["馬名"])
df["馬名"] = le.transform(df["馬名"])

# 天候に数値を割り当てる
le.fit(df["天候"])
df["天候"] = le.transform(df["天候"])

# victory、馬名、斤量、天候以外は削除
df = df.drop(["着 順", "枠", "馬 番", "性齢", "騎手", "タイム", "人 気", "単勝 オッズ", "厩舎", "馬体重 (増減)", "レースID", "Unnamed: 0", "着差", "後3F", "0", "1", "2", "3"], axis=1)

# 目標値
df.loc[df["victory"] == 0, "victory"] = "複勝頑張っちゃうかも！"
df.loc[df["victory"] == 1, "victory"] = "残念また今度頑張ろう"

# 予測モデルの構築
t = df["victory"].values
x = df.drop(["victory"], axis=1).values

# ロジスティック回帰
clf = LogisticRegression()
clf.fit(x, t)

# 入力画面
with st.form("my_form", clear_on_submit=False):
    st.write("出走馬情報を入力")
    name = st.text_input("馬名")
    weight = st.number_input("斤量", 0)
    wether = st.selectbox("天候", ["晴", "曇", "小", "雨", "雪"])

    submitted = st.form_submit_button("結果を予想！")
    if submitted:
        with st.spinner("計算中です・・・"):
            time.sleep(3)
    
st.write("結果発表‼‼")
# インプットデータ
value_df = pd.DataFrame({"出走馬情報":"　　", "馬名":name, "斤量":weight, "天候":wether}, index=[0])
value_df.set_index("出走馬情報", inplace=True)
st.write(value_df)

# 予測値のデータフレーム
# カテゴリカル変数のエンコード
le.fit(value_df["馬名"])
value_df["馬名"] = le.transform(value_df["馬名"])

le.fit(value_df["天候"])
value_df["天候"] = le.transform(value_df["天候"])

pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=["複勝頑張っちゃうかも！", "残念また今度頑張ろう"], index=["　　"])

st.write("勝利の行方は・・・")

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write("あなたの選んだ出走馬は", str(name[0]))

        




