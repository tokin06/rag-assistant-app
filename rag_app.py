import streamlit as st
import os
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ====================================================
# 0. 設定と初期化 (APIキーの秘匿化)
# ====================================================

# ★★★ APIキーをst.secretsから安全に取得します ★★★
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    # 秘匿化されたキーがない場合はエラーで停止
    st.error("エラー: Secretsに 'GEMINI_API_KEY' が設定されていません。Streamlit CloudのSecretsを確認してください。")
    st.stop() 

KNOWLEDGE_BASE_PATH = "knowledge_base.txt" # 👈 ファイルパス (単一ファイルに統一)
PERSIST_DIR = "chroma_db_cache"            # ChromaDBのキャッシュフォルダパス

st.set_page_config(page_title="要件事実支援アプリ", layout="wide")

# ====================================================
# 👈 ここからアクセス制限ロジック (最終クリーンアップ版)
# ====================================================

# 許可するメールアドレスのリスト
# 🚨 Web公開前に必ずあなたの正確なメールアドレスに修正してください
ALLOWED_USERS = ["wada.ayumu06@gmail.com"] 

user_email = None

# 認証情報を取得する試み
try:
    # st.user や st.experimental_user からメールアドレスを取得
    if st.user and st.user.email: 
        user_email = st.user.email
    elif st.experimental_user and st.experimental_user.email:
        user_email = st.experimental_user.email
except Exception:
    pass

# 認証チェックの実行
# Streamlit Cloud上で実行中かを判断し、本番環境でのみ厳格にチェック
IS_STREAMLIT_CLOUD = "STREAMLIT_SERVER_USER" in os.environ 

if IS_STREAMLIT_CLOUD:
    if user_email not in ALLOWED_USERS:
        st.error("🚨 アクセスが許可されていません。")
        st.error("このアプリケーションは、特定のユーザーのみが利用可能です。")
        st.stop()
else:
    # 💡 ローカル環境での実行時は開発続行を優先
    st.warning("ローカル開発モードで実行中です。Web公開時はアクセス制限がかかります。")
    
# ====================================================
# 👈 アクセス制限ロジックの終わり
# ====================================================

# ====================================================
# 1. RAGの「本棚」構築機能（単一ファイル対応とキャッシュ永続化付き）
# ====================================================
@st.cache_resource
def initialize_knowledge_base():
    """知識データベース（本棚）を初期化し、ChromaDBを返す"""
    
    # 既存のデータベースが存在するかチェック (高速ロード)
    if os.path.exists(PERSIST_DIR):
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings_model)
            st.success("既存の知識データベースをロードしました！ (高速起動)")
            return db
        except Exception as e:
            st.warning(f"キャッシュロード中にエラーが発生しました。再構築を試みます: {e}")
    
    # 既存DBがない場合、またはロード失敗した場合、新規作成ロジックへ
    try:
        # TextLoaderで単一のファイルを読み込む 
        loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
        all_documents = loader.load()
    except FileNotFoundError:
        st.error(f"エラー: 知識データベースファイルが見つかりません: {KNOWLEDGE_BASE_PATH}")
        st.warning(f"ファイル '{KNOWLEDGE_BASE_PATH}' を作成し、中身を入れてください。") 
        return None
        
    try:
        # テキストの分割 (チャンキング)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(all_documents)
        
        # 埋め込みモデル (タイムアウトを180秒に延長)
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            request_options={"timeout": 180}
        )

        # ChromaDBの作成と永続化
        db = Chroma.from_documents(
            texts, 
            embeddings_model, 
            persist_directory=PERSIST_DIR
        )
        db.persist() # 永続化を実行
        st.success("知識データベースを新規作成し、キャッシュに保存しました！")
        return db
    except Exception as e:
        st.error(f"データベース構築中にエラーが発生しました: {e}")
        return None

# RAGコアロジック
def get_required_elements_from_rag(db, description): 
    """RAGを実行し、事案に対する要件事実の構成を返す"""
    
    # 記述内容に関連する情報をデータベースから検索（「本を探す」）
    docs = db.similarity_search(description, k=3) 
    context = "\n".join([d.page_content for d in docs])

    # AIに与える指示（プロンプト）を作成 (要件事実生成用プロンプト)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            あなたは要件事実論の専門家AIです。要件事実の出力においては法的正確性を最優先してください。
            提供された「事案」と、参照情報に基づいて、以下のタスクを実行してください。
            
1. 事案から**請求の趣旨**を特定する。
2. 事案から**最も適切な訴訟物（請求権）**を特定する。
3. その訴訟物に必要な**要件事実**を、**明確な箇条書き**で抽出・作成する。
4. 抗弁、再抗弁、再々抗弁・・・が成り立つ場合は、成り立つ抗弁を作成する。２つ以上の抗弁が成り立つ場合は、それぞれの抗弁に対する再抗弁以下であるとわかるように、再抗弁以下も作成する。
5. 参照した要件事実の根拠となる**法令や判例**があれば、最後に明記してください。
            """),
            ("user", "以下の事案について、必要な要件事実を自動作成してください。\n\n事案:\n{contract_description}\n\n参照情報:\n{context}"),
        ]
    )

    # LLM（AIの脳みそ）の呼び出しと設定
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"contract_description": description, "context": context})
    return response

# ====================================================
# 2. Streamlitのウェブアプリ画面構築
# ====================================================

st.title("⚖️ 要件事実 自動作成アシスタント (RAG-POC)")

# データベースの初期化
db_instance = initialize_knowledge_base()

if db_instance:
    st.success("知識データベースの準備が完了しました！要件事実の出力を開始できます。")
    st.info("※事案の概要（いつ、誰が、何を、どうしたか）を詳細に入力してください。")

    # ユーザーからの入力エリア
    contract_description = st.text_area(
        "【事案の概要】を記述してください。",
        height=300,
        placeholder="例：\n令和6年5月1日、売主Aは買主Bに対し、マンションの一室を引き渡した。\n同年5月10日、Bは、契約書に「全室無垢材フローリング」とあるにも関わらず、リビングの床材が合板であることを発見したため、契約不適合による損害賠償を請求したい。"
    )

    # 判定実行ボタン
    if st.button("📝 要件事実を自動作成する", type="primary"):
        if not contract_description:
            st.warning("事案の概要を入力してください。")
        else:
            # 処理実行
            with st.spinner("AIが要件事実論と知識を参照して分析中です..."):
                try:
                    result = get_required_elements_from_rag(db_instance, contract_description)
                    
                    # 結果の表示
                    st.subheader("✅ 請求権と要件事実の構成")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"処理中にエラーが発生しました。詳細: {e}")
else:
    st.warning(f"データベースの初期化に失敗したため、アプリを起動できません。ファイル '{KNOWLEDGE_BASE_PATH}' の存在と中身を確認してください。")
