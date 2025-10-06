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
    st.error("エラー: Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop() 

KNOWLEDGE_BASE_PATH = "knowledge_base.txt" # 👈 ファイルパス (単一ファイルに統一)
PERSIST_DIR = "chroma_db_cache"            # ChromaDBのキャッシュフォルダパス

st.set_page_config(page_title="要件事実支援アプリ", layout="wide")

# ====================================================
# 👈 アクセス制限ロジック (最終スマート版)
# ====================================================
ALLOWED_USERS = ["あなたの正確なメールアドレス@gmail.com"] 
user_email = None
try:
    if st.user and st.user.email: 
        user_email = st.user.email
    elif st.experimental_user and st.experimental_user.email:
        user_email = st.experimental_user.email
except Exception:
    pass

IS_STREAMLIT_CLOUD = "STREAMLIT_SERVER_USER" in os.environ 

if IS_STREAMLIT_CLOUD or ('localhost:' not in st.url and '127.0.0.1:' not in st.url):
    if user_email not in ALLOWED_USERS:
        st.error("🚨 アクセスが許可されていません。")
        st.info("このアプリケーションは、特定のユーザーのみが利用可能です。Googleアカウントでログインしてください。")
        st.stop()
else:
    st.warning("ローカル開発モードです。")
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
            # st.success のメッセージは削除 (画面に表示しない)
            # st.session_state['kb_initialized'] = True は削除
            return db
        except Exception as e:
            st.warning(f"キャッシュロード中にエラーが発生しました。再構築を試みます: {e}")
    
    # 既存DBがない場合、またはロード失敗した場合、新規作成ロジックへ
    try:
        loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
        all_documents = loader.load()
    except FileNotFoundError:
        return None 

    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(all_documents)
        
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            request_options={"timeout": 180}
        )

        db = Chroma.from_documents(
            texts, 
            embeddings_model, 
            persist_directory=PERSIST_DIR
        )
        db.persist() # 永続化を実行
        # st.success のメッセージは削除 (画面に表示しない)
        # st.session_state['kb_initialized'] = True は削除
        return db
    except Exception as e:
        st.error(f"データベース構築中にエラーが発生しました: {e}")
        return None

# RAGコアロジック (変更なし)
def get_required_elements_from_rag(db, description): 
    # ... (この関数は変更なし)
    docs = db.similarity_search(description, k=3) 
    context = "\n".join([d.page_content for d in docs])

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
    # データベースが正常にロードまたは構築された場合、画面を表示
    st.info("※事案の概要（いつ、誰が、何を、どうしたか）を詳細に入力してください。")

    contract_description = st.text_area(
        "【事案の概要】を記述してください。",
        height=300,
        placeholder="例：\n令和6年5月1日、売主Aは買主Bに対し、マンションの一室を引き渡した。\n同年5月10日、Bは、契約書に「全室無垢材フローリング」とあるにも関わらず、リビングの床材が合板であることを発見したため、契約不適合による損害賠償を請求したい。"
    )

    if st.button("📝 要件事実を自動作成する", type="primary"):
        if not contract_description:
            st.warning("事案の概要を入力してください。")
        else:
            with st.spinner("AIが要件事実論と知識を参照して分析中です..."):
                try:
                    result = get_required_elements_from_rag(db_instance, contract_description)
                    
                    st.subheader("✅ 請求権と要件事実の構成")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"処理中にエラーが発生しました。詳細: {e}")
else:
    # 失敗時のみ、詳細なエラーメッセージを表示
    st.error(f"データベースの初期化に失敗しました。ファイル '{KNOWLEDGE_BASE_PATH}' の存在と中身を確認してください。")
