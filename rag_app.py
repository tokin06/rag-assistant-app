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
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("エラー: Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop() 

KNOWLEDGE_BASE_PATH = "knowledge_base.txt" 
PERSIST_DIR = "chroma_db_cache"            

st.set_page_config(page_title="要件事実支援アプリ", layout="wide")

# ====================================================
# 1. RAGの「本棚」構築機能（単一ファイル対応とキャッシュ永続化付き）
# ====================================================
@st.cache_resource
def initialize_knowledge_base():
    """知識データベース（本棚）を初期化し、ChromaDBを返す"""
    
    if os.path.exists(PERSIST_DIR):
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings_model)
            return db
        except Exception as e:
            st.warning(f"キャッシュロード中にエラーが発生しました。再構築を試みます: {e}")
    
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
        db.persist()
        return db
    except Exception as e:
        st.error(f"データベース構築中にエラーが発生しました: {e}")
        return None

# ====================================================
# 1.5. ユーティリティ機能
# ====================================================

@st.cache_data(ttl=600)
def check_query_relevance(query):
    """入力されたクエリが法律関連の事案であるかをAIに判定させる (ステップ1: ガードレール)"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) 
    prompt = f"""
    以下のユーザー入力は、**法的な紛争や主張**に関連する「事案の記述」ですか？
    全く関係のない雑談、レシピ、プログラミングコード、または意味のないランダムな文字列である場合は「No」とだけ回答してください。
    それ以外の場合は「Yes」とだけ回答してください。
    ユーザー入力："{query}"
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip().upper()
    except Exception as e:
        st.warning(f"クエリ関連性チェック中にエラーが発生しました。スキップします。詳細: {e}")
        return "YES" 

def check_for_missing_facts(db, query): # 👈 db を引数に追加
    """要件事実の作成に足りない事実があるかチェックし、足りない事実を返す (ステップ2: 事実補完)"""
    
    # --- RAG検索を実行し、専門知識に基づいてチェックする ---
    docs = db.similarity_search(query, k=3) 
    context = "\n".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    prompt = f"""
    あなたは要件事実論の専門家です。以下の「事案」と「参照情報」を読み、この事案に基づいて要件事実を作成する場合、**決定的に不足している事実**または**曖昧な事実**を特定し、ユーザーに補完を促す文章を作成してください。
    不足している事実がない場合は、**必ず**「OK」とだけ回答してください。
    
    【事案】
    {query}
    
    【参照情報】
    {context}
    
    【回答の例】
    ・不足している事実：原告が損害を受けた具体的な金額を追記してください。
    ・OK
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        st.error(f"事実補完チェック中にエラーが発生しました。詳細: {e}")
        return "OK" # チェック失敗時は安全のため実行を許可

# RAGコアロジック (最終生成 - ステップ3)
def get_required_elements_from_rag(db, description): 
    """RAGを実行し、事案に対する要件事実の構成を返す"""
    
    docs = db.similarity_search(description, k=3) 
    context = "\n".join([d.page_content for d in docs])

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            あなたは要件事実論の専門家AIです。法的正確性を最優先してください。提供された事案と参照情報に基づいて、以下のタスクを実行してください。
            【タスク】1. 請求の趣旨を特定する。2. 最も適切な訴訟物（請求権）を特定する。3. その訴訟物に必要な要件事実を明確な箇条書きで抽出・作成する。4. 抗弁、再抗弁があれば作成する。5. 参照した法令や判例を最後に明記する。
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

# --- アプリの状態管理 ---
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1  # 1: 事案入力, 2: 事実補完待ち

st.title("⚖️ 要件事実 自動作成アシスタント (RAG-POC)")

# --- キャッシュクリアボタンのロジック (サイドバー) ---
def clear_knowledge_cache():
    st.cache_resource.clear()
    st.rerun()

with st.sidebar:
    st.markdown("### 🛠️ データベース管理")
    if st.button("知識ベースを再構築/リロード", help="knowledge_base.txt を変更した後に押してください。", key="reload_db_sidebar"):
        clear_knowledge_cache()
    st.markdown("---")
    if 'fact_feedback' in st.session_state:
        st.markdown("#### 🔄 現在のステータス")
        st.markdown(f"**事案:** `{st.session_state['original_query'][:30]}...`")
        st.warning("事実補完ステップで待機中です。")


# データベースの初期化
db_instance = initialize_knowledge_base()
    
if db_instance:
    # 乱用防止ロジックの初期化
    if 'running' not in st.session_state:
        st.session_state['running'] = False 
    is_running = st.session_state['running']

    # ----------------------------------------------------
    # フェーズ表示
    # ----------------------------------------------------
    if st.session_state['current_step'] == 1:
        st.info("ステップ 1/3: 事案の概要を入力してください。")
    elif st.session_state['current_step'] == 2:
        st.info("ステップ 2/3: 不足事実を追記または修正してください。")
    elif st.session_state['current_step'] == 3:
        st.success("ステップ 3/3: 要件事実の最終構成を生成します。")


    # ----------------------------------------------------
    # メイン入力エリア (ステップ 1 & 2)
    # ----------------------------------------------------
    
    # ステップ2の場合、以前のクエリとフィードバックをテキストエリアに表示
    initial_query = st.session_state.get('original_query', "")
    if st.session_state['current_step'] == 2:
        st.subheader("💡 AIからのフィードバック")
        st.warning(f"{st.session_state['fact_feedback']}")
        initial_query = st.text_area(
            "【不足事実を追記・修正してください】",
            value=st.session_state['fact_feedback'] + "\n\n--- 修正点 ---",
            height=350,
            key="corrected_query"
        )
    else:
        initial_query = st.text_area(
            "【事案の概要を入力してください】",
            height=300,
            placeholder="例：\n令和6年5月1日、売主Aは買主Bに対し、マンションの一室を引き渡した。\n同年5月10日、Bは、契約書に「全室無垢材フローリング」とあるにも関わらず、リビングの床材が合板であることを発見したため、契約不適合による損害賠償を請求したい。",
            key="initial_query"
        )
    
    # ----------------------------------------------------
    # ボタンとロジックの実行
    # ----------------------------------------------------
    button_label = "次のステップへ (事実確認)" if st.session_state['current_step'] != 3 else "📝 要件事実を最終生成する"

    if st.button(button_label, type="primary", disabled=is_running): 
        if not initial_query:
            st.warning("事案の概要を入力してください。")
            st.session_state['running'] = False # ランニングフラグを確実に解除
            st.rerun()

        # Phase 1: ガードレールチェック
        if st.session_state['current_step'] == 1:
            st.session_state['running'] = True
            with st.spinner("ステップ1/3: 法律関連の事案かチェック中です..."):
                relevance = check_query_relevance(initial_query)

            if relevance == "NO":
                st.error("入力内容は法律関連の事案として認識されませんでした。要件事実に関する具体的な事案を記述してください。")
                st.session_state['running'] = False
                st.rerun()
            else:
                # 法律関連と判断 -> Phase 2: 事実補完チェックへ
                st.session_state['original_query'] = initial_query
                st.session_state['running'] = True
                with st.spinner("ステップ2/3: 不足事実のチェック中です..."):
                    # 👈 check_for_missing_facts に db を渡すように修正
                    missing_facts = check_for_missing_facts(db_instance, initial_query) 
                
                st.session_state['running'] = False
                
                if "OK" in missing_facts.upper():
                    # 不足事実なし -> Phase 3へスキップ
                    st.session_state['current_step'] = 3
                else:
                    # 不足事実あり -> Phase 2でフィードバック待ち
                    st.session_state['current_step'] = 2
                    st.session_state['fact_feedback'] = missing_facts
            st.rerun() # 画面を更新して次のステップへ

        # Phase 2: 事実補完後の最終実行 (ボタンが押されたら Phase 3へ)
        elif st.session_state['current_step'] == 2:
            st.session_state['current_step'] = 3
            st.session_state['original_query'] = initial_query # 修正されたクエリを保存
            del st.session_state['fact_feedback']
            st.rerun()

        # Phase 3: 最終生成
        elif st.session_state['current_step'] == 3:
            st.session_state['running'] = True
            with st.spinner("ステップ3/3: 要件事実の最終構成を生成中です..."):
                try:
                    result = get_required_elements_from_rag(db_instance, initial_query)
                    
                    st.subheader("✅ 請求権と要件事実の構成")
                    st.markdown(result)
                    
                except Exception as e:
                    st.error(f"処理中にエラーが発生しました。詳細: {e}")
                finally:
                    st.session_state['running'] = False
                    st.session_state['current_step'] = 1 # 処理完了後、ステップ1に戻す


else:
    # 失敗時のみ、詳細なエラーメッセージを表示
    st.error(f"データベースの初期化に失敗しました。ファイル '{KNOWLEDGE_BASE_PATH}' の存在と中身を確認してください。")
