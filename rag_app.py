import streamlit as st
import os
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader 
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
    st.error("エラー: Secretsに 'GEMINI_API_KEY' が設定されていません。")
    st.stop() 

KNOWLEDGE_BASE_PATH = "knowledge_base.txt" 
PERSIST_DIR = "chroma_db_cache" 
# 【セキュリティ修正1: リソース乱用対策】入力の最大文字数を設定
MAX_INPUT_LENGTH = 3500 # 3500文字に制限 (必要に応じて調整可能)

st.set_page_config(page_title="要件事実支援アプリ", layout="wide")

# --- カスタムCSS (視認性向上) の再定義 ---
st.markdown(
    """
    <style>
    /* 全体設定: フォントを読みやすく */
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* メインタイトル (H1) の視覚的区切り */
    h1 {
        color: #333333; /* 落ち着いたダークグレー */
        border-bottom: 3px solid #0078D4; /* Microsoft系の爽やかな青線 */
        padding-bottom: 5px;
    }

    /* 情報ボックス (st.info) のデザイン */
    .stAlert {
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }

    /* プライマリボタン (最終生成ボタン) の設定 */
    .stButton>button[type="primary"] {
        background-color: #0078D4; /* 鮮やかな青 */
        color: white;
        font-weight: bold;
        border-radius: 6px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button[type="primary"]:hover {
        background-color: #005A9E; 
    }
    
    /* セカンダリボタン (最初に戻る, 強制スキップ) の調整 */
    .stButton>button:not([type="primary"]) {
        background-color: #f0f0f0;
        color: #333333;
        border: 1px solid #cccccc;
        font-weight: 500;
        border-radius: 6px;
    }

    /* 実行結果 (subheader) の区切り */
    h2 {
        border-left: 5px solid #0078D4;
        padding-left: 10px;
        margin-top: 25px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

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
        # --- チャンキング最適化 ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,          # 5000文字 (実質無制限)
            chunk_overlap=0,            
            separators=["\n\n", "。", "、", "\n", " ", ""], # 句読点、改行、スペースを優先
            length_function=len,
            is_separator_regex=False
        )
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

# 【セキュリティ修正2: プロンプトインジェクション対策】
def create_safe_prompt(system_instruction, user_query, context=""):
    """ユーザー入力を明確なデリミタで囲んだ安全なプロンプトを生成する"""
    
    # 参照情報が提供されていない場合はセクションを省略
    context_section = f"""
    ---
    【参照情報】
    {context}
    ---
    """ if context else ""
    
    base_prompt = f"""
    {system_instruction}

    {context_section}

    【ユーザーが指定した事案】
    ***START_OF_USER_QUERY***
    {user_query}
    ***END_OF_USER_QUERY***
    """
    return base_prompt

@st.cache_data(ttl=600)
def check_query_relevance(query):
    """入力されたクエリが法律関連の事案であるかをAIに判定させる (ステップ1: ガードレール)"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) 
    
    system_instruction = """
    あなたは入力されたテキストを分類するAIです。以下のユーザー入力は、**法的な紛争や主張**に関連する「事案の記述」ですか？
    全く関係のない雑談、レシピ、プログラミングコード、または意味のないランダムな文字列である場合は「No」とだけ回答してください。
    それ以外の場合は「Yes」とだけ回答してください。
    回答は「Yes」または「No」のみを厳守してください。
    """
    
    prompt = create_safe_prompt(system_instruction, query)
    
    try:
        response = llm.invoke(prompt)
        # LLMの出力からデリミタを取り除く可能性のある文字をクリーンアップ
        return response.content.strip().upper().replace("*", "").replace("`", "")
    except Exception as e:
        st.warning(f"クエリ関連性チェック中にエラーが発生しました。スキップします。詳細: {e}")
        return "YES" # チェック失敗時は安全のため実行を許可

def check_for_missing_facts(db, query):
    """要件事実の作成に足りない事実があるかチェックし、足りない事実を返す (ステップ2: 事実補完)"""
    
    docs = db.similarity_search(query, k=3) 
    context = "\n".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    system_instruction = """
    あなたは要件事実の専門家です。提供された参照情報に基づき、ユーザーが指定した事案を読み、この事案に基づいて要件事実を作成する場合、**決定的に不足している主要事実**または**曖昧な主要事実**を特定し、ユーザーに補完を促す文章を作成してください。
    不足している主要事実、もしくは曖昧な主要事実がない場合は、**必ず**「OK」とだけ回答してください。
    要件事実は、重要な間接事実についての情報は不要ですから、主要事実だけに絞って検討するようにお願いします。
    """
    
    prompt = create_safe_prompt(
        system_instruction, 
        query, 
        context
    )
    
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

    # プロンプトテンプレートを LangChain の形式で安全に定義
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """
            あなたは要件事実論の専門家AIです。法的正確性を最優先してください。提供された参照情報と【ユーザーが指定した事案】に基づいて、以下のタスクを実行してください。
            【タスク】1. 請求の趣旨を特定する。2. 最も適切な訴訟物（請求権）を特定する。3. その訴訟物に必要な要件事実（末尾のよって書きを含む）を明確な箇条書きで抽出・作成する。4. 抗弁、再抗弁があれば作成する。5. 参照した法令や判例を最後に明記する。
            参照情報は以下の通りです：
            {context}
            """),
            # ユーザー入力はデリミタで囲まれた事案として渡す
            ("user", "以下の事案について、必要な要件事実を自動作成してください。\n\n【ユーザーが指定した事案】\n***START_OF_USER_QUERY***\n{contract_description}\n***END_OF_USER_QUERY***"),
        ]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"contract_description": description, "context": context})
    return response

# ====================================================
# 2. Streamlitのウェブアプリ画面構築
# ====================================================

# --- ユーティリティ関数: ステップをリセットし最初に戻る ---
def reset_workflow():
    # Streamlitのバグ回避のため、セッションステートをクリアし、キーを強制更新
    st.session_state['current_step'] = 1
    
    # ワークフローに必要なキーを削除
    keys_to_delete = ['original_query', 'edited_query_for_step2', 'initial_query', 'fact_feedback', 'running']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    
    # 入力ウィジェットのキーを更新し、新しい空のウィジェットを強制描画させる
    st.session_state['input_key'] = str(uuid.uuid4())
    
    st.rerun() 

# --- アプリの状態管理 ---
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 1 
if 'original_query' not in st.session_state:
    st.session_state['original_query'] = "" # 全てのステップで参照する「真実の源」を初期化
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = str(uuid.uuid4()) # 入力ウィジェットのキーを初期化

st.title("⚖️ 要件事実 自動作成アシスタント")


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
    # メイン入力エリア (ステップ 1 & 2 & 3)
    # ----------------------------------------------------
    
    original_query = st.session_state.get('original_query', "")
    
    if st.session_state['current_step'] == 2:
        # ステップ2の入力エリア
        st.subheader("💡 AIからのフィードバック")
        st.warning(f"以下の不足事実を追記・修正してください:\n\n{st.session_state['fact_feedback']}")
        
        # 以前のクエリを編集可能なテキストエリアに表示
        edited_query = st.text_area(
            "【不足事実を追記・修正してください】",
            value=original_query, # original_query の最新値を表示
            height=350,
            key="edited_query_for_step2", # ステップ2専用のキー
            max_chars=MAX_INPUT_LENGTH # 【セキュリティ修正1: リソース乱用対策】
        )

    else:
        # ステップ1と3の入力エリア
        current_query = st.text_area(
            "【事案の概要を入力してください】",
            value=original_query, # original_query の値を表示
            height=300,
            placeholder=f"例：\nＸの言い分\n私は、令和６年４月６日に、父Ａから相続して私が所有していた甲土地を、是非欲しいと言ってきた友人のＹに売りました。代金は２０００万円で、支払日は令和６年５月６日の約束で、同年４月６日にＹに甲土地を引き渡しました。ところが、Ｙは、いろいろと文句を言ってその代金を支払ってくれません。そこで、上記売買契約に基づいて代金２０００万円の支払を求めます。\n\nＹの言い分\n甲土地を売買することについては私もＸも異論がなかったのですが、結局、代金額について折り合いがつきませんでした。甲土地は、Ｘが相続で取得したのではなく、Ｘの叔父Ｂから贈与されたもののはずですから、Ｘは嘘をついています。　\n\n（最大{MAX_INPUT_LENGTH}文字）",
            key=st.session_state['input_key'], # ランダムなキーを使用
            max_chars=MAX_INPUT_LENGTH # 【セキュリティ修正1: リソース乱用対策】
        )
        # 入力された値を original_query にバインド
        st.session_state['original_query'] = current_query  
    
    # ----------------------------------------------------
    # ボタンとロジックの実行
    # ----------------------------------------------------
    
    # ボタン配置: メインボタンとユーティリティボタンを横並びにする
    col_main, col_reset = st.columns([0.75, 0.25]) 

    # --- 2. 最初に戻るボタン (col_reset) ---
    with col_reset:
        # Step 2 の場合は、メインボタンが2つあるため高さを調整
        if st.session_state['current_step'] == 2:
             # 強制スキップボタンの分だけ高さを合わせるため、スペースは短くする
             st.markdown("<div style='height: 1px;'></div>", unsafe_allow_html=True) 
        else:
            st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True) 
        if st.button("最初に戻る", help="ワークフローをステップ1にリセットします。", use_container_width=True):
            reset_workflow()
            
    # --- 1. メインボタン (col_main) ---
    with col_main:
        
        # Step 2: 2つのボタンを並べる (再チェック/強制スキップ)
        if st.session_state['current_step'] == 2:
            
            st.markdown("---")
            st.subheader("実行オプション")
            
            col_recheck, col_force = st.columns([0.5, 0.5])
            
            # --- オプション A: AIに再チェックさせる (既存ロジックの維持) ---
            with col_recheck:
                if st.button("修正内容で再チェックし、次のステップへ", type="primary", disabled=is_running, help="AIが修正後の事案を再度チェックし、不足事実がない場合に最終生成に進みます。", key="btn_recheck"):
                    
                    # Step 2の入力値を取得
                    current_query = st.session_state.get('edited_query_for_step2', st.session_state['original_query'])
                    
                    if not current_query or current_query.strip() == "":
                        st.warning("事案の概要を入力してください。")
                        st.session_state['running'] = False
                        st.rerun()
                    elif len(current_query) > MAX_INPUT_LENGTH:
                        st.error(f"入力が長すぎます。{MAX_INPUT_LENGTH}文字以下にしてください。")
                        st.session_state['running'] = False
                        st.stop()
                        
                    st.session_state['original_query'] = current_query # 修正された最新のクエリを保存
                    
                    st.session_state['running'] = True
                    with st.spinner("ステップ2/3: 修正された事案で不足事実を再チェック中です..."):
                        missing_facts_recheck = check_for_missing_facts(db_instance, current_query)
                    
                    st.session_state['running'] = False
                    
                    if "OK" in missing_facts_recheck.upper():
                        st.session_state['current_step'] = 3
                        if 'fact_feedback' in st.session_state: del st.session_state['fact_feedback']
                    else:
                        st.session_state['current_step'] = 2
                        st.session_state['fact_feedback'] = missing_facts_recheck 
                        st.error("まだ不足している事実があります。AIの指摘を参考に再度追記してください。")

                    st.rerun()
                    
            # --- オプション B: 強制スキップ (ユーザー要望の新機能) ---
            with col_force:
                if st.button("この情報で要件事実を最終生成する (不足事実を無視)", disabled=is_running, help="AIのフィードバックを無視し、現在の事案記述で最終的な要件事実の生成に進みます。", key="btn_force_skip"):
                    
                    # Step 2の入力値を取得
                    current_query = st.session_state.get('edited_query_for_step2', st.session_state['original_query'])
                    
                    if not current_query or current_query.strip() == "":
                        st.warning("事案の概要を入力してください。")
                        st.session_state['running'] = False
                        st.rerun()
                    elif len(current_query) > MAX_INPUT_LENGTH:
                        st.error(f"入力が長すぎます。{MAX_INPUT_LENGTH}文字以下にしてください。")
                        st.session_state['running'] = False
                        st.stop()
                        
                    # 強制スキップ時は、編集後のクエリを保存し、ステップ3へ
                    st.session_state['original_query'] = current_query 
                    st.session_state['current_step'] = 3
                    if 'fact_feedback' in st.session_state: del st.session_state['fact_feedback']
                    st.rerun()
                    
        
        else: # Step 1 または Step 3 の場合 (単一ボタン)
            
            button_label = "次のステップへ (事実確認)" if st.session_state['current_step'] != 3 else "📝 要件事実を最終生成する"
            
            # Step 1/3の入力値 (st.session_state['original_query'] にバインド済み)
            current_query = st.session_state.get('original_query', "") 

            if st.button(button_label, type="primary", disabled=is_running): 
                
                # 入力長チェック (Step 1/3 の入力エリアは 'original_query' が最新値)
                if not current_query or current_query.strip() == "":
                    st.warning("事案の概要を入力してください。")
                    st.session_state['running'] = False 
                    st.rerun()
                elif len(current_query) > MAX_INPUT_LENGTH:
                    st.error(f"入力が長すぎます。{MAX_INPUT_LENGTH}文字以下にしてください。")
                    st.session_state['running'] = False
                    st.stop() 
                    
                # Phase 1: ガードレールチェック
                if st.session_state['current_step'] == 1:
                    st.session_state['running'] = True
                    with st.spinner("ステップ1/3: 法律関連の事案かチェック中です..."):
                        relevance = check_query_relevance(current_query)

                    if relevance == "NO":
                        st.error("入力内容は法律関連の事案として認識されませんでした。要件事実に関する具体的な事案を記述してください。")
                        st.session_state['running'] = False
                        st.rerun()
                    else:
                        # 法律関連と判断 -> Phase 2: 事実補完チェックへ
                        st.session_state['original_query'] = current_query 
                        st.session_state['running'] = True
                        with st.spinner("ステップ2/3: 不足事実のチェック中です..."):
                            missing_facts = check_for_missing_facts(db_instance, current_query) 
                        
                        st.session_state['running'] = False
                        
                        if "OK" in missing_facts.upper():
                            st.session_state['current_step'] = 3
                        else:
                            st.session_state['current_step'] = 2
                            st.session_state['fact_feedback'] = missing_facts
                        st.rerun() 

                # Phase 3: 最終生成
                elif st.session_state['current_step'] == 3:
                    st.session_state['running'] = True
                    with st.spinner("ステップ3/3: 要件事実の最終構成を生成中です..."):
                        try:
                            # 最終的に使用するクエリは st.session_state['original_query']
                            result = get_required_elements_from_rag(db_instance, st.session_state['original_query'])
                            
                            st.subheader("✅ 請求権と要件事実の構成")
                            st.markdown(result)
                            
                        except Exception as e:
                            st.error(f"処理中にエラーが発生しました。詳細: {e}")
                        finally:
                            st.session_state['running'] = False
                            st.session_state['current_step'] = 1 # 処理完了後、ステップ1に戻る

else:
    # 失敗時のみ、詳細なエラーメッセージを表示
    st.error(f"データベースの初期化に失敗しました。ファイル '{KNOWLEDGE_BASE_PATH}' の存在と中身を確認してください。")
