import os
import re
import tomllib
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from io import BytesIO
from google import genai


# =========================
# تنظیمات اصلی برنامه
# =========================

MODEL_NAME = "gemini-flash-latest"
APP_TITLE = "دستیار هوشمند پرسش و پاسخ اسناد"
APP_SUBTITLE = "فایل PDF مقاله خود را آپلود کنید تا سیستم آن را پردازش کند. سپس می‌توانید هر چندتا سؤال که دوست دارید درباره متن یا تصاویر آن بپرسید."


# =========================
# تنظیمات صفحه
# =========================

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)




st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif !important;
        direction: rtl;
    }

    .stApp {
        background-color: #ffffff;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 4rem;
        padding-bottom: 2rem;
        direction: rtl;
    }

    section[data-testid="stSidebar"] {
        background-color: #f5f6fa;
        border-left: 1px solid #e5e7eb;
        direction: rtl;
    }

    section[data-testid="stSidebar"] * {
        direction: rtl;
        text-align: right;
    }

    .hero-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #30313d;
        margin-bottom: 12px;
        line-height: 1.5;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 15px;
        color: #333333;
        margin-bottom: 34px;
        line-height: 2;
    }

    .upload-label {
        text-align: right;
        font-weight: 600;
        margin-bottom: 8px;
        color: #30313d;
    }

    div[data-testid="stFileUploader"] {
        direction: rtl;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #f1f2f6;
        border: none;
        border-radius: 10px;
        padding: 12px;
    }

    div[data-testid="stFileUploader"] label {
        direction: rtl;
        text-align: right;
    }

    .chat-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin: 20px 0;
        direction: rtl;
    }

    .chat-bubble-user {
        background-color: #ffffff;
        color: #2f313d;
        padding: 16px 18px;
        border-radius: 12px;
        width: 100%;
        line-height: 2;
        font-size: 15px;
        text-align: right;
    }

    .chat-bubble-assistant {
        background-color: #f7f8fb;
        color: #2f313d;
        padding: 18px 20px;
        border-radius: 12px;
        width: 100%;
        line-height: 2.1;
        font-size: 15px;
        text-align: right;
    }

    .avatar-user {
        min-width: 42px;
        height: 42px;
        border-radius: 10px;
        background-color: #ffa51f;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 22px;
    }

    .avatar-assistant {
        min-width: 42px;
        height: 42px;
        border-radius: 10px;
        background-color: #ff4d57;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 22px;
    }

    .sidebar-profile {
        text-align: center !important;
        margin-top: 28px;
        margin-bottom: 24px;
    }

    .profile-circle {
        width: 116px;
        height: 116px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ffd54f, #ffb74d);
        margin: 0 auto 22px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
    }

    .sidebar-info {
        font-size: 15px;
        color: #30313d;
        line-height: 2.3;
        font-weight: 500;
    }

    .about-box {
        background-color: #dcecff;
        color: #00508f;
        padding: 20px;
        border-radius: 8px;
        line-height: 2.2;
        font-size: 15px;
        margin-top: 24px;
        text-align: center;
    }

    .small-muted {
        color: #6b7280;
        font-size: 13px;
        line-height: 2;
    }

    .status-box {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 10px;
        color: #374151;
        line-height: 2;
        margin-top: 10px;
        font-size: 14px;
    }

    textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    input {
        direction: rtl !important;
        text-align: right !important;
    }

    div[data-testid="stChatInput"] {
        direction: rtl;
    }

    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    .stButton button {
        direction: rtl;
        border-radius: 8px;
    }

    hr {
        margin: 28px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# خواندن API Key
# =========================

def load_api_key():
    secrets_path = os.path.join(".streamlit", "secrets.toml")

    if not os.path.exists(secrets_path):
        st.error("فایل secrets.toml در پوشه .streamlit پیدا نشد.")
        st.stop()

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)

    api_key = secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("کلید GEMINI_API_KEY داخل فایل secrets.toml پیدا نشد.")
        st.stop()

    return api_key


@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# =========================
# پردازش PDF
# =========================

def normalize_text(text):
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text_and_pages(uploaded_file):
    """
    خروجی:
    full_text: کل متن PDF
    page_texts: لیست متن هر صفحه
    page_images: لیست تصویر رندرشده هر صفحه
    """
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_texts = []
    page_images = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        text = page.get_text("text")
        text = normalize_text(text)
        page_texts.append(text)

        # رندر صفحه به تصویر برای پرسش‌های تصویری
        pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        img_bytes = pix.tobytes("png")
        image = Image.open(BytesIO(img_bytes))
        page_images.append(image)

    full_text = "\n\n".join(page_texts)

    return full_text, page_texts, page_images


def make_chunks(page_texts, chunk_size=1300, overlap=250):
    chunks = []

    for page_number, text in enumerate(page_texts, start=1):
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "page": page_number,
                "text": chunk_text
            })

            start += chunk_size - overlap

    return chunks


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\sآ-ی]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


def retrieve_relevant_chunks(question, chunks, top_k=5):
    """
    Retrieval ساده بر اساس هم‌پوشانی کلمات.
    برای پروژه دانشگاهی قابل توضیح و سبک است.
    """
    q_tokens = set(tokenize(question))

    scored_chunks = []
    for chunk in chunks:
        c_tokens = set(tokenize(chunk["text"]))
        score = len(q_tokens.intersection(c_tokens))

        # کمی امتیاز اضافه برای وجود مستقیم عبارت‌ها
        for token in q_tokens:
            if token in chunk["text"].lower():
                score += 0.5

        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    selected = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    if not selected:
        selected = [chunk for _, chunk in scored_chunks[:min(top_k, len(scored_chunks))]]

    return selected


def build_context(relevant_chunks):
    context_parts = []

    for item in relevant_chunks:
        context_parts.append(
            f"صفحه {item['page']}:\n{item['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_relevant_page_images(relevant_chunks, page_images, max_images=3):
    pages = []
    for item in relevant_chunks:
        page = item["page"]
        if page not in pages:
            pages.append(page)

    selected_images = []
    for page in pages[:max_images]:
        index = page - 1
        if 0 <= index < len(page_images):
            selected_images.append(page_images[index])

    return selected_images


# =========================
# تولید پاسخ با Gemini
# =========================

def answer_with_gemini(client, question, context, relevant_images=None):
    system_prompt = """
تو یک دستیار هوشمند برای پرسش و پاسخ درباره اسناد PDF هستی.
وظیفه تو پاسخ دادن فقط بر اساس متن و تصاویر بازیابی‌شده از سند است.

قواعد پاسخ:
1. پاسخ را فارسی، دقیق و ساختاریافته بده.
2. اگر پاسخ در سند وجود ندارد، واضح بگو که در بخش‌های بازیابی‌شده سند اطلاعات کافی دیده نمی‌شود.
3. اگر از متن سند استفاده می‌کنی، در صورت امکان شماره صفحه را ذکر کن.
4. از حدس‌زدن بی‌دلیل خودداری کن.
5. اگر سؤال درباره شکل، نمودار، جدول یا تصویر بود، از تصاویر صفحات مرتبط هم کمک بگیر.
"""

    user_prompt = f"""
{system_prompt}

متن‌های بازیابی‌شده از سند:

{context}

سؤال کاربر:
{question}

پاسخ نهایی را به فارسی بنویس.
"""

    contents = [user_prompt]

    if relevant_images:
        for img in relevant_images:
            contents.append(img)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents
    )

    return response.text


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-profile">
            <div class="profile-circle">👩‍💻</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-info">
            <b>توسعه‌دهنده:</b> غزال اسدی 👩‍🎓<br>
            <b>پروژه:</b> پردازش زبان طبیعی<br>
            <b>استاد:</b> جناب آقای دکتر جلالی<br>
            <b>دانشگاه:</b> قم
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="about-box">
            💡 <b>درباره سیستم:</b><br>
            این پلتفرم یک دستیار هوشمند مبتنی بر معماری RAG چندوجهی است
            که قادر به درک هم‌زمان متن و تصویر از اسناد PDF می‌باشد.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="status-box">
            <b>مدل فعال:</b><br>
            {MODEL_NAME}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Header
# =========================

st.markdown(
    f"""
    <div class="hero-title">
        🧠 {APP_TITLE} <span style="font-size: 36px;">(Multimodal RAG)</span>
    </div>
    <div class="hero-subtitle">
        {APP_SUBTITLE}
    </div>
    """,
    unsafe_allow_html=True
)




if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "سلام! من سند شما را می‌خوانم. چه سؤالی درباره متن یا تصاویر آن دارید؟"
        }
    ]

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "page_images" not in st.session_state:
    st.session_state.page_images = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# =========================
# API Client
# =========================

api_key = load_api_key()
client = get_gemini_client(api_key)


# =========================
# File Upload
# =========================

st.markdown('<div class="upload-label">📁 فایل PDF مقاله را اینجا آپلود کنید</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    if st.session_state.pdf_name != uploaded_file.name:
        with st.spinner("در حال پردازش PDF..."):
            try:
                full_text, page_texts, page_images = extract_pdf_text_and_pages(uploaded_file)
                chunks = make_chunks(page_texts)

                st.session_state.chunks = chunks
                st.session_state.page_images = page_images
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name

                st.success("PDF با موفقیت پردازش شد.")

            except Exception as e:
                st.error("هنگام پردازش PDF خطا رخ داد.")
                st.exception(e)
                st.stop()

    if st.session_state.pdf_processed:
        st.markdown(
            f"""
            <div class="small-muted">
                سند فعال: <b>{st.session_state.pdf_name}</b> |
                تعداد بخش‌های متنی ساخته‌شده: <b>{len(st.session_state.chunks)}</b> |
                تعداد صفحات تصویری: <b>{len(st.session_state.page_images)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# نمایش پیام‌ها
# =========================

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="avatar-user">🤖</div>
                <div class="chat-bubble-user">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="avatar-assistant">🧠</div>
                <div class="chat-bubble-assistant">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# ورودی سؤال
# =========================

question = st.chat_input("سؤال خود را اینجا بنویسید...")

if question:
    if not st.session_state.pdf_processed:
        st.warning("لطفاً ابتدا یک فایل PDF آپلود کنید.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("در حال جست‌وجو در سند و تولید پاسخ..."):
        try:
            relevant_chunks = retrieve_relevant_chunks(
                question=question,
                chunks=st.session_state.chunks,
                top_k=5
            )

            context = build_context(relevant_chunks)

            relevant_images = get_relevant_page_images(
                relevant_chunks=relevant_chunks,
                page_images=st.session_state.page_images,
                max_images=3
            )

            answer = answer_with_gemini(
                client=client,
                question=question,
                context=context,
                relevant_images=relevant_images
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

        except Exception as e:
            error_message = str(e)

            if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
                st.error("سهمیه API برای این مدل تمام شده یا محدود شده است. کمی بعد دوباره امتحان کنید یا مدل دیگری قرار دهید.")
            elif "NOT_FOUND" in error_message or "404" in error_message:
                st.error("مدل انتخاب‌شده در API فعلی در دسترس نیست.")
            else:
                st.error("هنگام تولید پاسخ خطا رخ داد.")
                st.exception(e)
