import streamlit as st
import os
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# ==========================================
# ۱. مدیریت زیرساخت شبکه و عبور از تحریم‌های API
# ==========================================
# به دلیل مسدود بودن دسترسی IPهای ایران به سرورهای Google، 
# ترافیک پایتون به صورت محلی از طریق پورت‌های پروکسی (پروتکل HTTP/HTTPS) هدایت شده است.
os.environ["HTTP_PROXY"] = "http://127.0.0.1:2081"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:2081"

# ==========================================
# ۲. تنظیمات هسته RAG و اتصال به مدل‌های چندوجهی
# ==========================================
# نکته امنیتی: کلید API واقعی پیش از آپلود در گیت‌هاب حذف شده است.
api_key = "YOUR_API_KEY_HERE"

# پیکربندی مدل اصلی برای پردازش همزمان متن و تصویر (Multimodal)
Settings.llm = Gemini(api_key=api_key, model_name="models/gemini-1.5-flash")
# پیکربندی مدل تعبیه‌سازی (Embedding) برای تبدیل داده‌ها به فضای برداری
Settings.embed_model = GeminiEmbedding(api_key=api_key, model_name="models/text-embedding-004")

# ==========================================
# ۳. تنظیمات ظاهری سایت (RTL و شخصی‌سازی UI)
# ==========================================
st.set_page_config(page_title="سیستم RAG چندوجهی", page_icon="🧠", layout="wide")

# تزریق کدهای CSS برای راست‌چین کردن و اصلاح فونت‌های فارسی
st.markdown("""
<style>
    * { direction: rtl; text-align: right; }
    .stTextInput input, .stChatInput textarea { direction: rtl !important; text-align: right !important; }
    html, body, [class*="css"] { font-family: 'Tahoma', 'Vazir', sans-serif; }
</style>
""", unsafe_allow_html=True)

# منوی کناری (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4140/4140047.png", width=120)
    st.markdown("### 👩‍💻 توسعه‌دهنده: غزال اسدی")
    st.markdown("**پروژه:** پردازش زبان طبیعی")
    st.markdown("**استاد:** جناب آقای دکتر جلالی")
    st.markdown("**دانشگاه:** قم")
    st.divider()
    st.info("💡 **درباره سیستم:**\nاین پلتفرم یک دستیار هوشمند مبتنی بر معماری RAG چندوجهی است که قادر به درک همزمان متن و تصویر از اسناد PDF می‌باشد.")

# ==========================================
# ۴. بدنه اصلی و منطق پردازش اسناد
# ==========================================
st.title("🧠 دستیار هوشمند پرسش و پاسخ اسناد (Multimodal RAG)")
st.markdown("فایل PDF مقاله خود را آپلود کنید تا سیستم آن را پردازش کند. سپس می‌توانید هر چند تا سوال که دوست دارید درباره متن یا تصاویر آن بپرسید.")

uploaded_file = st.file_uploader("📂 فایل PDF مقاله را اینجا آپلود کنید", type="pdf")

if uploaded_file:
    # جلوگیری از پردازش مجدد فایل در هر بار رفرش شدن رابط کاربری
    if "query_engine" not in st.session_state:
        with st.spinner('⚙️ در حال پردازش سند، استخراج متن و تصاویر و ساخت دیتابیس برداری (لطفاً صبور باشید)...'):
            
            # ذخیره موقت فایل آپلود شده در سرور
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # خواندن اطلاعات فایل و تبدیل آن به Index برداری
            reader = SimpleDirectoryReader(input_files=["temp.pdf"])
            documents = reader.load_data()
            index = VectorStoreIndex.from_documents(documents)
            
            # ساخت موتور جستجو از روی دیتابیس و ذخیره در حافظه پنهان نشست
            st.session_state.query_engine = index.as_query_engine()
            
        st.success("✅ فایل با موفقیت پردازش و ایندکس شد. حالا می‌توانید سوالات خود را بپرسید.")
        st.divider()

    # راه‌اندازی حافظه تاریخچه چت
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "سلام! من سند شما را خواندم. چه سوالی درباره معماری یا تصاویر آن دارید؟"}]

    # رندر کردن پیام‌های قبلی در صفحه
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # دریافت پرسش جدید از کاربر
    if prompt := st.chat_input("سؤال خود را اینجا بنویسید..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("در حال تحلیل و بازیابی اطلاعات مرتبط..."):
                # اجرای پرسش روی موتور RAG واقعی
                response = st.session_state.query_engine.query(prompt)
                full_response = str(response)
            
            message_placeholder = st.empty()
            
            # افکت تایپ شدن ایمن برای متون فارسی
            words = full_response.split()
            for i in range(len(words)):
                displayed_text = " ".join(words[:i+1])
                message_placeholder.markdown(displayed_text + " ▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})