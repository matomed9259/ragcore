"""Streamlit RAG UI."""
import streamlit as st, httpx, time

st.set_page_config(page_title="RAGCore", page_icon="🔍", layout="wide")
st.title("🔍 RAGCore — Enterprise RAG Engine")

API = st.sidebar.text_input("API URL", "http://localhost:8000")
use_rerank = st.sidebar.checkbox("Cross-Encoder Reranking", True)
top_k = st.sidebar.slider("Top-K Results", 3, 15, 8)

tab1, tab2 = st.tabs(["💬 Query", "📄 Ingest"])

with tab2:
    uploaded = st.file_uploader("Upload Document", type=["pdf","docx","txt","html","csv","md"])
    if uploaded and st.button("📤 Ingest"):
        with st.spinner("Ingesting..."):
            r = httpx.post(f"{API}/ingest", files={"file": (uploaded.name, uploaded.getvalue())}, timeout=120)
            if r.status_code == 200:
                st.success(f"✅ Indexed {r.json()['chunks']} chunks!")
            else:
                st.error(f"Error: {r.text}")

with tab1:
    q = st.text_input("Ask a question about your document:")
    if st.button("🔍 Search", disabled=not q):
        with st.spinner("Retrieving..."):
            t0 = time.time()
            r = httpx.post(f"{API}/query", json={"query": q, "k": top_k, "rerank": use_rerank}, timeout=60)
            latency = (time.time()-t0)*1000
        if r.status_code == 200:
            data = r.json()
            st.markdown(f"**Answer** ({latency:.0f}ms):\n\n{data['answer']}")
            with st.expander("📚 Sources"):
                for i, src in enumerate(data["sources"]):
                    st.markdown(f"**Source {i+1}:** {src['content']}...")
        else:
            st.error(r.text)
