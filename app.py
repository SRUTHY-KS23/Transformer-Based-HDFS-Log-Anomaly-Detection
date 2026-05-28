import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import torch
import numpy as np
import tempfile
import os
import hashlib
import json
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from drain import LogParser as DrainParser
from transformer import TransformerModel
from config import *

SEQ_LEN = MAX_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="HDFS ANOMALY DETECTION", layout="wide")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("styles.css")
except:
    pass

with open("threshold.txt", "r") as f:
    THRESHOLD = float(f.read().strip())

TEMPERATURE_FIXED = 1.0
SCALE_FACTOR_FIXED = 0.5
USER_THRESHOLD_FIXED = THRESHOLD

def compute_score(logit, temperature=TEMPERATURE_FIXED, scale=SCALE_FACTOR_FIXED):
    adjusted = logit * scale / temperature
    prob = torch.sigmoid(torch.tensor(adjusted)).item()
    prob = max(1e-7, min(prob, 1 - 1e-7))
    return prob

@st.cache_resource
def load_artifacts():
    event_map = pd.read_csv("event_mapping.csv")
    template_to_final_id = {}
    for _, row in event_map.iterrows():
        template = row['Content']
        original_id = row['EventId']
        final_id = original_id + 2
        template_to_final_id[template] = final_id

    state = torch.load("hdfs_transformer_model.pt", map_location=DEVICE)
    vocab_size = state["embedding.weight"].shape[0]

    model = TransformerModel(
        vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT, MAX_LEN
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, template_to_final_id, event_map

model, template_to_final_id, event_map = load_artifacts()
VOCAB_SIZE = model.embedding.num_embeddings
CACHE_FILE = "app_cache.pkl"

def save_cache(block_dict, event_stats, raw_logits=None, file_hash=None):
    try:
        data = {
            "block_dict": block_dict,
            "event_stats": event_stats,
            "raw_logits": raw_logits if raw_logits is not None else [],
            "file_hash": file_hash
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass

def load_cache():
    if os.path.exists(CACHE_FILE) and st.session_state.block_dict is None:
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            st.session_state.block_dict = data["block_dict"]
            st.session_state.event_stats = data["event_stats"]
            st.session_state.cached_file_hash = data.get("file_hash")
            if len(data.get("raw_logits", [])) == len(st.session_state.block_dict):
                st.session_state.raw_logits = data["raw_logits"]
            else:
                st.session_state.raw_logits = []
            st.session_state.results = None
            st.session_state.event_stats_by_label = None
        except Exception:
            pass

if "block_dict" not in st.session_state:
    st.session_state.block_dict = None
if "results" not in st.session_state:
    st.session_state.results = None
if "event_stats" not in st.session_state:
    st.session_state.event_stats = {}
if "event_stats_by_label" not in st.session_state:
    st.session_state.event_stats_by_label = None
if "raw_logits" not in st.session_state:
    st.session_state.raw_logits = []
if "cached_file_hash" not in st.session_state:
    st.session_state.cached_file_hash = None

load_cache()

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

st.sidebar.title("HDFS Anomaly Detection")
page = st.sidebar.radio("Navigation", [
    "Upload Data",
    "Anomaly Analysis",
    "Event Analysis",
    "Block Investigation",
])


if page == "Upload Data":
    st.markdown("<h2 style='color:#9ac7ff;'>📤 Upload Data</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file", type=["log", "txt", "csv"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = get_file_hash(file_bytes)
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if (st.session_state.cached_file_hash == current_hash
                and st.session_state.block_dict is not None):
            if file_ext != "csv":
                try:
                    raw_text = file_bytes.decode("utf-8", errors="ignore")
                    log_lines = raw_text.splitlines()
                    st.subheader("Log Preview")
                    log_text = "<br>".join(log_lines[:200])
                    st.markdown(
                        f"""<div style="height:350px;overflow-y:scroll;background-color:#0b0c10;
                        border:1px solid #1f77ff;padding:10px;font-family:monospace;
                        font-size:13px;color:#9ad1ff;border-radius:8px;">{log_text}</div>""",
                        unsafe_allow_html=True
                    )
                except:
                    pass
            st.info(f"📊 {len(st.session_state.block_dict)} blocks already loaded. Go to **Anomaly Analysis**.")
            st.stop()

        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            required = ["BlockID", "Sequence"]
            if not all(col in df.columns for col in required):
                st.error(f"CSV must contain columns: {required}")
                st.stop()

            def parse_sequence(seq_str):
                cleaned = str(seq_str).strip("[]")
                return [int(x) for x in cleaned.split() if x.isdigit()]

            sequences = df["Sequence"].apply(parse_sequence).tolist()
            block_ids = df["BlockID"].values
            block_dict = {}
            event_stats = {}
            for bid, seq in zip(block_ids, sequences):
                block_dict[bid] = seq
                for eid in seq:
                    event_stats[eid] = event_stats.get(eid, 0) + 1

            st.session_state.block_dict = block_dict
            st.session_state.event_stats = event_stats
            st.session_state.results = None
            st.session_state.raw_logits = []
            st.session_state.event_stats_by_label = None
            st.session_state.cached_file_hash = current_hash
            save_cache(block_dict, event_stats, file_hash=current_hash)
            st.success(f"✅ Loaded {len(block_dict)} blocks from CSV.")

        else:
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            log_lines = raw_text.splitlines()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
                tmp.write(raw_text)
                tmp_path = tmp.name

            parser = DrainParser(
                log_format='<Date> <Time> <Pid> <Level> <Component>: <Content>',
                indir='./', outdir='./temp_drain/', depth=4, st=0.4, maxChild=100,
                rex=[r'blk_-?\d+', r'(\d+\.){3}\d+', r'\d+'], keep_para=True
            )
            parser.parse(tmp_path)

            struct_file = os.path.join('./temp_drain/', os.path.basename(tmp_path) + '_structured.csv')
            block_dict = {}
            event_stats = {}
            unknown_templates = 0
            chunk_iter = pd.read_csv(struct_file, chunksize=100000)
            for df_chunk in chunk_iter:
                for _, row in df_chunk.iterrows():
                    content = row['Content']
                    blk_match = re.search(r"(blk_[-0-9]+)", content)
                    if not blk_match:
                        continue
                    block_id = blk_match.group(1)
                    template = row['EventTemplate']
                    if template in template_to_final_id:
                        event_id = template_to_final_id[template]
                    else:
                        event_id = 3
                        unknown_templates += 1
                    if event_id >= VOCAB_SIZE:
                        event_id = event_id % VOCAB_SIZE
                    block_dict.setdefault(block_id, []).append(event_id)
                    event_stats[event_id] = event_stats.get(event_id, 0) + 1

            os.unlink(tmp_path)
            st.session_state.block_dict = block_dict
            st.session_state.event_stats = event_stats
            st.session_state.results = None
            st.session_state.raw_logits = []
            st.session_state.event_stats_by_label = None
            st.session_state.cached_file_hash = current_hash
            save_cache(block_dict, event_stats, file_hash=current_hash)

            st.subheader("Log File Summary")
            col1, col2 = st.columns(2)
            col1.metric("Total Log Lines", len(log_lines))
            col2.metric("Detected Blocks", len(block_dict))

            st.subheader("Log Preview")
            log_text = "<br>".join(log_lines[:200])
            st.markdown(
                f"""<div style="height:350px;overflow-y:scroll;background-color:#0b0c10;
                border:1px solid #1f77ff;padding:10px;font-family:monospace;
                font-size:13px;color:#9ad1ff;border-radius:8px;">{log_text}</div>""",
                unsafe_allow_html=True
            )

        st.success("Data loaded. Go to **Anomaly Analysis** to see predictions.")

elif page == "Anomaly Analysis":

    # ── Header
    st.markdown(
        """
        <h2 style="color:#9ac7ff; display:flex; align-items:center; gap:10px; margin-bottom:4px;">
        <img src="https://cdn-icons-png.flaticon.com/512/10397/10397132.png" width="24"
        style="filter: brightness(0) invert(1);">
        Anomaly Detection Results
        </h2>
        <hr style="border:1px solid #1f77ff; margin-top:0; margin-bottom:1.2rem;">
        """,
        unsafe_allow_html=True
    )

    if st.session_state.block_dict is None:
        st.warning("⚠️ Please upload a file first")
        st.stop()

    block_dict      = st.session_state.block_dict
    temperature     = TEMPERATURE_FIXED
    scale_factor    = SCALE_FACTOR_FIXED
    fixed_threshold = USER_THRESHOLD_FIXED

    if not st.session_state.raw_logits or len(st.session_state.raw_logits) != len(block_dict):
        st.session_state.raw_logits = []
        progress_bar = st.progress(0)
        for i, (block_id, event_ids) in enumerate(block_dict.items()):
            seq = event_ids[:SEQ_LEN] + [0] * max(0, SEQ_LEN - len(event_ids))
            X   = torch.tensor([seq]).to(DEVICE)
            with torch.no_grad():
                logit = model(X).item()
            st.session_state.raw_logits.append(logit)
            progress_bar.progress((i + 1) / len(block_dict))
        save_cache(block_dict, st.session_state.event_stats,
                   st.session_state.raw_logits, st.session_state.cached_file_hash)

    def compute_predictions(logits, thr):
        results = []
        event_counts_anom   = Counter()
        event_counts_normal = Counter()
        block_ids = list(block_dict.keys())
        for bid, logit in zip(block_ids, logits):
            prob  = compute_score(logit, temperature, scale_factor)
            label = "ANOMALY" if prob >= thr else "NORMAL"
            results.append({"Block ID": bid, "Score": round(prob, 8), "Prediction": label})
            ev_list = block_dict[bid]
            if label == "ANOMALY":
                event_counts_anom.update(ev_list)
            else:
                event_counts_normal.update(ev_list)
        return results, event_counts_anom, event_counts_normal

    results, anom_counts, norm_counts = compute_predictions(
        st.session_state.raw_logits, fixed_threshold
    )
    df = pd.DataFrame(results)
    st.session_state.results = df
    st.session_state.event_stats_by_label = {"ANOMALY": anom_counts, "NORMAL": norm_counts}

    anomalies = df[df["Prediction"] == "ANOMALY"]
    normals   = df[df["Prediction"] == "NORMAL"]
    n_total   = len(df)
    n_anom    = len(anomalies)
    n_norm    = len(normals)
    anom_pct  = round(n_anom / n_total * 100, 1) if n_total > 0 else 0
    norm_pct  = round(n_norm / n_total * 100, 1) if n_total > 0 else 0

  
    st.markdown("""
        <style>
            .metric-card {
                background-color: #0e1117;
                border: 1px solid #31333f;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                height: 200px;
                width: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                box-sizing: border-box;
            }
            .metric-title {
                color: #9ac7ff;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1.2px;
                margin-bottom: 8px;
                text-transform: uppercase;
            }
            .metric-value {
                font-size: 28px;
                font-weight: 800;
                margin: 0;
            }
            .metric-delta {
                font-size: 13px;
                margin-top: 4px;
            }
        </style>
    """, unsafe_allow_html=True)

  
    m1, m2, m3, chart_col = st.columns([1, 1, 1, 1.8])

    with m1:
        st.markdown(f'''<div class="metric-card" style="border-color: #1f77ff;">
            <div class="metric-title">Total Blocks</div>
            <div class="metric-value" style="color: white;">{n_total}</div>
        </div>''', unsafe_allow_html=True)

    with m2:
        
        st.markdown(f'''<div class="metric-card" style="border-color: #ff4b4b; background-color: #1a0a0a;">
            <div class="metric-title">🚨 Anomalies</div>
            <div class="metric-value" style="color: #ff4b4b;">{n_anom}</div>
        </div>''', unsafe_allow_html=True)

    with m3:
      
        st.markdown(f'''<div class="metric-card" style="border-color: #00c49a; background-color: #0a1a0f;">
            <div class="metric-title">✅ Normal</div>
            <div class="metric-value" style="color: #00c49a;">{n_norm}</div>
        </div>''', unsafe_allow_html=True)

    with chart_col:
        if n_total > 0:
            st.markdown('<p style="color:#9ac7ff; font-size:12px; font-weight:700; text-align:center; margin-bottom:0px; text-transform:uppercase;">Block Distribution Analysis</p>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(4, 2.5), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")

            sizes = [n_anom, n_norm]
            colors = ["#ff4b4b", "#00c49a"] 
            
            wedges, texts, autotexts = ax.pie(
                sizes, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colors,
                pctdistance=0.75,
                explode=[0.1, 0] if n_anom > 0 else [0, 0],
                wedgeprops={'width': 0.4, 'edgecolor': '#0d1117', 'linewidth': 2}
            )

            plt.setp(autotexts, size=8, weight="bold", color="white")

            ax.legend(
                wedges, ["Anomaly", "Normal"],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
                frameon=False,
                labelcolor="white"
            )

            ax.axis('equal')  
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No data available")

    st.markdown("<hr style='border:0.5px solid #1f3355; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ac7ff; font-size:15px; font-weight:700; margin-bottom:6px;'>📋 Detailed Results</p>", unsafe_allow_html=True)
    
    df["sort"] = df["Prediction"].apply(lambda x: 0 if x == "ANOMALY" else 1)
    df = df.sort_values(["sort", "Score"], ascending=[True, False]).drop(columns=["sort"])
    
    styled_df = df.style.map(
        lambda val: "color: #ff4b4b; font-weight: bold" if val == "ANOMALY"
        else "color: #00c49a; font-weight: bold" if val == "NORMAL" else "",
        subset=["Prediction"]
    )
    

    st.dataframe(styled_df, width="stretch", height=450)
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results as CSV", csv, "anomaly_results.csv")

elif page == "Event Analysis":
    st.markdown(
        "<h2><span style='color:white;'>⚠</span> <span style='color:#9ac7ff;'>Events Strongly Associated with Anomalies</span></h2>",
        unsafe_allow_html=True
    )

    if st.session_state.event_stats_by_label is None:
        st.warning("Please run **Anomaly Analysis** first.")
        st.stop()

    event_counts_anom   = st.session_state.event_stats_by_label["ANOMALY"]
    event_counts_normal = st.session_state.event_stats_by_label["NORMAL"]
    all_event_ids = set(event_counts_anom.keys()) | set(event_counts_normal.keys())

    rows = []
    for eid in all_event_ids:
        anom_count = event_counts_anom.get(eid, 0)
        norm_count = event_counts_normal.get(eid, 0)

        
        if anom_count > norm_count and anom_count > 0:
            if eid == 0:
                desc = "PADDING"
                original_id = 0
            elif eid == 3:
                desc = "OOV (Unknown Template)"
                original_id = 3
            else:
                original_id = eid - 2
                row = event_map[event_map["EventId"] == original_id]
                desc = row["Content"].values[0] if not row.empty else f"Event_{eid}"

            rows.append({"Event ID": original_id, "Event Description": desc})

    df_events = pd.DataFrame(rows)

    if df_events.empty:
        st.info("No anomaly-related events detected.")
    else:
        df_events = df_events.sort_values("Event ID")
        styled_events = df_events.style.set_properties(**{
            "white-space": "pre-wrap", "text-align": "left"
        })
        st.dataframe(styled_events, width="stretch")

        event_list = df_events["Event Description"].tolist()
        sentence   = ", ".join(event_list)

        st.markdown(
            f"""
<div style="margin-top:18px; padding:14px 18px;
            background:rgba(154,199,255,0.05);
            border-left:3px solid #9ac7ff;
            border-radius:6px; font-size:13px;
            color:#c2c0b6; line-height:1.8;">
 
  The anomalous blocks in the uploaded log dataset are primarily associated with the following log events:<br><br>
  <span style="color:#9ac7ff; font-weight:600;">{sentence}</span><br><br>
  <span style="color:#9c9a92;">
  These events appear more frequently in anomalous blocks than in normal blocks,
  indicating abnormal patterns in the HDFS system behavior.
  </span>
</div>
""",
            unsafe_allow_html=True,
        )


elif page == "Block Investigation":

    st.markdown(
        """
<h2 style="color:#9ac7ff; display:flex; align-items:center; gap:10px;">
<img src="https://cdn-icons-png.flaticon.com/512/9479/9479251.png"
     width="24"
     style="filter: brightness(0) invert(1);">
Block Investigation
</h2>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.block_dict is None:
        st.warning("Upload data first")
        st.stop()

    block_dict = st.session_state.block_dict

    block_id = st.text_input("Enter Block ID")

    if block_id and block_id in block_dict:
        event_ids = block_dict[block_id]
        seq = event_ids[:SEQ_LEN] + [0] * max(0, SEQ_LEN - len(event_ids))
        X = torch.tensor([seq]).to(DEVICE)

        with torch.no_grad():
            logit, attn_weights = model(X, return_attention=True)

        logit = logit.item()
        prob = compute_score(logit)
        label = "ANOMALY" if prob >= USER_THRESHOLD_FIXED else "NORMAL"
        if label == "ANOMALY":
          
            
            
            st.error(f"🚨 Block {block_id} is detected as ANOMALOUS (Anomaly score: {prob:.6f})")
        else:
            st.success(f"✅ Block {block_id} is NORMAL")
        
        if label == "ANOMALY":

            st.subheader("Explanation")

            attn = attn_weights.squeeze(0).cpu().numpy()
            attn = attn / (attn.sum() + 1e-8)

            event_attn = []
            for pos, eid in enumerate(seq):
                if eid != 0:
                    event_attn.append((pos, eid, float(attn[pos].item())))

            event_attn.sort(key=lambda x: x[2], reverse=True)

            unique_events = []
            seen = set()

            for pos, eid, att in event_attn:
               
                if eid == 0 or eid == 3:
                    continue

                if eid not in seen:
                    unique_events.append((pos, eid, att))
                    seen.add(eid)

            def get_event_desc(original_id):
                row = event_map[event_map["EventId"] == original_id]
                if not row.empty:
                    return row["Content"].values[0]
                return f"Event {original_id}"

            rows = []

            for pos, eid, att in unique_events:
                original_id = eid - 2
                desc = get_event_desc(original_id)
                desc_lower = desc.lower()

                reason = None  

           
                if any(word in desc_lower for word in ["exception", "error", "fail", "failed"]):
                    reason = "System exception or failure detected during block processing."
                elif any(word in desc_lower for word in ["packet", "connection", "timeout", "reset"]):
                    reason = "Network communication issue detected during data transfer."
                elif any(word in desc_lower for word in ["replicate", "replication"]):
                    reason = "Block replication activity observed, possibly indicating system recovery."
                elif any(word in desc_lower for word in ["invalid", "delete", "missing"]):
                    reason = "Block metadata inconsistency detected in HDFS."
                elif "writeblock" in desc_lower:
                    reason = "Failure detected while writing block data to the DataNode."
                elif "receiveblock" in desc_lower:
                    reason = "Error occurred during block reception."
                elif "allocateblock" in desc_lower:
                    reason = "Block allocation event appears in an unusual sequence position."
                elif "pipeline" in desc_lower:
                    reason = "Data transfer pipeline disruption detected."
                elif any(word in desc_lower for word in ["disk", "volume", "storage"]):
                    reason = "Storage or disk-related issue detected during block operation."
                elif "corrupt" in desc_lower:
                    reason = "Block corruption detected in the distributed storage system."
                elif "datanode" in desc_lower and "shutdown" in desc_lower:
                    reason = "DataNode shutdown detected during block operation."
                elif "namenode" in desc_lower and "fail" in desc_lower:
                    reason = "NameNode communication failure detected."
                elif "verification succeeded" in desc_lower:
                    
                    reason = "Verification success event appears at an unusual point in the sequence."
                elif "addstoredblock" in desc_lower:
                    reason = "Block storage registration event occurring out of expected order."
                elif "served block" in desc_lower:
                    reason = "Block serve event detected at an unexpected point in sequence."
                elif "received block" in desc_lower:
                    reason = "Block receipt event appears at an abnormal sequence position."

              
                if reason is None and pos == 0:
                    reason = "Event occurs at the beginning of the sequence where it normally does not appear."

              
                if reason is None:
                    reason = "Log event pattern deviates from normal block operation behaviour."

                rows.append({
                    "Event ID": original_id,
                    "Event Description": desc[:120],
                    "Sequence Position": pos,
                    "Attention Weight": round(att, 4),
                    "Reason": reason
                })

            df_explain = pd.DataFrame(rows)
            st.dataframe(df_explain, width="stretch")

            
            st.subheader("Attention Weight Chart")

            actual_len = min(len(event_ids), SEQ_LEN)
            attn_vals_list = [round(float(attn[i].item()), 6) for i in range(actual_len)]

            peak_labels = []
            for r in rows:
                peak_labels.append(
                    {
                        "pos": r["Sequence Position"],
                        "val": r["Attention Weight"],
                        "label": f"E{r['Event ID']}",
                    }
                )

            chart_html = f"""
<div style="max-width: 600px; max-height=800px margin: 0 auto;">

  <div style="display:flex;flex-wrap:wrap;gap:16px;margin-bottom:10px;font-size:12px;color:#9ac7ff;">
    <span style="display:flex;align-items:center;gap:5px;">
      <span style="width:20px;height:2px;background:#E24B4A;display:inline-block;border-radius:1px;"></span>
      Attention weight
    </span>
    <span style="display:flex;align-items:center;gap:5px;">
      <span style="width:9px;height:9px;background:#BA7517;border-radius:50%;display:inline-block;"></span>
      Top suspicious events
    </span>
  </div>

  <div style="position:relative;width:100%;height:300px;">
    <canvas id="attnChart"></canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js"></script>

<script>
var positions  = {list(range(actual_len))};
var attnVals   = {attn_vals_list};
var peakLabels = {json.dumps(peak_labels)};

var isDark   = matchMedia('(prefers-color-scheme: dark)').matches;
var gridCol  = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
var tickCol  = isDark ? '#9c9a92' : '#73726c';
var zeroLine = isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.12)';

new Chart(document.getElementById('attnChart'), {{
  type: 'line',
  data: {{
    labels: positions.map(function(p) {{ return 'P' + p; }}),
    datasets: [
      {{
        label: 'Attention weight',
        data: attnVals,
        borderColor: '#E24B4A',
        borderWidth: 1.8,
        pointRadius: 2.5,
        pointBackgroundColor: '#E24B4A',
        pointBorderWidth: 0,
        fill: true,
        backgroundColor: function(ctx) {{
          var g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 300);
          g.addColorStop(0, 'rgba(226,75,74,0.15)');
          g.addColorStop(1, 'rgba(226,75,74,0.01)');
          return g;
        }},
        tension: 0.38
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: isDark ? '#2c2c2a' : '#ffffff',
        titleColor: isDark ? '#c2c0b6' : '#3d3d3a',
        bodyColor: isDark ? '#9c9a92' : '#73726c',
        borderColor: isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.1)',
        borderWidth: 0.5,
        padding: 10,
        cornerRadius: 8,
        callbacks: {{
          label: function(item) {{ return ' Attention: ' + item.parsed.y.toFixed(4); }}
        }}
      }},
      annotation: {{
        annotations: peakLabels.reduce(function(acc, pk) {{
          acc['dot_' + pk.pos] = {{
            type: 'point',
            xValue: pk.pos,
            yValue: pk.val,
            radius: 7,
            backgroundColor: '#BA7517',
            borderColor: isDark ? '#1a1a1a' : '#ffffff',
            borderWidth: 1.5
          }};
          acc['label_' + pk.pos] = {{
            type: 'label',
            xValue: pk.pos,
            yValue: pk.val,
            yAdjust: 20,
            content: pk.label,
            font: {{ size: 11, weight: '500' }},
            color: '#BA7517',
            backgroundColor: 'transparent',
            borderWidth: 0
          }};
          return acc;
        }}, {{}})
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: tickCol, font: {{ size: 11 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 }},
        grid: {{ color: gridCol }},
        border: {{ color: zeroLine }},
        title: {{ display: true, text: 'Sequence position', color: tickCol, font: {{ size: 11 }} }}
      }},
      y: {{
        ticks: {{ color: tickCol, font: {{ size: 11 }}, callback: function(v) {{ return v.toFixed(3); }} }},
        grid: {{ color: gridCol }},
        border: {{ color: zeroLine }},
        suggestedMin: 0,
        title: {{ display: true, text: 'Attention weight', color: tickCol, font: {{ size: 11 }} }}
      }}
    }}
  }}
}});
</script>
"""

            st.components.v1.html(chart_html, height=430)
            top_row = max(rows, key=lambda x: x["Attention Weight"])

            st.markdown(
                f"""
<div style='margin-top:14px; padding:10px 14px;
            background:rgba(154,199,255,0.07);
            border-left:3px solid #9ac7ff;
            border-radius:6px; font-size:12px; color:#9ac7ff;'>
  The event <b>{top_row['Event ID']}</b>
  at sequence position <b>P{top_row['Sequence Position']}</b>
  receives the highest attention weight
  and contributes most to the anomaly prediction.
</div>
""",
                unsafe_allow_html=True,
            )
    elif block_id:
        st.warning("Block ID not found in dataset.")