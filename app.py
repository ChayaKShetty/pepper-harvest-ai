import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from datetime import datetime
from io import BytesIO
import json
import html

st.set_page_config(
    page_title="Pepper Harvest AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- DESIGN --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --green: #087443;
  --dark: #123b2b;
  --lime: #c8f169;
  --muted: #6b7d72;
}
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background: linear-gradient(180deg,#effaf1 0%,#ffffff 38%,#f5faf5 100%);
}
.block-container {
  max-width: 760px;
  padding: 1.2rem 1rem 6rem;
}
#MainMenu, footer, header {visibility:hidden;}
.hero {
  background: linear-gradient(135deg,#104e32,#07884b);
  color: white;
  padding: 34px 22px;
  text-align: center;
  border-radius: 28px;
  margin-bottom: 20px;
  box-shadow: 0 14px 34px #0a6d3924;
  animation: rise .65s ease-out;
}
.logo {
  font-size: 48px;
  animation: float 3s ease-in-out infinite;
}
.hero h1 {font-size: clamp(27px,6vw,39px);margin: 8px 0;font-weight:800;}
.hero p {color:#e5f8e9;font-size:15px;margin:8px 0;}
.eyebrow {
  color:#087443;font-size:12px;font-weight:800;
  letter-spacing:2px;text-transform:uppercase;
}
.section-title {font-size:25px;font-weight:800;color:#153c2b;margin:5px 0;}
.subtle {color:#708075;font-size:14px;}
.card {
  background:white;border:1px solid #e0eee3;border-radius:20px;
  padding:18px;margin:12px 0;
  box-shadow:0 7px 24px #183d2510;
}
.metric {
  background:linear-gradient(145deg,#ffffff,#eff9f0);
  border:1px solid #d9eddd;border-radius:18px;
  padding:17px 10px;text-align:center;
}
.metric .number {font-size:29px;font-weight:800;color:#087443;}
.metric .label {font-size:12px;color:#6b7d72;}
.pill {
  display:inline-block;background:#e7f7e9;color:#087443;
  padding:5px 10px;border-radius:30px;font-size:12px;font-weight:700;
  margin:3px 4px 3px 0;
}
.guide-step {
  border-left:4px solid #16a05b;background:white;
  padding:13px 16px;margin:10px 0;border-radius:0 15px 15px 0;
  box-shadow:0 3px 14px #183d250a;
}
.stButton > button {
  border-radius:14px!important;
  min-height:46px;
  font-weight:700!important;
  transition:all .2s ease;
}
.stButton > button:hover {transform:translateY(-2px);}
div[data-testid="stFileUploader"] {
  border:2px dashed #a7d7b2;border-radius:18px;padding:12px;
  background:#ffffffb8;
}
div[data-testid="stImage"] img {border-radius:16px;}
div[data-testid="stMetric"] {
  background:white;padding:12px;border-radius:14px;border:1px solid #e1eee4;
}
@keyframes float {0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes rise {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:480px) {
 .block-container {padding: .7rem .7rem 5rem;}
 .hero {padding:27px 15px;border-radius:22px;}
 .card {padding:14px;}
}
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
defaults = {
    "page": "Home",
    "history": [],
    "current_result": None,
    "current_image": None,
    "api_key": "",
    "uploaded_bytes": None,
    "uploaded_name": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def go(page):
    st.session_state.page = page
    st.rerun()

def header():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
      <div style="font-size:30px">🌿</div>
      <div>
        <div style="font-weight:800;color:#123b2b;font-size:20px">Pepper Harvest AI</div>
        <div style="font-size:12px;color:#728477">SMART FARMING • AI DETECTION</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)

def get_predictions(result):
    """Handle the usual Roboflow workflow response and show clear errors."""
    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError("Unexpected response from Roboflow.")
    predictions = result.get("predictions", {})
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("Could not find predictions in the workflow response.")
    return predictions

def annotate_image(image, predictions):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for p in predictions:
        try:
            x, y = float(p["x"]), float(p["y"])
            w, h = float(p["width"]), float(p["height"])
            confidence = float(p.get("confidence", 0))
            label = str(p.get("class", "pepper spike"))
            x1, y1 = max(0, x-w/2), max(0, y-h/2)
            x2, y2 = min(image.width, x+w/2), min(image.height, y+h/2)
            draw.rectangle((x1,y1,x2,y2), outline="#00d66b", width=4)
            text = f"{label} {confidence:.0%}"
            bounds = draw.textbbox((x1, max(0,y1-22)), text, font=font)
            draw.rectangle(bounds, fill="#087443")
            draw.text((x1, max(0,y1-22)), text, fill="white", font=font)
        except (KeyError, TypeError, ValueError):
            continue
    return annotated

# -------------------- TOP NAVIGATION --------------------
header()
pages = ["Home", "Detect", "Results", "Growing Guide"]
icons = ["🏡", "📷", "📊", "🌱"]
nav_cols = st.columns(4)
for i, (page, icon) in enumerate(zip(pages, icons)):
    with nav_cols[i]:
        if st.button(f"{icon}\n{page}", key=f"nav_{page}", use_container_width=True):
            go(page)

st.markdown("---")

# -------------------- PAGE 1: HOME --------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero">
      <div class="logo">🌿</div>
      <div style="font-size:12px;letter-spacing:3px;color:#d9f7df;font-weight:700">SMART AGRICULTURE</div>
      <h1>Harvest Readiness AI</h1>
      <p>Smarter decisions for better black pepper harvests.</p>
      <div style="margin-top:18px">
        <span style="background:#ffffff20;padding:7px 12px;border-radius:25px;font-size:12px">
          ✨ Powered by your trained Roboflow model
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section("Welcome, pepper grower 👋",
            "Check pepper spikes with AI, review detection results, and learn post-harvest best practices.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div style="font-size:28px">📸</div><b>Quick detection</b><p class="subtle">Upload a plant image and detect spikes.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div style="font-size:28px">📚</div><b>Growing guide</b><p class="subtle">Learn harvesting and processing steps.</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="eyebrow">HOW IT WORKS</div>
      <p>① Upload a clear image</p>
      <p>② Run AI spike detection</p>
      <p>③ Review counts and confidence</p>
      <p style="margin-bottom:0">④ Use the growing guide for next steps</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start detecting pepper spikes →", type="primary", use_container_width=True):
        go("Detect")

    st.caption("AI predictions are estimates. Always verify the crop visually before making harvest decisions.")

# -------------------- PAGE 2: DETECT --------------------
elif st.session_state.page == "Detect":
    section("Inspect your pepper plant 📷",
            "Upload a clear image showing the pepper spikes. Keep the image focused and well lit.")

    with st.expander("🔐 Connect to your Roboflow model", expanded=not bool(st.session_state.api_key)):
        api_key = st.text_input(
            "Roboflow API key",
            type="password",
            value=st.session_state.api_key,
            help="Your key is used for this session. Do not share it publicly."
        )
        if api_key:
            st.session_state.api_key = api_key
        st.caption("Connected workflow: final-year-project-wacrp")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="pepper_uploader",
        help="JPG, PNG or WEBP"
    )

    if uploaded:
        image_bytes = uploaded.getvalue()
        st.session_state.uploaded_bytes = image_bytes
        st.session_state.uploaded_name = uploaded.name
        preview = Image.open(BytesIO(image_bytes)).convert("RGB")
        st.image(preview, caption=uploaded.name, width="stretch")

        st.markdown('<span class="pill">✓ Image ready</span>', unsafe_allow_html=True)

        if st.button("🔍 Detect Pepper Spikes", type="primary", use_container_width=True):
            if not st.session_state.api_key.strip():
                st.error("Enter your Roboflow API key above first.")
            else:
                try:
                    with st.spinner("🌱 Your AI model is analysing the image..."):
                        client = InferenceHTTPClient(
                            api_url="https://serverless.roboflow.com",
                            api_key=st.session_state.api_key.strip()
                        ).configure(
                            InferenceConfiguration(api_key_transport="header")
                        )
                        result = client.run_workflow(
                            workspace_name="chaya-shetty",
                            workflow_id="final-year-project-wacrp",
                            images={"image": preview},
                            use_cache=True
                        )
                        predictions = get_predictions(result)
                        annotated = annotate_image(preview, predictions)

                    counts = {}
                    for p in predictions:
                        label = str(p.get("class", "unknown"))
                        counts[label] = counts.get(label, 0) + 1

                    avg_conf = (
                        sum(float(p.get("confidence", 0)) for p in predictions) / len(predictions)
                        if predictions else 0
                    )

                    record = {
                        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                        "filename": uploaded.name,
                        "total": len(predictions),
                        "counts": counts,
                        "average_confidence": avg_conf,
                        "predictions": predictions,
                        "annotated_bytes": None
                    }
                    buffer = BytesIO()
                    annotated.save(buffer, format="PNG")
                    record["annotated_bytes"] = buffer.getvalue()

                    st.session_state.current_result = record
                    st.session_state.current_image = image_bytes
                    st.session_state.history.insert(0, record)
                    st.session_state.history = st.session_state.history[:20]
                    st.success("Detection completed!")
                    go("Results")

                except Exception as e:
                    st.error("Detection failed. Please check your internet connection and Roboflow workflow.")
                    st.code(str(e))

    st.markdown("---")
    section("Recent detections 🕘")
    if not st.session_state.history:
        st.info("Your detection history will appear here after your first successful scan.")
    else:
        for i, item in enumerate(st.session_state.history[:5]):
            with st.container(border=True):
                st.markdown(f"**{html.escape(item['filename'])}**")
                st.caption(item["time"])
                st.write(f"🌿 {item['total']} spike(s) detected")
                if st.button("View result", key=f"history_{i}", use_container_width=True):
                    st.session_state.current_result = item
                    go("Results")

# -------------------- PAGE 3: RESULTS --------------------
elif st.session_state.page == "Results":
    record = st.session_state.current_result
    if not record:
        st.info("No results yet. Upload an image and run detection first.")
        if st.button("Go to image upload", type="primary", use_container_width=True):
            go("Detect")
    else:
        section("Your detection report 📊", record["time"])
        st.markdown('<span class="pill">✓ Roboflow analysis complete</span>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric">
              <div class="number">{record['total']}</div>
              <div class="label">TOTAL SPIKES DETECTED</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric">
              <div class="number">{record['average_confidence']:.0%}</div>
              <div class="label">AVERAGE CONFIDENCE</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 🌱 Annotated image")
        if record.get("annotated_bytes"):
            st.image(record["annotated_bytes"], width="stretch")
            st.download_button(
                "⬇️ Download detection image",
                data=record["annotated_bytes"],
                file_name="pepper_detection_result.png",
                mime="image/png",
                use_container_width=True
            )

        st.markdown("#### 📍 Class-wise count")
        if record["counts"]:
            cols = st.columns(min(2, max(1, len(record["counts"]))))
            for i, (label, count) in enumerate(record["counts"].items()):
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div class="card">
                      <div style="font-size:13px;color:#6b7d72">{html.escape(label.title())}</div>
                      <div style="font-size:27px;font-weight:800;color:#087443">{count}</div>
                      <div class="subtle">spike(s)</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.warning("No spikes detected. Try a closer, clearer image with less blur.")

        with st.expander("See individual predictions"):
            if record["predictions"]:
                st.dataframe([
                    {
                        "Class": p.get("class", "unknown"),
                        "Confidence": f"{float(p.get('confidence', 0)):.1%}",
                        "Center X": round(float(p.get("x", 0)), 1),
                        "Center Y": round(float(p.get("y", 0)), 1)
                    } for p in record["predictions"]
                ], use_container_width=True, hide_index=True)
            else:
                st.write("No detections returned by the model.")

        st.info("Important: model confidence is not a guarantee of biological maturity. Verify the spikes in the field before harvesting.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📷 Scan another image", use_container_width=True):
                go("Detect")
        with c2:
            if st.button("🌱 Open growing guide", use_container_width=True):
                go("Growing Guide")

        report_json = {
            "time": record["time"],
            "filename": record["filename"],
            "total_spikes": record["total"],
            "class_counts": record["counts"],
            "average_confidence": record["average_confidence"],
            "predictions": record["predictions"]
        }
        st.download_button(
            "Download report as JSON",
            data=json.dumps(report_json, indent=2),
            file_name="pepper_harvest_report.json",
            mime="application/json",
            use_container_width=True
        )

# -------------------- PAGE 4: GROWING GUIDE --------------------
elif st.session_state.page == "Growing Guide":
    section("Black pepper growing guide 🌱",
            "A practical reference from plant care to post-harvest handling.")

    st.markdown("""
    <div class="hero" style="padding:25px 18px">
      <div style="font-size:36px">🌿</div>
      <h1 style="font-size:25px">From vine to quality pepper</h1>
      <p>Good field care and careful processing help protect quality.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("1. 🌳 Growing conditions", expanded=True):
        st.markdown("""
        - **Climate:** Warm, humid tropical conditions with well-distributed rainfall.
        - **Shade:** Provide suitable filtered shade and support trees or standards for vines.
        - **Soil:** Deep, well-drained soil rich in organic matter. Avoid waterlogging.
        - **Water:** Maintain moisture during dry periods, but do not let roots stand in water.
        - **Nutrition:** Use soil-test recommendations and locally approved agricultural advice.
        - **Pruning:** Remove dead or diseased growth and manage vines for airflow.
        """)

    with st.expander("2. 🔎 Checking spike maturity"):
        st.markdown("""
        - Inspect several spikes on the vine rather than relying on one spike.
        - Black pepper spikes usually begin changing colour as berries mature; some berries may turn yellow, orange or red depending on the stage and variety.
        - Consider the intended product and local harvesting practice.
        - The AI result is a visual aid, not a substitute for field inspection.
        - Avoid harvesting solely because a detection box has high confidence.
        """)

    with st.expander("3. 🧺 Harvesting"):
        st.markdown("""
        1. Harvest at the locally recommended maturity stage for the product you intend to make.
        2. Use clean hands or tools and clean collection containers.
        3. Avoid collecting berries from the ground or mixing them with soil and plant debris.
        4. Keep harvested spikes shaded and move them for processing promptly.
        5. Separate visibly damaged or mouldy material.
        """)

    with st.expander("4. ☀️ After harvest: processing and drying"):
        st.markdown("""
        **Step 1 — Threshing:** Separate berries from spikes using a clean, suitable method.

        **Step 2 — Cleaning:** Remove stems, leaves, soil and foreign material. Use potable water if washing is part of the recommended local process, then drain thoroughly.

        **Step 3 — Drying:** Spread berries in a thin layer on a clean, raised drying surface or use a suitable dryer. Protect them from rain, animals, dust and contamination. Turn regularly for even drying.

        **Step 4 — Check dryness:** Dry to the moisture level required by local standards and the intended market. A moisture meter is more reliable than judging by touch alone.

        **Step 5 — Cooling:** Let the dried pepper cool completely before packing. Packing warm product can cause condensation.

        **Step 6 — Sorting:** Remove damaged, discoloured or contaminated berries and grade according to buyer requirements.
        """)

    with st.expander("5. 📦 Storage and quality"):
        st.markdown("""
        - Store fully dried pepper in clean, food-grade, moisture-resistant packaging.
        - Keep bags off the floor and away from walls in a cool, dry, clean area.
        - Protect stock from humidity, insects, rodents and strong odours.
        - Never mix new harvest with old or damp stock.
        - Inspect stored lots regularly for moisture, mould or pest damage.
        - Follow applicable food-safety, pesticide-residue and market requirements.
        """)

    with st.expander("6. 🛡️ Plant health and safety"):
        st.markdown("""
        - Monitor vines for yellowing, wilting, root problems, leaf spots and unusual growth.
        - Maintain drainage and remove diseased material appropriately.
        - Avoid diagnosing a disease from one image alone.
        - Use only locally registered crop-protection products and follow the label.
        - Consult a local horticulture officer for serious or spreading symptoms.
        """)

    st.markdown("""
    <div class="card">
      <div class="eyebrow">QUALITY CHECKLIST</div>
      <p>✓ Correct harvest stage</p>
      <p>✓ Clean handling and separation</p>
      <p>✓ Uniform drying and moisture check</p>
      <p>✓ Complete cooling before packing</p>
      <p style="margin-bottom:0">✓ Dry, hygienic storage</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("This is general guidance. Follow local agricultural extension advice, food-safety standards and buyer specifications for your region.")

# -------------------- FOOTER --------------------
st.markdown("""
<div style="text-align:center;padding:25px 5px 5px;color:#829086;font-size:11px">
  🌿 PEPPER HARVEST AI · SMART FARMING ASSISTANT<br>
  Powered by your Roboflow model
</div>
""", unsafe_allow_html=True)
