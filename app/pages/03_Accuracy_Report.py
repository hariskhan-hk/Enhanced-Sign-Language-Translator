import streamlit as st
import pandas as pd
import numpy as np # Import numpy for checking None/NaN reliably

# --- Configuration ---
PAGE_TITLE = "Accuracy Report"
PAGE_ICON = "📊"

# --- Logo Paths (Update these to your actual file paths) ---
# If you don't have logos, set these to None or comment out the st.image lines
YOLONAS_LOGO_PATH = "images/yolonas_logo.png"  # Replace with your path or None
YOLOV8_LOGO_PATH = "images/yolov8_logo.png"    # Replace with your path or None


# --- Data from your console output ---
yolonas_metrics = {
    "Model": "YOLO-NAS (s)",
    "mAP@0.50": 0.8952391743659973,
    "mAP@0.50:0.95": None # Explicitly None if not available
}

yolov8_metrics = {
    "Model": "YOLOv8 (s)", # Assuming 's' model
    "mAP@0.50": 0.8785060485338948,
    "mAP@0.50:0.95": 0.721373872185325,
}

# --- Helper Function to display logo ---
def display_logo(logo_path, caption="", width=100):
    if logo_path:
        try:
            st.image(logo_path, caption=caption, width=width)
        except Exception as e:
            st.warning(f"Could not load logo: {logo_path}. Error: {e}")
    # else:
    #     st.markdown(f"*{caption} Logo Placeholder*") # Optional placeholder text

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} Model Accuracy Report")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True) # Reduce top padding
st.markdown("---")
st.write("Summary of key evaluation metrics (mAP) from testing YOLO-NAS and YOLOv8 on the validation dataset.")


# --- Individual Model Results ---
st.header("🚀 Evaluation Results")
st.markdown("---")

col1, col_spacer, col2 = st.columns([5, 1, 5]) # Add a spacer column

with col1:
    # display_logo(YOLONAS_LOGO_PATH, "YOLO-NAS Logo") # Display logo if available
    st.subheader(f"🔵 {yolonas_metrics['Model']}")
    st.metric(label="**mAP @ 0.50 IoU**", value=f"{yolonas_metrics['mAP@0.50']:.3f}")
    map_50_95_nas = yolonas_metrics['mAP@0.50:0.95']
    if map_50_95_nas is not None and not np.isnan(map_50_95_nas):
         st.metric(label="**mAP @ 0.50:0.95 IoU**", value=f"{map_50_95_nas:.3f}")
    else:
        st.info("ℹ️ mAP@0.50:0.95 score not available in test output.")
    st.markdown("*(Based on provided test run)*")


with col2:
    # display_logo(YOLOV8_LOGO_PATH, "YOLOv8 Logo") # Display logo if available
    st.subheader(f"🟢 {yolov8_metrics['Model']}")
    st.metric(label="**mAP @ 0.50 IoU**", value=f"{yolov8_metrics['mAP@0.50']:.3f}")
    map_50_95_v8 = yolov8_metrics['mAP@0.50:0.95']
    if map_50_95_v8 is not None and not np.isnan(map_50_95_v8):
        st.metric(label="**mAP @ 0.50:0.95 IoU**", value=f"{map_50_95_v8:.3f}")
    else:
         st.info("ℹ️ mAP@0.50:0.95 score not available in test output.") # Should not happen based on data, but good practice
    st.markdown("*(Based on provided test run)*")

st.markdown("<br>", unsafe_allow_html=True) # Add some vertical space


# --- Comparison Section ---
st.header("📈 Side-by-Side Comparison (mAP Scores)")
st.markdown("---")

# Prepare data for comparison table
comparison_data = [
    {
        "Model": yolonas_metrics["Model"],
        "mAP@0.50": yolonas_metrics["mAP@0.50"],
        "mAP@0.50:0.95": yolonas_metrics["mAP@0.50:0.95"],
     },
    {
        "Model": yolov8_metrics["Model"],
        "mAP@0.50": yolov8_metrics["mAP@0.50"],
        "mAP@0.50:0.95": yolov8_metrics["mAP@0.50:0.95"],
    }
]
df_comp = pd.DataFrame(comparison_data).set_index("Model")

# Display styled DataFrame
st.subheader("🔢 Numerical Comparison")
st.dataframe(
    df_comp.style.format("{:.3f}", na_rep="-")
                 .highlight_max(axis=0, props='color:white; background-color: #008080;') # Teal highlight for max
                 .highlight_null(color='#FFA07A') # Light salmon for missing - FIXED ARGUMENT
)

st.markdown("<br>", unsafe_allow_html=True)



# --- Metric Definitions (in an Expander) ---
with st.expander("📘 Metric Definitions"):
    st.markdown("""
    *   **mAP @ 0.50 IoU:**
        *   **What:** Mean Average Precision calculated using an **Intersection over Union (IoU)** threshold of **0.50**.
        *   **Meaning:** Measures how well the model detects objects, considering a detection correct if its bounding box overlaps the true object's box by at least 50%. It's a standard, relatively lenient metric.
        *   **Higher is better.**

    *   **mAP @ 0.50:0.95 IoU:**
        *   **What:** Mean Average Precision calculated by averaging the mAP scores across multiple IoU thresholds, starting from **0.50** up to **0.95** (typically in steps of 0.05). This is the primary COCO challenge metric.
        *   **Meaning:** Measures detection accuracy more rigorously, rewarding models that produce bounding boxes very close to the true object's location (high IoU overlap).
        *   **Higher is better.**
    """)


# --- Conclusions ---
st.header("💡 Conclusions & Considerations")
st.markdown("---")

# Use columns for better layout of conclusions
conc_col1, conc_col2 = st.columns(2)

with conc_col1:
    st.markdown(f"""
    #### Performance Insights:
    *   **Standard Accuracy (mAP@0.50):**
        *   **YOLO-NAS shows a slight edge** ({yolonas_metrics['mAP@0.50']:.3f}) in this test run compared to YOLOv8 ({yolov8_metrics['mAP@0.50']:.3f}). Both models perform well under this standard condition.
    *   **Localization Precision (mAP@0.50:0.95):**
        *   **YOLOv8 demonstrates superior performance** ({yolov8_metrics['mAP@0.50:0.95']:.3f}) when stricter bounding box accuracy is required.
        *   *(This metric was unavailable for YOLO-NAS in the provided output, preventing a direct comparison on this stricter benchmark based on the given data.)*

    #### Model Philosophy:
    *   **YOLO-NAS:** Often designed via Neural Architecture Search focusing on **hardware efficiency** and **quantization-friendliness**. Aims for a strong balance between accuracy and speed, especially post-quantization.
    *   **YOLOv8:** Represents an iteration focused on pushing **state-of-the-art accuracy** and usability, excelling in standard floating-point benchmarks like mAP@0.50:0.95.
    """)

with conc_col2:
    st.markdown(f"""
    #### Which Model is "Better"?
    The choice depends heavily on the specific application needs:

    *   **Choose YOLOv8 if:**
        *   Precise object localization (high IoU overlap) is critical.
        *   Maximizing accuracy on standard benchmarks (like COCO mAP) is the main goal.
        *   Deployment primarily uses floating-point precision.

    *   **Consider YOLO-NAS if:**
        *   The slightly higher mAP@0.50 observed here is decisive.
        *   Deployment involves **quantization** (e.g., for edge devices), where its design *might* yield better post-quantization results (requires verification).
        *   A balance between good accuracy and optimized inference speed (especially post-quantization) is needed.

    #### Other Factors:
    Remember that performance also depends on:
    *   Specific training dataset and augmentations.
    *   Training duration and hyperparameters.
    *   The exact model variant (e.g., -n, -s, -m, -l, -x).

    #### Final Thought:
    Based *solely* on this test data, **YOLOv8 shows better robustness for precise localization**, while YOLO-NAS achieved a slightly higher score on the more lenient mAP@0.50 metric. Test both in your specific environment, especially if considering quantization.
    """)

st.markdown("---")
st.caption(f"Report generated based on provided test logs.")