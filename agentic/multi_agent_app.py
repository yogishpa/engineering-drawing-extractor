"""
Streamlit Interface for True Multi-Agent Barrel Dimension Extractor
"""

import streamlit as st
import os
import pdf2image
import pandas as pd
import logging
from multi_agent_extractor import TrueMultiAgentExtractor

# Configure logging for Streamlit app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streamlit_app.log')
    ]
)
logger = logging.getLogger(__name__)

def display_human_review_interface(result):
    """Display human review interface for extracted dimensions"""
    st.header("👤 Human Review & Confidence Calculation")
    st.markdown("Review each extracted field and mark as correct/incorrect:")
    
    # Use a stable key based on part number to maintain state across reruns
    stable_key = f"review_{result.part_number or 'unknown'}".replace(" ", "_").replace("-", "_")
    
    # Part Number Review
    if result.part_number:
        st.subheader("Part Number")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Extracted:** {result.part_number}")
        with col2:
            key = f"part_number_{stable_key}"
            if key not in st.session_state:
                st.session_state[key] = True
            st.checkbox("✓ Correct", key=key)
    
    # Dimension Reviews with separate checkboxes for dimensions and tolerances
    dimensions = [
        ("Overall Barrel Length", result.overall_barrel_length, "overall_length"),
        ("Barrel Head Length", result.barrel_head_length, "head_length"), 
        ("Port to Shoulder Length", result.port_to_shoulder_length, "port_length"),
        ("Barrel Head Diameter", result.barrel_head_diameter, "head_diameter"),
        ("Barrel Shaft Diameter", result.barrel_shaft_diameter, "shaft_diameter")
    ]
    
    for name, dim_data, key in dimensions:
        if dim_data and dim_data.value is not None:
            st.subheader(name)
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.write(f"**Value:** {dim_data.value:.2f} {dim_data.unit}")
            with col2:
                tolerance_text = dim_data.tolerance if dim_data.tolerance else "No tolerance"
                st.write(f"**Tolerance:** {tolerance_text}")
            with col3:
                dim_checkbox_key = f"{key}_dim_{stable_key}"
                if dim_checkbox_key not in st.session_state:
                    st.session_state[dim_checkbox_key] = True
                st.checkbox("✓ Dimension", key=dim_checkbox_key)
            with col4:
                tol_checkbox_key = f"{key}_tol_{stable_key}"
                if tol_checkbox_key not in st.session_state:
                    st.session_state[tol_checkbox_key] = True
                st.checkbox("✓ Tolerance", key=tol_checkbox_key)
    
    # Calculate and display confidence percentage
    total_fields = 0
    correct_fields = 0
    
    # Count part number if exists
    if result.part_number:
        total_fields += 1
        if st.session_state.get(f"part_number_{stable_key}", True):
            correct_fields += 1
    
    # Count dimensions and tolerances
    for _, dim_data, key in dimensions:
        if dim_data and dim_data.value is not None:
            total_fields += 2  # dimension + tolerance
            if st.session_state.get(f"{key}_dim_{stable_key}", True):
                correct_fields += 1
            if st.session_state.get(f"{key}_tol_{stable_key}", True):
                correct_fields += 1
    
    if total_fields > 0:
        confidence_percentage = (correct_fields / total_fields) * 100
        st.markdown("---")
        st.subheader("📊 Confidence Score")
        
        # Use columns to prevent layout shift
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Accuracy", f"{confidence_percentage:.1f}%")
        with col2:
            st.metric("Fields Correct", f"{correct_fields}/{total_fields}")
    
    return {
        'total_fields': total_fields,
        'correct_fields': correct_fields,
        'confidence_percentage': confidence_percentage if total_fields > 0 else 0
    }


def _format_dimension(dim_data):
    """Format dimension data for display"""
    if dim_data and hasattr(dim_data, 'value') and dim_data.value is not None:
        value = f"{dim_data.value:.2f} {dim_data.unit}"
        tolerance = dim_data.tolerance or "N/A"
        confidence = f"{dim_data.confidence:.1%}" if dim_data.confidence else "N/A"
        source_agent = dim_data.source_agent or "Unknown"
        return (value, tolerance, confidence, source_agent)
    return ("Not Found", "N/A", "N/A", "N/A")

def main():
    st.set_page_config(
        page_title="True Multi-Agent Barrel Extractor", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 True Multi-Agent Barrel Dimension Extractor")
    st.markdown("**Powered by Autonomous AI Agents with Dynamic Collaboration**")
    
    # Sidebar - Agent Status
    with st.sidebar:
        st.header("🎯 Agent Status")
        
        # Initialize extractor to show agent info
        try:
            logger.info("Initializing TrueMultiAgentExtractor for sidebar display")
            extractor = TrueMultiAgentExtractor()
            logger.info("Extractor initialized successfully")
            
            st.subheader("Active Agents")
            agents_info = {
                "👁️ Vision Analyst": "Computer vision & geometric analysis",
                "📝 OCR Specialist": "Text extraction & dimension reading", 
                "🎛️ Coordinator": "Workflow orchestration & final decisions"
            }
            
            for agent_name, description in agents_info.items():
                with st.expander(agent_name):
                    st.write(description)
                    st.write("Status: ✅ Ready")
            
            st.subheader("Collaboration Features")
            st.write("✅ Autonomous decision making")
            st.write("✅ Dynamic agent communication")
            st.write("✅ Consensus building")
            st.write("✅ Conflict resolution")
            st.write("✅ Quality assurance")
            
        except Exception as e:
            logger.error(f"Agent initialization error: {e}")
            st.error(f"Agent initialization error: {e}")
            st.stop()
    
    # Main interface
    st.header("📄 Upload Engineering Drawing")
    uploaded_files = st.file_uploader(
        "Upload PDF files for multi-agent analysis", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="The agents will collaborate to extract precise dimensions"
    )
    
    if uploaded_files:
        st.header("🤖 Multi-Agent Analysis")
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            collaboration_mode = st.selectbox(
                "Collaboration Mode",
                ["Full Collaboration", "Parallel Analysis", "Sequential Review"],
                help="How agents should work together"
            )
        
        with col2:
            quality_threshold = st.slider(
                "Quality Threshold", 
                0.5, 1.0, 0.8, 0.05,
                help="Minimum confidence required for accepting results"
            )
        
        if st.button("🚀 Start Multi-Agent Analysis", type="primary"):
            
            for uploaded_file in uploaded_files:
                st.subheader(f"📄 Analyzing: {uploaded_file.name}")
                
                # Simple progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process file
                temp_pdf_path = None
                temp_image_path = None
                
                try:
                    logger.info(f"Processing uploaded file: {uploaded_file.name}")
                    # Save and convert PDF
                    temp_pdf_path = f"temp_{uploaded_file.name}"
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    logger.info(f"Saved PDF to: {temp_pdf_path}")
                    
                    # Convert to image
                    logger.info("Converting PDF to image...")
                    images = pdf2image.convert_from_path(temp_pdf_path, dpi=300)
                    temp_image_path = temp_pdf_path.replace('.pdf', '_page1.png')
                    images[0].save(temp_image_path, 'PNG')
                    logger.info(f"Converted to image: {temp_image_path}")
                    
                    # Initialize progress
                    progress_bar.progress(0.3)
                    status_text.info("🔍 Analyzing drawing...")
                    
                    # Run multi-agent analysis with simplified callbacks
                    logger.info("Starting multi-agent analysis...")
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.info(message)
                    
                    status_callbacks = {
                        'vision': lambda msg: update_progress(0.5, "👁️ Vision analysis..."),
                        'ocr': lambda msg: update_progress(0.7, "📝 Text extraction..."),
                        'coordinator': lambda msg: update_progress(0.9, "🎛️ Building consensus...")
                    }
                    
                    extractor = TrueMultiAgentExtractor(status_callbacks=status_callbacks)
                    
                    with st.spinner("🤖 Agents collaborating..."):
                        result = extractor.extract_dimensions(temp_image_path)
                    logger.info("Multi-agent analysis completed")
                    
                    # Store result in session state
                    st.session_state.extraction_result = result
                    
                    # Update final status
                    progress_bar.progress(1.0)
                    status_text.success("✅ Analysis complete!")
                    
                    # Display results
                    st.subheader("📊 Multi-Agent Results")
                    
                    # Results summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Overall Confidence", f"{result.confidence_score:.1%}")
                    with col2:
                        st.metric("Consensus Reached", "✅" if result.consensus_reached else "❌")
                    with col3:
                        st.metric("Agent Interactions", len(result.agent_collaboration_log) if result.agent_collaboration_log else 0)
                    with col4:
                        st.metric("Quality Score", f"{result.confidence_score:.2f}")
                    
                    # Detailed results table
                    if result.confidence_score > 0:
                        results_data = []
                        
                        dimensions = {
                            "Part Number": (result.part_number, "N/A", "N/A", "N/A"),
                            "Overall Barrel Length": _format_dimension(result.overall_barrel_length),
                            "Barrel Head Length": _format_dimension(result.barrel_head_length),
                            "Port to Shoulder Length": _format_dimension(result.port_to_shoulder_length),
                            "Barrel Head Diameter": _format_dimension(result.barrel_head_diameter),
                            "Barrel Shaft Diameter": _format_dimension(result.barrel_shaft_diameter)
                        }
                        
                        for dim_name, (value, tolerance, confidence, source_agent) in dimensions.items():
                            results_data.append({
                                "Dimension": dim_name,
                                "Value": value,
                                "Tolerance": tolerance,
                                "Confidence": confidence,
                                "Source Agent": source_agent
                            })
                        
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Human Review Interface
                        st.markdown("---")
                        review_data = display_human_review_interface(result)
                        st.markdown("---")
                        
                        # Agent collaboration log
                        if result.agent_collaboration_log:
                            with st.expander("🤖 Agent Collaboration Log"):
                                for log_entry in result.agent_collaboration_log:
                                    st.text(log_entry)
                    
                    else:
                        st.error("❌ Multi-agent analysis failed to extract dimensions")
                
                except Exception as e:
                    logger.error(f"Processing error for {uploaded_file.name}: {e}")
                    st.error(f"❌ Processing error: {e}")
                
                finally:
                    # Cleanup
                    logger.info("Cleaning up temporary files...")
                    try:
                        if temp_pdf_path and os.path.exists(temp_pdf_path):
                            os.remove(temp_pdf_path)
                            logger.info(f"Removed: {temp_pdf_path}")
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            logger.info(f"Removed: {temp_image_path}")
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup error: {cleanup_error}")
                        pass
    
    # Analytics section
    if st.button("📈 View Multi-Agent Analytics"):
        st.header("📈 Multi-Agent Performance Analytics")
        
        try:
            # Load CSV data
            if os.path.exists("multi_agent_results.csv"):
                df = pd.read_csv("multi_agent_results.csv")
                
                if not df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Agent Performance")
                        # Agent attribution analysis
                        agent_cols = [col for col in df.columns if col.endswith('_source_agent')]
                        agent_stats = {}
                        
                        for col in agent_cols:
                            agents = df[col].value_counts()
                            for agent, count in agents.items():
                                if agent and agent != '':
                                    agent_stats[agent] = agent_stats.get(agent, 0) + count
                        
                        if agent_stats:
                            st.bar_chart(pd.Series(agent_stats))
                    
                    with col2:
                        st.subheader("Collaboration Metrics")
                        st.metric("Total Extractions", len(df))
                        st.metric("Average Confidence", f"{df['overall_confidence'].mean():.1%}")
                        st.metric("Consensus Rate", f"{df['consensus_reached'].sum() / len(df):.1%}")
                        st.metric("Avg Collaboration Count", f"{df['agent_collaboration_count'].mean():.1f}")
                    
                    # Recent results
                    st.subheader("Recent Results")
                    st.dataframe(df.tail(10), use_container_width=True)
                
                else:
                    st.info("No analysis data available yet")
            else:
                st.info("No analysis data available yet")
                
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            st.error(f"Analytics error: {e}")

if __name__ == "__main__":
    main()
