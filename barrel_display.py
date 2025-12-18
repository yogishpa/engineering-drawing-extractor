import streamlit as st
import pandas as pd

def display_barrel_results(result):
    """Display only barrel measurements in table format."""
    
    # Debug information
    st.write("**Debug Info:**")
    st.write(f"- Dimensions: {len(result.dimensions)}")
    st.write(f"- Part Numbers: {len(result.part_numbers)}")
    
    # Overall summary with method used
    method_name = result.extraction_method.replace('_', ' ').title()
    if 'bedrock_data_automation' in result.extraction_method.lower():
        method_display = "🤖 Bedrock Data Automation (BDA)"
    elif 'claude' in result.extraction_method.lower():
        method_display = "🧠 Claude 4.5 Sonnet"
    elif 'textract' in result.extraction_method.lower():
        method_display = "📄 Amazon Textract"
    else:
        method_display = f"🔧 {method_name}"
    
    st.success(f"✅ Extraction completed in {result.processing_time:.1f}s using {method_display}")
    
    # Show only barrel measurements table
    if result.dimensions or result.part_numbers:
        st.subheader("📏 Barrel Measurements")
        
        # Create tabular data for barrel dimensions
        barrel_data = {}
        part_number = "Unknown"
        
        # Extract part number if available
        if result.part_numbers:
            part_number = result.part_numbers[0].identifier
        
        # Organize dimensions by location (barrel type)
        for dim in result.dimensions:
            barrel_name = dim.location_description  # This now contains the barrel type name
            barrel_data[barrel_name] = f"{dim.value} {dim.unit}".strip()
        
        # Create DataFrame for display
        df_data = {
            'Part Number': [part_number],
            'Overall barrel length': [barrel_data.get('Overall barrel length', '')],
            'Barrel head length': [barrel_data.get('Barrel head length', '')],
            'Port to shoulder length': [barrel_data.get('Port to shoulder length', '')],
            'Barrel head Dia': [barrel_data.get('Barrel head Dia', '')],
            'Barrel shaft Dia': [barrel_data.get('Barrel shaft Dia', '')]
        }
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Show download options for barrel data
        st.subheader("📥 Download Barrel Data")
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"barrel_measurements_{part_number}.csv",
            mime="text/csv"
        )
    else:
        st.info("No barrel measurements found in the document.")
