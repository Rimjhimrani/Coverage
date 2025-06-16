import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
from collections import Counter
import warnings

# Suppress warnings - Fixed the NumPy warning issue
warnings.filterwarnings('ignore')
# Remove the problematic np.RankWarning line as it doesn't exist in newer NumPy versions

# Page configuration
st.set_page_config(
    page_title="Inventory Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-excess {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .status-normal {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .status-short {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .coverage-critical {
        color: #f44336;
        font-weight: bold;
    }
    .coverage-warning {
        color: #ff9800;
        font-weight: bold;
    }
    .coverage-good {
        color: #4caf50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class InventoryManagementSystem:
    def __init__(self):
        self.coverage_categories = {
            'Less Than 3 Days': '#ff4444',      # Red
            'In Between 3-7 Days': '#ff8800',   # Orange
            'In Between 7-30 Days': '#ffaa00',  # Yellow-Orange
            'In Between 30-90 Days': '#88cc00', # Light Green
            'In Between 90-180 Days': '#00cc44', # Green
            'More Than 6 Month': '#0088cc'      # Blue
        }
        
        # Initialize session state
        if 'inventory_data' not in st.session_state:
            st.session_state.inventory_data = []
            self.load_sample_data()
        
        if 'tolerance' not in st.session_state:
            st.session_state.tolerance = 30
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        sample_data = [
            {
                'Part No': '123/45376',
                'Part Description': 'Tyre A',
                'Part Classification': 'A',
                'Inventory Class': 'A1',
                'Average Consumption/Day': 32,
                'RM Norms - In Days': 3,
                'RM Norms - In Qty': 96,
                'Current Stock Qty': 2020,
            },
            {
                'Part No': '124/45377',
                'Part Description': 'Brake Pad B',
                'Part Classification': 'A',
                'Inventory Class': 'A1',
                'Average Consumption/Day': 25,
                'RM Norms - In Days': 7,
                'RM Norms - In Qty': 175,
                'Current Stock Qty': 50,
            },
            {
                'Part No': '125/45378',
                'Part Description': 'Oil Filter C',
                'Part Classification': 'B',
                'Inventory Class': 'B1',
                'Average Consumption/Day': 15,
                'RM Norms - In Days': 14,
                'RM Norms - In Qty': 210,
                'Current Stock Qty': 80,
            },
            {
                'Part No': '126/45379',
                'Part Description': 'Air Filter D',
                'Part Classification': 'B',
                'Inventory Class': 'B2',
                'Average Consumption/Day': 10,
                'RM Norms - In Days': 21,
                'RM Norms - In Qty': 210,
                'Current Stock Qty': 150,
            },
            {
                'Part No': '127/45380',
                'Part Description': 'Spark Plug E',
                'Part Classification': 'C',
                'Inventory Class': 'C1',
                'Average Consumption/Day': 5,
                'RM Norms - In Days': 30,
                'RM Norms - In Qty': 150,
                'Current Stock Qty': 400,
            },
            {
                'Part No': '128/45381',
                'Part Description': 'Battery F',
                'Part Classification': 'A',
                'Inventory Class': 'A2',
                'Average Consumption/Day': 3,
                'RM Norms - In Days': 45,
                'RM Norms - In Qty': 135,
                'Current Stock Qty': 600,
            },
            {
                'Part No': '129/45382',
                'Part Description': 'Engine Oil G',
                'Part Classification': 'B',
                'Inventory Class': 'B1',
                'Average Consumption/Day': 8,
                'RM Norms - In Days': 20,
                'RM Norms - In Qty': 160,
                'Current Stock Qty': 240,
            },
            {
                'Part No': '130/45383',
                'Part Description': 'Coolant H',
                'Part Classification': 'C',
                'Inventory Class': 'C2',
                'Average Consumption/Day': 2,
                'RM Norms - In Days': 60,
                'RM Norms - In Qty': 120,
                'Current Stock Qty': 280,
            }
        ]
        
        self.process_imported_data(sample_data)
    
    def process_imported_data(self, raw_data):
        """Process imported data and calculate metrics"""
        inventory_data = []
        
        for row in raw_data:
            try:
                # Handle different possible column names
                part_no = self.get_column_value(row, ['Part No', 'Part Number', 'PartNo'])
                description = self.get_column_value(row, ['Part Description', 'Description', 'Desc'])
                classification = self.get_column_value(row, ['Part Classification', 'Classification', 'Class'])
                inv_class = self.get_column_value(row, ['Inventory Class', 'Inv Class', 'InvClass'])
                avg_consumption = float(self.get_column_value(row, ['Average Consumption/Day', 'Avg Consumption', 'Daily Consumption']) or 0)
                rm_norms_days = float(self.get_column_value(row, ['RM Norms - In Days', 'RM Days', 'Norms Days']) or 0)
                rm_norms_qty = float(self.get_column_value(row, ['RM Norms - In Qty', 'RM Qty', 'Norms Qty']) or 0)
                current_stock = float(self.get_column_value(row, ['Current Stock Qty', 'Current Stock', 'Stock']) or 0)
                
                # Calculate revised quantity based on tolerance
                revised_qty = rm_norms_qty * (1 + st.session_state.tolerance / 100)
                
                # Calculate stock coverage
                coverage_days = current_stock / avg_consumption if avg_consumption > 0 else 999
                coverage_category = self.determine_coverage_category(coverage_days)
                
                # Calculate variance and status
                variance_pct, variance_qty, status = self.calculate_inventory_status(current_stock, rm_norms_qty)
                
                processed_row = {
                    'Part No': str(part_no) if part_no else '',
                    'Description': str(description) if description else '',
                    'Classification': str(classification) if classification else '',
                    'Inventory Class': str(inv_class) if inv_class else '',
                    'Avg Consumption': avg_consumption,
                    'RM Norms Days': rm_norms_days,
                    'RM Norms Qty': rm_norms_qty,
                    'Revised Qty': round(revised_qty, 2),
                    'Current Stock': current_stock,
                    'Variance %': round(variance_pct, 2),
                    'Variance Qty': round(variance_qty, 2),
                    'Status': status,
                    'Coverage Days': round(coverage_days, 2),
                    'Coverage Category': coverage_category
                }
                
                inventory_data.append(processed_row)
                
            except Exception as e:
                st.error(f"Error processing row: {e}")
                continue
        
        st.session_state.inventory_data = inventory_data
    
    def get_column_value(self, row, possible_names):
        """Get value from row using possible column names"""
        for name in possible_names:
            if name in row:
                return row[name]
        return None
    
    def determine_coverage_category(self, days):
        """Determine coverage category based on days"""
        if days < 3:
            return 'Less Than 3 Days'
        elif days < 7:
            return 'In Between 3-7 Days'
        elif days < 30:
            return 'In Between 7-30 Days'
        elif days < 90:
            return 'In Between 30-90 Days'
        elif days < 180:
            return 'In Between 90-180 Days'
        else:
            return 'More Than 6 Month'
    
    def calculate_inventory_status(self, current_qty, rm_qty):
        """Calculate variance and determine inventory status"""
        if rm_qty == 0:
            variance_pct = 0 if current_qty == 0 else 100
            variance_qty = current_qty
        else:
            variance_pct = ((current_qty - rm_qty) / rm_qty) * 100
            variance_qty = current_qty - rm_qty
        
        # Determine status based on tolerance
        upper_limit = st.session_state.tolerance
        lower_limit = -st.session_state.tolerance
        
        if variance_pct > upper_limit:
            status = "Excess Inventory"
        elif variance_pct < lower_limit:
            status = "Short Inventory"
        else:
            status = "Within Norms"
        
        return variance_pct, variance_qty, status
    
    def apply_filters(self, df, search_term, category_filter, class_filter, status_filter, critical_only):
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Search filter
        if search_term:
            search_mask = (
                filtered_df['Part No'].str.contains(search_term, case=False, na=False) |
                filtered_df['Description'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # Category filter
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['Coverage Category'] == category_filter]
        
        # Classification filter
        if class_filter != "All":
            filtered_df = filtered_df[filtered_df['Classification'] == class_filter]
        
        # Status filter
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['Status'] == status_filter]
        
        # Critical stock filter
        if critical_only:
            filtered_df = filtered_df[filtered_df['Coverage Days'] < 7]
        
        return filtered_df
    
    def create_coverage_dashboard(self, df):
        """Create coverage dashboard with multiple charts"""
        if df.empty:
            st.warning("No data available for dashboard!")
            return
        
        # Coverage Categories Pie Chart
        coverage_counts = df['Coverage Category'].value_counts()
        fig_coverage = px.pie(
            values=coverage_counts.values,
            names=coverage_counts.index,
            title="Coverage Categories Distribution",
            color_discrete_map=self.coverage_categories
        )
        fig_coverage.update_traces(textposition='inside', textinfo='percent+label')
        
        # Status Distribution Bar Chart
        status_counts = df['Status'].value_counts()
        status_colors = {'Excess Inventory': '#ff4444', 'Within Norms': '#44aa44', 'Short Inventory': '#ffaa00'}
        fig_status = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Inventory Status Distribution",
            color=status_counts.index,
            color_discrete_map=status_colors
        )
        fig_status.update_layout(showlegend=False, xaxis_title="Status", yaxis_title="Number of Items")
        
        # Classification Breakdown
        class_counts = df['Classification'].value_counts()
        fig_class = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title="Classification Breakdown",
            color=class_counts.index,
            color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
        )
        fig_class.update_layout(showlegend=False, xaxis_title="Classification", yaxis_title="Number of Items")
        
        # Coverage Days Histogram
        fig_histogram = px.histogram(
            df[df['Coverage Days'] < 365],
            x='Coverage Days',
            nbins=20,
            title="Coverage Days Distribution",
            color_discrete_sequence=['skyblue']
        )
        fig_histogram.add_vline(x=7, line_dash="dash", line_color="red", 
                               annotation_text="Critical Level (7 days)")
        fig_histogram.update_layout(xaxis_title="Coverage Days", yaxis_title="Frequency")
        
        return fig_coverage, fig_status, fig_class, fig_histogram
    
    def create_stock_analysis_charts(self, df):
        """Create stock analysis charts"""
        if df.empty:
            st.warning("No data available for analysis!")
            return None, None
        
        # Stock vs Norms Scatter Plot
        fig_scatter = px.scatter(
            df,
            x='RM Norms Qty',
            y='Current Stock',
            color='Status',
            size='Avg Consumption',
            hover_data=['Part No', 'Description', 'Coverage Days'],
            title="Current Stock vs RM Norms",
            color_discrete_map={'Excess Inventory': '#ff4444', 'Within Norms': '#44aa44', 'Short Inventory': '#ffaa00'}
        )
        # Add diagonal line for reference
        max_val = max(df['RM Norms Qty'].max(), df['Current Stock'].max())
        fig_scatter.add_shape(
            type="line",
            x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="gray", width=2, dash="dash"),
        )
        
        # Variance Analysis
        fig_variance = px.bar(
            df.head(20),  # Top 20 items
            x='Part No',
            y='Variance %',
            color='Status',
            title="Variance Analysis (Top 20 Items)",
            color_discrete_map={'Excess Inventory': '#ff4444', 'Within Norms': '#44aa44', 'Short Inventory': '#ffaa00'}
        )
        fig_variance.update_layout(xaxis_tickangle=-45)
        
        return fig_scatter, fig_variance
    
    def create_consumption_trends_chart(self, df):
        """Create consumption trends visualization"""
        if df.empty:
            st.warning("No data available for trends!")
            return None
        
        # Group by classification and calculate metrics
        trends_data = df.groupby('Classification').agg({
            'Avg Consumption': ['mean', 'sum'],
            'Current Stock': ['mean', 'sum'],
            'Coverage Days': 'mean'
        }).round(2)
        
        trends_data.columns = ['Avg Daily Consumption', 'Total Daily Consumption', 
                              'Avg Current Stock', 'Total Current Stock', 'Avg Coverage Days']
        trends_data = trends_data.reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Daily Consumption by Class', 'Total Stock by Class',
                           'Average Coverage Days by Class', 'Consumption vs Stock Ratio'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=trends_data['Classification'], y=trends_data['Avg Daily Consumption'],
                   name='Avg Daily Consumption', marker_color='#ff6b6b'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=trends_data['Classification'], y=trends_data['Total Current Stock'],
                   name='Total Stock', marker_color='#4ecdc4'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=trends_data['Classification'], y=trends_data['Avg Coverage Days'],
                   name='Avg Coverage Days', marker_color='#45b7d1'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=trends_data['Total Daily Consumption'], y=trends_data['Total Current Stock'],
                      mode='markers+text', text=trends_data['Classification'],
                      textposition='top center', name='Consumption vs Stock',
                      marker=dict(size=15, color='#ff9800')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Consumption Trends Analysis")
        
        return fig
    
    def export_to_excel(self, df):
        """Export dataframe to Excel"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main inventory sheet
            df.to_excel(writer, sheet_name='Inventory Summary', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': [
                    'Total Items',
                    'Critical Items (< 3 days)',
                    'Excess Inventory',
                    'Short Inventory',
                    'Within Norms'
                ],
                'Value': [
                    len(df),
                    len(df[df['Coverage Days'] < 3]),
                    len(df[df['Status'] == 'Excess Inventory']),
                    len(df[df['Status'] == 'Short Inventory']),
                    len(df[df['Status'] == 'Within Norms'])
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Critical items
            critical_df = df[df['Coverage Days'] < 7]
            if not critical_df.empty:
                critical_df.to_excel(writer, sheet_name='Critical Items', index=False)
            
            # Excess inventory
            excess_df = df[df['Status'] == 'Excess Inventory']
            if not excess_df.empty:
                excess_df.to_excel(writer, sheet_name='Excess Inventory', index=False)
            
            # Short inventory
            short_df = df[df['Status'] == 'Short Inventory']
            if not short_df.empty:
                short_df.to_excel(writer, sheet_name='Short Inventory', index=False)
        
        return output.getvalue()

def main():
    # Initialize the system
    inv_system = InventoryManagementSystem()
    
    # Header
    st.title("📦 Inventory Coverage Meter")
    
    st.markdown(
        "<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>"
        "Designed and Developed by Agilomatrix</p>",
        unsafe_allow_html=True
    )
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Tolerance Zone
        st.subheader("Tolerance Zone")
        tolerance_options = [10, 20, 30, 40, 50]
        tolerance = st.selectbox(
            "Tolerance (+/-) %:",
            tolerance_options,
            index=tolerance_options.index(st.session_state.tolerance)
        )
        
        if tolerance != st.session_state.tolerance:
            st.session_state.tolerance = tolerance
            # Recalculate data with new tolerance
            inv_system.process_imported_data([
                {
                    'Part No': item['Part No'],
                    'Part Description': item['Description'],
                    'Part Classification': item['Classification'],
                    'Inventory Class': item['Inventory Class'],
                    'Average Consumption/Day': item['Avg Consumption'],
                    'RM Norms - In Days': item['RM Norms Days'],
                    'RM Norms - In Qty': item['RM Norms Qty'],
                    'Current Stock Qty': item['Current Stock']
                }
                for item in st.session_state.inventory_data
            ])
            st.rerun()
        
        st.markdown("---")
        
        # File Operations
        st.subheader("📁 File Operations")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Import Excel File",
            type=['xlsx', 'xls'],
            help="Upload an Excel file with inventory data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                raw_data = df.to_dict('records')
                inv_system.process_imported_data(raw_data)
                st.success(f"Successfully imported {len(raw_data)} records!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing file: {str(e)}")
        
        # Add new part
        if st.button("➕ Add New Part", use_container_width=True):
            st.session_state.show_add_form = True
        
        st.markdown("---")
        
        # Filters
        st.subheader("🔍 Filters & Search")
        
        # Search
        search_term = st.text_input("Search:", placeholder="Enter part number or description")
        
        # Category filter
        category_options = ["All"] + list(inv_system.coverage_categories.keys())
        category_filter = st.selectbox("Coverage Category:", category_options)
        
        # Classification filter
        class_filter = st.selectbox("Classification:", ["All", "A", "B", "C"])
        
        # Status filter
        status_filter = st.selectbox(
            "Inventory Status:",
            ["All", "Excess Inventory", "Within Norms", "Short Inventory"]
        )
        
        # Critical stock only
        critical_only = st.checkbox("Critical Stock Only (< 7 days)")
        
        st.markdown("---")
        
        # Status Legend
        st.subheader("📊 Status Legend")
        st.markdown("""
        🔴 **Excess Inventory** - Above tolerance  
        🟢 **Within Norms** - Within tolerance  
        🟠 **Short Inventory** - Below tolerance  
        """)
    
    # Main content area
    if st.session_state.inventory_data:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.inventory_data)
        
        # Apply filters
        filtered_df = inv_system.apply_filters(
            df, search_term, category_filter, class_filter, status_filter, critical_only
        )
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Items", len(filtered_df))
        
        with col2:
            critical_count = len(filtered_df[filtered_df['Coverage Days'] < 3])
            st.metric("Critical Items", critical_count, delta=f"{critical_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%")
        
        with col3:
            excess_count = len(filtered_df[filtered_df['Status'] == 'Excess Inventory'])
            st.metric("Excess Items", excess_count)
        
        with col4:
            short_count = len(filtered_df[filtered_df['Status'] == 'Short Inventory'])
            st.metric("Short Items", short_count)
        
        with col5:
            normal_count = len(filtered_df[filtered_df['Status'] == 'Within Norms'])
            st.metric("Within Norms", normal_count)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Inventory Data", 
            "📊 Coverage Dashboard", 
            "📈 Stock Analysis", 
            "🔍 Critical Items", 
            "📑 Reports"
        ])
        
        with tab1:
            st.subheader("Inventory Data")
            
            # Display controls
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Showing {len(filtered_df)} of {len(df)} items")
            with col2:
                # Export button
                if st.button("📥 Export to Excel"):
                    excel_data = inv_system.export_to_excel(filtered_df)
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Display dataframe
            if not filtered_df.empty:
                # Format numeric columns
                display_df = filtered_df.copy()
                display_df['Variance %'] = display_df['Variance %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Coverage Days": st.column_config.NumberColumn(
                            "Coverage Days",
                            help="Days of stock coverage",
                            format="%.1f",
                        ),
                        "Status": st.column_config.TextColumn(
                            "Status",
                            help="Inventory status based on tolerance"
                        )
                    }
                )
            else:
                st.warning("No data matches the current filters.")
        
        with tab2:
            st.subheader("Coverage Dashboard")
            
            if not filtered_df.empty:
                # Create dashboard charts
                fig_coverage, fig_status, fig_class, fig_histogram = inv_system.create_coverage_dashboard(filtered_df)
                
                # Display charts in grid
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_coverage, use_container_width=True)
                    st.plotly_chart(fig_class, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig_status, use_container_width=True)
                    st.plotly_chart(fig_histogram, use_container_width=True)
            else:
                st.warning("No data available for dashboard with current filters.")
        
        with tab3:
            st.subheader("Stock Analysis")
            
            if not filtered_df.empty:
                # Create analysis charts
                fig_scatter, fig_variance = inv_system.create_stock_analysis_charts(filtered_df)
                
                if fig_scatter and fig_variance:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.plotly_chart(fig_variance, use_container_width=True)
                
                # Consumption trends
                st.subheader("Consumption Trends")
                fig_trends = inv_system.create_consumption_trends_chart(filtered_df)
                if fig_trends:
                    st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.warning("No data available for analysis with current filters.")
        
        with tab4:
            st.subheader("Critical Items Analysis")
            
            # Critical items (< 7 days coverage)
            critical_df = filtered_df[filtered_df['Coverage Days'] < 7]
            
            if not critical_df.empty:
                st.error(f"⚠️ {len(critical_df)} items have critical stock levels (< 7 days coverage)")
                
                # Sort by coverage days
                critical_df_sorted = critical_df.sort_values('Coverage Days')
                
                # Display critical items
                st.dataframe(
                    critical_df_sorted[['Part No', 'Description', 'Classification', 
                                      'Current Stock', 'Avg Consumption', 'Coverage Days', 'Status']],
                    use_container_width=True,
                    height=300
                )
                
                # Critical items chart
                fig_critical = px.bar(
                    critical_df_sorted.head(10),
                    x='Part No',
                    y='Coverage Days',
                    color='Coverage Days',
                    title="Top 10 Most Critical Items",
                    color_continuous_scale='Reds_r'
                )
                fig_critical.update_layout(xaxis_tickangle=-45)
                fig_critical.add_hline(y=3, line_dash="dash", line_color="red", 
                                     annotation_text="Critical Threshold (3 days)")
                st.plotly_chart(fig_critical, use_container_width=True)
                
                # Action recommendations
                st.subheader("📋 Recommended Actions")
                for _, item in critical_df_sorted.head(5).iterrows():
                    days_left = item['Coverage Days']
                    if days_left < 1:
                        urgency = "🚨 URGENT"
                        color = "red"
                    elif days_left < 3:
                        urgency = "⚠️ HIGH PRIORITY"
                        color = "orange"
                    else:
                        urgency = "⚡ MEDIUM PRIORITY"
                        color = "yellow"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}20; border-left: 4px solid {color}; padding: 10px; margin: 5px 0; border-radius: 5px;">
                        <strong>{urgency}</strong><br>
                        <strong>Part:</strong> {item['Part No']} - {item['Description']}<br>
                        <strong>Days Left:</strong> {days_left:.1f} days<br>
                        <strong>Recommended Action:</strong> Order immediately - {int(item['Avg Consumption'] * item['RM Norms Days'])} units needed
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical stock items found!")
        
        with tab5:
            st.subheader("📑 Reports & Analytics")
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Summary Statistics")
                
                if not filtered_df.empty:
                    total_value = filtered_df['Current Stock'].sum()
                    avg_coverage = filtered_df['Coverage Days'].mean()
                    
                    summary_stats = {
                        'Total Items': len(filtered_df),
                        'Total Stock Quantity': f"{total_value:,.0f}",
                        'Average Coverage Days': f"{avg_coverage:.1f}",
                        'Critical Items (< 3 days)': len(filtered_df[filtered_df['Coverage Days'] < 3]),
                        'Warning Items (3-7 days)': len(filtered_df[(filtered_df['Coverage Days'] >= 3) & (filtered_df['Coverage Days'] < 7)]),
                        'Excess Inventory Items': len(filtered_df[filtered_df['Status'] == 'Excess Inventory']),
                        'Short Inventory Items': len(filtered_df[filtered_df['Status'] == 'Short Inventory']),
                        'Items Within Norms': len(filtered_df[filtered_df['Status'] == 'Within Norms'])
                    }
                    
                    for key, value in summary_stats.items():
                        st.metric(key, value)
            
            with col2:
                st.subheader("🎯 Key Performance Indicators")
                
                if not filtered_df.empty:
                    # Calculate KPIs
                    total_items = len(filtered_df)
                    critical_pct = (len(filtered_df[filtered_df['Coverage Days'] < 3]) / total_items * 100) if total_items > 0 else 0
                    excess_pct = (len(filtered_df[filtered_df['Status'] == 'Excess Inventory']) / total_items * 100) if total_items > 0 else 0
                    within_norms_pct = (len(filtered_df[filtered_df['Status'] == 'Within Norms']) / total_items * 100) if total_items > 0 else 0
                    
                    # Stock efficiency
                    avg_variance = abs(filtered_df['Variance %']).mean()
                    
                    kpis = {
                        'Inventory Accuracy': f"{within_norms_pct:.1f}%",
                        'Critical Stock Rate': f"{critical_pct:.1f}%",
                        'Excess Stock Rate': f"{excess_pct:.1f}%",
                        'Average Variance': f"{avg_variance:.1f}%",
                        'Stock Efficiency Score': f"{max(0, 100 - avg_variance):.0f}/100"
                    }
                    
                    for key, value in kpis.items():
                        # Color coding for KPIs
                        if 'Critical' in key:
                            delta_color = "inverse" if critical_pct > 10 else "normal"
                        elif 'Excess' in key:
                            delta_color = "inverse" if excess_pct > 20 else "normal"
                        else:
                            delta_color = "normal"
                        
                        st.metric(key, value)
            
            # Detailed breakdowns
            st.subheader("📈 Detailed Breakdowns")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**By Classification:**")
                if not filtered_df.empty:
                    class_breakdown = filtered_df['Classification'].value_counts()
                    for class_name, count in class_breakdown.items():
                        pct = (count / len(filtered_df)) * 100
                        st.write(f"• Class {class_name}: {count} ({pct:.1f}%)")
            
            with col2:
                st.write("**By Coverage Category:**")
                if not filtered_df.empty:
                    coverage_breakdown = filtered_df['Coverage Category'].value_counts()
                    for category, count in coverage_breakdown.items():
                        pct = (count / len(filtered_df)) * 100
                        st.write(f"• {category}: {count} ({pct:.1f}%)")
            
            with col3:
                st.write("**By Status:**")
                if not filtered_df.empty:
                    status_breakdown = filtered_df['Status'].value_counts()
                    for status, count in status_breakdown.items():
                        pct = (count / len(filtered_df)) * 100
                        st.write(f"• {status}: {count} ({pct:.1f}%)")
            
            # Action Items
            st.subheader("🎯 Action Items")
            
            if not filtered_df.empty:
                # Generate action items based on analysis
                action_items = []
                
                critical_items = len(filtered_df[filtered_df['Coverage Days'] < 3])
                if critical_items > 0:
                    action_items.append(f"🚨 Urgent: Order stock for {critical_items} critical items")
                
                excess_items = len(filtered_df[filtered_df['Status'] == 'Excess Inventory'])
                if excess_items > 0:
                    action_items.append(f"📉 Review: {excess_items} items have excess inventory")
                
                short_items = len(filtered_df[filtered_df['Status'] == 'Short Inventory'])
                if short_items > 0:
                    action_items.append(f"📈 Reorder: {short_items} items need stock replenishment")
                
                high_consumption = filtered_df[filtered_df['Avg Consumption'] > filtered_df['Avg Consumption'].quantile(0.8)]
                if not high_consumption.empty:
                    action_items.append(f"👀 Monitor: {len(high_consumption)} high-consumption items need close monitoring")
                
                if action_items:
                    for i, item in enumerate(action_items, 1):
                        st.write(f"{i}. {item}")
                else:
                    st.success("✅ No immediate action items - inventory levels are healthy!")
    
    else:
        st.info("No inventory data available. Upload a file or use the sample data to get started.")
    
    # Add new part form (modal-like)
    if st.session_state.get('show_add_form', False):
        with st.expander("➕ Add New Part", expanded=True):
            with st.form("add_part_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_part_no = st.text_input("Part Number*")
                    new_description = st.text_input("Description*")
                    new_classification = st.selectbox("Classification*", ["A", "B", "C"])
                    new_inv_class = st.text_input("Inventory Class")
                
                with col2:
                    new_avg_consumption = st.number_input("Average Consumption/Day*", min_value=0.0, step=0.1)
                    new_rm_norms_days = st.number_input("RM Norms (Days)*", min_value=0.0, step=1.0)
                    new_rm_norms_qty = st.number_input("RM Norms (Qty)*", min_value=0.0, step=1.0)
                    new_current_stock = st.number_input("Current Stock*", min_value=0.0, step=1.0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    submitted = st.form_submit_button("Add Part", use_container_width=True)
                with col2:
                    if st.form_submit_button("Cancel", use_container_width=True):
                        st.session_state.show_add_form = False
                        st.rerun()
                
                if submitted:
                    if all([new_part_no, new_description, new_avg_consumption, new_rm_norms_days, new_rm_norms_qty]):
                        # Add new part to the data
                        new_part_data = [{
                            'Part No': new_part_no,
                            'Part Description': new_description,
                            'Part Classification': new_classification,
                            'Inventory Class': new_inv_class,
                            'Average Consumption/Day': new_avg_consumption,
                            'RM Norms - In Days': new_rm_norms_days,
                            'RM Norms - In Qty': new_rm_norms_qty,
                            'Current Stock Qty': new_current_stock
                        }]
                        
                        # Process and add the new data
                        inv_system.process_imported_data(st.session_state.inventory_data + 
                                                       [dict(zip(['Part No', 'Description', 'Classification', 'Inventory Class',
                                                                'Avg Consumption', 'RM Norms Days', 'RM Norms Qty', 'Current Stock'],
                                                               [item['Part No'], item['Description'], item['Classification'], 
                                                                item['Inventory Class'], item['Avg Consumption'], item['RM Norms Days'],
                                                                item['RM Norms Qty'], item['Current Stock']])) 
                                                        for item in st.session_state.inventory_data] + 
                                                       [{'Part No': new_part_no, 'Description': new_description, 
                                                         'Classification': new_classification, 'Inventory Class': new_inv_class,
                                                         'Avg Consumption': new_avg_consumption, 'RM Norms Days': new_rm_norms_days,
                                                         'RM Norms Qty': new_rm_norms_qty, 'Current Stock': new_current_stock}])
                        
                        st.success(f"Part {new_part_no} added successfully!")
                        st.session_state.show_add_form = False
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields (marked with *)")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
        "<p>Inventory Coverage Meter v2.0 | Built with ❤️ by Agilomatrix</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
