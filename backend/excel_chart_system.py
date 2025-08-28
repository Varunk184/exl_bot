import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

class SimpleChartGenerator:
    def __init__(self):
        # Power BI Colors
        self.colors = {
            'blue': '#118DFF',
            'dark_blue': '#004B87',
            'orange': '#FF6B35'
        }
       
        # Create charts folder
        os.makedirs("generated_charts", exist_ok=True)
   
    def should_create_chart(self, query, search_results):
        """Simple check - only for financial and attrition"""
        query_lower = query.lower()
       
        # Check for financial keywords
        if any(word in query_lower for word in ['budget', 'revenue', 'financial', 'forecast']):
            return 'financial'
           
        # Check for attrition keywords  
        if any(word in query_lower for word in ['attrition', 'turnover', 'retention']):
            return 'attrition'
           
        return None
   
    def extract_data_from_results(self, search_results, chart_type):
        """Extract actual data from your search results"""
       
        all_data = []
       
        for metadata, score in search_results:
            chunk_text = metadata['chunk_text']
            lines = chunk_text.split('\n')
           
            data_row = {}
           
            # Extract key-value pairs from chunk
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                       
                        # Try to extract number
                        number_match = re.search(r'[-+]?\d*\.?\d+', value)
                        if number_match:
                            try:
                                num_value = float(number_match.group())
                                data_row[key] = num_value
                            except ValueError:
                                continue
           
            # Extract period/date
            period = self.extract_period(chunk_text)
            if period:
                data_row['period'] = period
           
            # Add metadata
            data_row['sheet'] = metadata['sheet_name']
            data_row['file'] = metadata['file_name']
           
            if data_row:
                all_data.append(data_row)
       
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
   
    def extract_period(self, text):
        """Extract time period from text"""
       
        # Quarter patterns
        quarter_match = re.search(r'[Qq]([1-4])', text)
        if quarter_match:
            return f"Qtr {quarter_match.group(1)}"
       
        # Month patterns
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
       
        for i, month in enumerate(months, 1):
            if month in text.lower():
                year_match = re.search(r'20[0-9]{2}', text)
                year = year_match.group()[-2:] if year_match else '25'
                return f"{month.capitalize()}-{year}"
       
        return None
   
    def create_financial_chart(self, search_results):
        """Create financial chart from actual data"""
       
        # Extract data from search results
        df = self.extract_data_from_results(search_results, 'financial')
       
        if df.empty:
            print("No financial data found")
            return None
       
        # Try to identify financial columns
        budget_col = None
        actual_col = None
        previous_col = None
       
        for col in df.columns:
            if 'budget' in str(col).lower():
                budget_col = col
            elif 'actual' in str(col).lower() or 'forecast' in str(col).lower():
                actual_col = col
            elif 'previous' in str(col).lower():
                previous_col = col
       
        # Group by period if available
        if 'period' in df.columns:
            df_grouped = df.groupby('period').mean()
        else:
            df_grouped = df.head(6)  # Take first 6 rows
       
        periods = df_grouped.index.tolist()
       
        # Get actual data values
        budget_data = df_grouped[budget_col].values if budget_col else [0] * len(periods)
        actual_data = df_grouped[actual_col].values if actual_col else [0] * len(periods)
        previous_data = df_grouped[previous_col].values if previous_col else [0] * len(periods)
       
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
       
        x = np.arange(len(periods))
        width = 0.25
       
        # Create bars with actual data
        bars1 = ax.bar(x - width, budget_data, width, label='● Budget',
                      color=self.colors['blue'], alpha=0.9)
        bars2 = ax.bar(x, actual_data, width, label='● Actual+Forecast',
                      color=self.colors['dark_blue'], alpha=0.9)
        bars3 = ax.bar(x + width, previous_data, width, label='● Previous Forecast',
                      color=self.colors['orange'], alpha=0.9)
       
        # Add value labels with actual values
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if height >= 1000000:
                        label = f'${height/1000000:.1f}M'
                    elif height >= 1000:
                        label = f'${height/1000:.0f}K'
                    else:
                        label = f'${height:.0f}'
                   
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(max(budget_data), max(actual_data), max(previous_data))*0.02,
                           label, ha='center', va='bottom', fontweight='bold')
       
        # Styling
        ax.set_title('Financial - Revenue', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
       
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"generated_charts/financial_chart_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
       
        print(f"Financial chart created with {len(periods)} periods")
        return filepath
   
    def create_attrition_chart(self, search_results):
        """Create attrition chart from actual data"""
       
        # Extract data from search results
        df = self.extract_data_from_results(search_results, 'attrition')
       
        if df.empty:
            print("No attrition data found")
            return None
       
        # Try to identify attrition columns
        ytd_col = None
        mtd_col = None
       
        for col in df.columns:
            if 'ytd' in str(col).lower():
                ytd_col = col
            elif 'mtd' in str(col).lower():
                mtd_col = col
            elif '%' in str(col).lower() and not ytd_col:
                ytd_col = col  # Use first percentage column as YTD
       
        # Group by period if available
        if 'period' in df.columns:
            df_grouped = df.groupby('period').mean()
        else:
            df_grouped = df.head(6)  # Take first 6 rows
       
        periods = df_grouped.index.tolist()
       
        # Get actual data values
        ytd_data = df_grouped[ytd_col].values if ytd_col else [0] * len(periods)
        mtd_data = df_grouped[mtd_col].values if mtd_col else ytd_data  # Use YTD if no MTD
       
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
       
        x = np.arange(len(periods))
       
        # Create bars with actual data
        bars = ax.bar(x, ytd_data, color=self.colors['blue'], alpha=0.8,
                     label='● YTD Annualized', width=0.6)
       
        # Create line with actual data
        ax.plot(x, mtd_data, color=self.colors['orange'], marker='o',
               linewidth=3, markersize=8, label='● MTD Annualized')
       
        # Add percentage labels with actual values
        for i, (bar, ytd_val, mtd_val) in enumerate(zip(bars, ytd_data, mtd_data)):
            # YTD labels
            if ytd_val > 5:
                ax.text(bar.get_x() + bar.get_width()/2., ytd_val/2,
                       f'{ytd_val:.1f}%', ha='center', va='center',
                       fontweight='bold', color='white')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., ytd_val + 0.5,
                       f'{ytd_val:.1f}%', ha='center', va='bottom',
                       fontweight='bold')
           
            # MTD labels
            ax.text(i, mtd_val + 0.8, f'{mtd_val:.1f}%', ha='center', va='bottom',
                   color=self.colors['orange'], fontweight='bold')
       
        # Target line (use 17% as default, or calculate from data)
        target = 17.0
        ax.axhline(y=target, color='red', linestyle='--', alpha=0.7)
        ax.text(len(periods)-1, target + 0.5, f'YTD Target - {target}%',
               ha='right', va='bottom', color='red', fontweight='bold')
       
        # Styling
        ax.set_title('Attrition', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Attrition Rate (%)')
       
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"generated_charts/attrition_chart_{timestamp}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
       
        print(f"Attrition chart created with {len(periods)} periods")
        return filepath
   
    def generate_chart(self, query, search_results):
        """Main function - extracts real data and creates charts"""
       
        chart_type = self.should_create_chart(query, search_results)
       
        if chart_type == 'financial':
            filepath = self.create_financial_chart(search_results)
            if filepath:
                return {
                    'has_chart': True,
                    'chart_type': 'financial',
                    'chart_path': filepath,
                    'message': f'Financial chart created from your data: {filepath}'
                }
        elif chart_type == 'attrition':
            filepath = self.create_attrition_chart(search_results)
            if filepath:
                return {
                    'has_chart': True,
                    'chart_type': 'attrition',
                    'chart_path': filepath,
                    'message': f'Attrition chart created from your data: {filepath}'
                }
       
        return {'has_chart': False}


# Integration with your existing query system
def integrate_with_query_system():
    """Add this to your existing query system"""
   
    # In your query() function, add:
    chart_generator = SimpleChartGenerator()
    chart_info = chart_generator.generate_chart(user_query, search_results)
   
    if chart_info['has_chart']:
        print(f"📊 Chart generated from your Excel data: {chart_info['chart_path']}")
   
    return chart_info