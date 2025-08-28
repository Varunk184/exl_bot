import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import logging
from typing import Dict, List, Tuple, Any
import openai
import re
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelQuerySystem:
    def __init__(self, vector_db_path: str = "vector_db", model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the query system"""
        self.vector_db_path = vector_db_path
        self.model = SentenceTransformer(model_name)
       
        # Load vector database
        self.faiss_index = self.load_faiss_index()
        self.metadata = self.load_metadata()
        self.db_info = self.load_db_info()
       
        # Configure OpenAI (set your credentials)
        self.setup_openai()
       
        # Simple graph keywords
        self.finance_keywords = ['finance', 'financial', 'revenue', 'budget', 'trend']
        self.attrition_keywords = ['attrition', 'trend']
       
        # Create charts directory
        self.charts_dir = "generated_charts"
        os.makedirs(self.charts_dir, exist_ok=True)
       
        logger.info(f"Loaded vector DB with {len(self.metadata)} chunks")
   
    def setup_openai(self):
        """Setup OpenAI configuration"""
        # Set your OpenAI credentials here
        openai.api_type = "azure"  # or "openai" for regular OpenAI
        openai.api_base = "https://testdemo-aoai.openai.azure.com/" # Your API base URL
        openai.api_version = "2024-03-01-preview"  # Your API version
        openai.api_key = "f3f47f64b30748b193bbbc78ee889763" # Your API key
       
        # Note: Replace the above with your actual credentials
   
    def load_faiss_index(self) -> faiss.Index:
        """Load FAISS index"""
        index_path = os.path.join(self.vector_db_path, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        return faiss.read_index(index_path)
   
    def load_metadata(self) -> List[Dict]:
        """Load metadata"""
        metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
       
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
   
    def load_db_info(self) -> Dict:
        """Load database info"""
        info_path = os.path.join(self.vector_db_path, "db_info.pkl")
        if not os.path.exists(info_path):
            return {}
       
        with open(info_path, 'rb') as f:
            return pickle.load(f)
   
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Perform semantic search on the vector database"""
       
        # Create embedding for the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
       
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
       
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
       
        # Get results with metadata
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                metadata = self.metadata[idx]
                results.append((metadata, float(score)))
       
        return results
   
    def extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        # Pattern to match numbers (including decimals, percentages, currency)
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        numbers = re.findall(number_pattern, text)
        return [float(num) for num in numbers if num]
   
    def should_generate_graph(self, query: str) -> str:
        """Simple check for graph generation"""
        query_lower = query.lower()
       
        if 'trend' in query_lower and any(word in query_lower for word in self.finance_keywords):
            return 'finance'
        elif 'trend' in query_lower and any(word in query_lower for word in self.attrition_keywords):
            return 'attrition'
       
        return None
   
    def extract_data_for_graph(self, results: List[Tuple[Dict, float]], graph_type: str) -> Dict:
        """Extract data from search results for graphing"""
        data = {}
       
        print(f"Extracting data for {graph_type} chart from {len(results)} results...")
       
        for metadata, score in results:
            chunk_text = metadata['chunk_text']
            print(f"Processing chunk: {chunk_text[:100]}...")
           
            if graph_type == 'finance':
                # Look for quarterly data (Q1, Q2, etc.) OR convert months to quarters
                quarters = re.findall(r'Q[1-4]|Qtr\s*[1-4]', chunk_text, re.IGNORECASE)
               
                # If no quarters found, look for months and convert to quarters
                if not quarters:
                    month_matches = re.findall(r'2024-(\d{2})-|2025-(\d{2})-', chunk_text)
                    for match in month_matches:
                        month_num = int(match[0] if match[0] else match[1])
                        if 1 <= month_num <= 3:
                            quarters.append('Q1')
                        elif 4 <= month_num <= 6:
                            quarters.append('Q2')
                        elif 7 <= month_num <= 9:
                            quarters.append('Q3')
                        elif 10 <= month_num <= 12:
                            quarters.append('Q4')
               
                # Extract revenue/cost numbers
                revenue_matches = re.findall(r'Revenue:\s*([\d,]+\.?\d*)', chunk_text)
                cost_matches = re.findall(r'Cost:\s*([\d,]+\.?\d*)', chunk_text)
               
                numbers = []
                for rev in revenue_matches:
                    numbers.append(float(rev.replace(',', '')))
                for cost in cost_matches:
                    numbers.append(float(cost.replace(',', '')))
               
                print(f"Found quarters: {quarters}, revenue/cost numbers: {numbers}")
               
                for quarter in quarters:
                    if quarter not in data:
                        data[quarter] = []
                    data[quarter].extend(numbers)
           
            elif graph_type == 'attrition':
                # Look for monthly data
                months = re.findall(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', chunk_text, re.IGNORECASE)
               
                # Also look for month numbers like 2024-01, 2024-02
                if not months:
                    month_matches = re.findall(r'2024-(\d{2})-|2025-(\d{2})-', chunk_text)
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    for match in month_matches:
                        month_num = int(match[0] if match[0] else match[1])
                        if 1 <= month_num <= 12:
                            months.append(month_names[month_num - 1])
               
                numbers = self.extract_numbers_from_text(chunk_text)
               
                print(f"Found months: {months}, numbers: {numbers[:5]}...")
               
                for month in months:
                    if month not in data:
                        data[month] = []
                    data[month].extend(numbers)
       
        print(f"Final extracted data: {data}")
        return data
   
    def create_simple_chart(self, data: Dict, chart_type: str, query: str) -> str:
        """Create simple bar chart"""
        if not data:
            print("No data found for chart generation")
            return None
       
        print(f"Creating {chart_type} chart with data: {data}")
       
        # Prepare data for plotting
        labels = list(data.keys())
        values = [sum(data[label])/len(data[label]) if data[label] else 0 for label in labels]
       
        if not labels or not any(values):
            print("No valid data points found")
            return None
       
        # Create simple bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color='#3498db')
       
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
       
        # Customize chart
        title = f"{'Financial' if chart_type == 'finance' else 'Attrition'} Trend"
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Period')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
       
        # Save chart with absolute path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{chart_type}_trend_{timestamp}.png"
       
        # Make sure directory exists
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir, exist_ok=True)
            print(f"Created directory: {os.path.abspath(self.charts_dir)}")
       
        chart_path = os.path.join(self.charts_dir, filename)
       
        try:
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
           
            # Verify file was created
            if os.path.exists(chart_path):
                print(f"✅ Chart saved successfully at: {os.path.abspath(chart_path)}")
                return chart_path
            else:
                print(f"❌ Failed to save chart at: {os.path.abspath(chart_path)}")
                return None
               
        except Exception as e:
            print(f"Error saving chart: {e}")
            plt.close()
            return None
   
    def aggregate_numerical_data(self, results: List[Tuple[Dict, float]], query: str) -> Dict[str, Any]:
        """Aggregate numerical data from search results"""
        all_numbers = []
        numerical_context = []
       
        for metadata, score in results:
            chunk_text = metadata['chunk_text']
            numbers = self.extract_numbers_from_text(chunk_text)
           
            if numbers:
                all_numbers.extend(numbers)
                numerical_context.append({
                    'file': metadata['file_name'],
                    'sheet': metadata['sheet_name'],
                    'numbers': numbers,
                    'context': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'relevance_score': score
                })
       
        # Calculate aggregations if we have numbers
        aggregations = {}
        if all_numbers:
            aggregations = {
                'total_sum': sum(all_numbers),
                'average': np.mean(all_numbers),
                'min_value': min(all_numbers),
                'max_value': max(all_numbers),
                'count': len(all_numbers),
                'unique_values': len(set(all_numbers))
            }
       
        return {
            'aggregations': aggregations,
            'numerical_context': numerical_context,
            'total_chunks_found': len(results)
        }
   
    def create_context_for_llm(self, results: List[Tuple[Dict, float]], query: str) -> str:
        """Create context string for LLM from search results"""
       
        context_parts = []
        context_parts.append(f"User Query: {query}\n")
        context_parts.append("Relevant Data Found:\n")
        context_parts.append("=" * 50)
       
        # Group results by file and sheet for better organization
        grouped_results = {}
        for metadata, score in results:
            key = f"{metadata['file_name']}::{metadata['sheet_name']}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append((metadata, score))
       
        # Add organized context
        for file_sheet, chunk_results in grouped_results.items():
            context_parts.append(f"\nData from {file_sheet}:")
            context_parts.append("-" * 30)
           
            for metadata, score in chunk_results[:5]:  # Limit to top 5 per sheet
                chunk_text = metadata['chunk_text']
                # Clean up the chunk text
                clean_text = chunk_text.replace('\n', ' | ').strip()
                context_parts.append(f"• {clean_text}")
                context_parts.append(f"  (Relevance: {score:.3f})")
       
        return "\n".join(context_parts)
   
    def generate_llm_response(self, context: str, query: str, numerical_summary: Dict, chart_path: str = None) -> str:
        """Generate response using OpenAI"""
       
        # Create a comprehensive prompt
        system_prompt = """You are an expert data analyst helping users understand their Excel data.
       
        CRITICAL INSTRUCTIONS:
        1. Provide EXACT numbers from the data - never estimate or generate numbers
        2. If you see specific numerical values, report them precisely
        3. Always mention which file/sheet the data comes from
        4. If data spans multiple files, clearly combine and explain the relationship
        5. Be concise but comprehensive
        6. If you cannot find specific data to answer the query, say so clearly
        7. If a chart was generated, mention it in your response
       
        Focus on:
        - Exact numerical values and calculations
        - Clear source attribution (file and sheet names)
        - Actionable insights based on the actual data
        - Chart interpretation if available
        """
       
        chart_info = f"\n\nA visual chart has been generated and saved as: {chart_path}" if chart_path else ""
       
        user_prompt = f"""
        Based on the following data context, please answer the user's query.
       
        {context}
       
        Additional Numerical Summary:
        {json.dumps(numerical_summary, indent=2)}
       
        {chart_info}
       
        User Query: {query}
       
        Please provide a clear, accurate answer based ONLY on the data provided above.
        Include specific numbers, sources, and be precise about what the data shows.
        """
       
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4",  # or your deployment name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for factual responses
            )
           
            return response.choices[0].message.content.strip()
           
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error generating response: {str(e)}"
   
    def query(self, user_query: str, top_k: int = 15) -> Dict[str, Any]:
        """Main query function with graph generation"""
        logger.info(f"Processing query: {user_query}")
       
        # Step 1: Semantic search
        search_results = self.semantic_search(user_query, top_k)
       
        if not search_results:
            return {
                'query': user_query,
                'answer': "No relevant data found for your query.",
                'sources': [],
                'numerical_summary': {},
                'chart_generated': False,
                'chart_path': None
            }
       
        # Step 2: Check if graph needed and generate
        graph_type = self.should_generate_graph(user_query)
        chart_path = None
       
        if graph_type:
            logger.info(f"Generating {graph_type} chart")
            extracted_data = self.extract_data_for_graph(search_results, graph_type)
            chart_path = self.create_simple_chart(extracted_data, graph_type, user_query)
       
        # Step 3: Extract and aggregate numerical data
        numerical_summary = self.aggregate_numerical_data(search_results, user_query)
       
        # Step 4: Create context for LLM
        llm_context = self.create_context_for_llm(search_results, user_query)
       
        # Step 5: Generate response using OpenAI
        ai_response = self.generate_llm_response(llm_context, user_query, numerical_summary, chart_path)
       
        # Step 6: Prepare sources information
        sources = []
        for metadata, score in search_results[:10]:  # Top 10 sources
            sources.append({
                'file': metadata['file_name'],
                'sheet': metadata['sheet_name'],
                'relevance_score': round(score, 3),
                'preview': metadata['chunk_text'][:150] + "..."
            })
       
        return {
            'query': user_query,
            'answer': ai_response,
            'sources': sources,
            'numerical_summary': numerical_summary,
            'total_results_found': len(search_results),
            'chart_generated': chart_path is not None,
            'chart_path': chart_path,
            'chart_type': graph_type if chart_path else None
        }
   
    def interactive_query_loop(self):
        """Interactive query loop for testing"""
        print("\n" + "="*60)
        print("EXCEL DATA QUERY SYSTEM WITH GRAPH GENERATION")
        print("="*60)
        print("Available data sources:")
       
        # Show available sheets
        available_sheets = set()
        for metadata in self.metadata[:10]:  # Sample to show available sheets
            available_sheets.add(f"{metadata['file_name']}::{metadata['sheet_name']}")
       
        for sheet in sorted(available_sheets):
            print(f"• {sheet}")
       
        print("\nGraph Generation:")
        print("• Use 'financial trend' or 'attrition trend' in your query")
       
        print("\nEnter your queries (type 'quit' to exit):")
        print("-" * 60)
       
        while True:
            try:
                query = input("\nYour query: ").strip()
               
                if query.lower() in ['quit', 'exit', 'q']:
                    break
               
                if not query:
                    continue
               
                # Process query
                result = self.query(query)
               
                # Display results
                print(f"\nQuery: {result['query']}")
                print(f"Results found: {result['total_results_found']}")
               
                if result['chart_generated']:
                    print(f"📊 Chart Generated: {result['chart_type']} trend")
                    print(f"📁 Chart saved as: {result['chart_path']}")
                    print(f"📂 Location: {os.path.abspath(result['chart_path'])}")
               
                print("\nAnswer:")
                print("-" * 40)
                print(result['answer'])
               
                if result['numerical_summary']['aggregations']:
                    print(f"\nNumerical Summary:")
                    for key, value in result['numerical_summary']['aggregations'].items():
                        print(f"• {key}: {value}")
               
                print(f"\nTop Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"{i}. {source['file']} → {source['sheet']} (Score: {source['relevance_score']})")
               
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
       
        print("\nGoodbye!")


def main():
    """Main function to run the query system"""
    try:
        # Initialize query system
        query_system = ExcelQuerySystem()
       
        # Example queries for testing (including graph generation)
        test_queries = [
            "Show me financial trend",
            "What is attrition trend?",
            "What is the total budget for this year?",
            "Show me finance data",
            "Display attrition numbers"
        ]
       
        print("Example test queries (including graph generation):")
        for i, query in enumerate(test_queries, 1):
            print(f"{i}. {query}")
       
        # Start interactive loop
        query_system.interactive_query_loop()
       
    except Exception as e:
        logger.error(f"Error initializing query system: {str(e)}")
        print(f"Make sure you have:")
        print("1. Run the vector DB creation script first")
        print("2. Set up your OpenAI credentials")
        print("3. Installed matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()