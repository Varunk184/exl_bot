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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelQuerySystem:
    def __init__(self, vector_db_path: str = "vector_db", model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the query system"""
        self.vector_db_path = vector_db_path
        self.model = SentenceTransformer(model_name)
       

        self.faiss_index = self.load_faiss_index()
        self.metadata = self.load_metadata()
        self.db_info = self.load_db_info()
       

        self.setup_openai()
       
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
   
    def generate_llm_response(self, context: str, query: str, numerical_summary: Dict) -> str:
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
       
        Focus on:
        - Exact numerical values and calculations
        - Clear source attribution (file and sheet names)
        - Actionable insights based on the actual data
        """
       
        user_prompt = f"""
        Based on the following data context, please answer the user's query.
       
        {context}
       
        Additional Numerical Summary:
        {json.dumps(numerical_summary, indent=2)}
       
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
        """Main query function"""
        logger.info(f"Processing query: {user_query}")
       
        # Step 1: Semantic search
        search_results = self.semantic_search(user_query, top_k)
       
        if not search_results:
            return {
                'query': user_query,
                'answer': "No relevant data found for your query.",
                'sources': [],
                'numerical_summary': {}
            }
       
        # Step 2: Extract and aggregate numerical data
        numerical_summary = self.aggregate_numerical_data(search_results, user_query)
       
        # Step 3: Create context for LLM
        llm_context = self.create_context_for_llm(search_results, user_query)
       
        # Step 4: Generate response using OpenAI
        ai_response = self.generate_llm_response(llm_context, user_query, numerical_summary)
       
        # Step 5: Prepare sources information
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
            'total_results_found': len(search_results)
        }
   
    def interactive_query_loop(self):
        """Interactive query loop for testing"""
        print("\n" + "="*60)
        print("EXCEL DATA QUERY SYSTEM")
        print("="*60)
        print("Available data sources:")
       
        # Show available sheets
        available_sheets = set()
        for metadata in self.metadata[:10]:  # Sample to show available sheets
            available_sheets.add(f"{metadata['file_name']}::{metadata['sheet_name']}")
       
        for sheet in sorted(available_sheets):
            print(f"• {sheet}")
       
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
       
        # Example queries for testing
        test_queries = [
            "What is the total budget for this year?",
            "Show me attrition data by department",
            "What are the finance numbers for Q3?",
            "Compare MEI data across different periods",
            "What is the shrinkage percentage?"
        ]
       
        print("Example test queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"{i}. {query}")
       
        # Start interactive loop
        query_system.interactive_query_loop()
       
    except Exception as e:
        logger.error(f"Error initializing query system: {str(e)}")
        print(f"Make sure you have:")
        print("1. Run the vector DB creation script first")
        print("2. Set up your OpenAI credentials")
        print("3. Installed all required packages")


if __name__ == "__main__":
    main()