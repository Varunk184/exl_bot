import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import logging
from typing import Dict, List, Tuple
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelVectorDB:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the vector database creator"""
        # Optimize for available hardware
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
       
        self.model = SentenceTransformer(model_name, device=device)
        self.vector_db = {}
        self.metadata = {}
       
        # Sheets to skip for faster processing
        self.skip_sheets = ['SLA_week', 'SLA']  # These are the massive sheets
       
    def load_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Load all sheets from Excel file"""
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            logger.info(f"Processing file: {os.path.basename(file_path)}")
            return excel_data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return {}
   
    def preprocess_dataframe(self, df: pd.DataFrame, sheet_name: str) -> List[str]:
        """Convert dataframe to text chunks for embedding"""
        chunks = []
       
        # Get column headers
        headers = df.columns.tolist()
       
        # Create text representation for each row
        for idx, row in df.iterrows():
            # Create a structured text representation
            row_text = f"Sheet: {sheet_name}\n"
            row_text += f"Row {idx + 1}:\n"
           
            # Add each column value with its header
            for col, value in zip(headers, row.values):
                if pd.notna(value):  # Skip NaN values
                    row_text += f"{col}: {value}\n"
           
            chunks.append(row_text.strip())
       
        logger.info(f"Created {len(chunks)} content chunks for {sheet_name}")
        return chunks
   
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks with batching for speed"""
        logger.info("Creating embeddings...")
       
        # Use batching for faster processing
        embeddings = self.model.encode(
            texts,
            batch_size=64,  # Increased batch size for speed
            show_progress_bar=True,
            convert_to_numpy=True
        )
       
        return embeddings
   
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index from embeddings"""
        dimension = embeddings.shape[1]
       
        # Use IndexFlatIP for cosine similarity (good for semantic search)
        index = faiss.IndexFlatIP(dimension)
       
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
       
        logger.info(f"Created FAISS index with {index.ntotal} vectors")
        return index
   
    def process_excel_files(self, file_paths: List[str], output_dir: str = "vector_db"):
        """Main function to process Excel files and create vector databases"""
       
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
       
        all_chunks = []
        all_metadata = []
       
        for file_path in file_paths:
            excel_data = self.load_excel_file(file_path)
            file_name = os.path.basename(file_path)
           
            for sheet_name, df in excel_data.items():
                # Skip massive sheets for POC
                if sheet_name in self.skip_sheets:
                    logger.info(f"Skipping large sheet: {sheet_name} (has {len(df)} rows)")
                    continue
               
                logger.info(f"Loaded sheet '{sheet_name}' with {len(df)} rows and {len(df.columns)} columns")
               
                # Skip empty sheets
                if df.empty:
                    logger.warning(f"Skipping empty sheet: {sheet_name}")
                    continue
               
                # Preprocess dataframe to text chunks
                chunks = self.preprocess_dataframe(df, sheet_name)
               
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'file_name': file_name,
                        'sheet_name': sheet_name,
                        'row_index': i,
                        'chunk_text': chunk
                    }
                    all_metadata.append(metadata)
               
                all_chunks.extend(chunks)
       
        logger.info(f"Total chunks to process: {len(all_chunks)}")
       
        # Create embeddings for all chunks
        if not all_chunks:
            logger.error("No data to process!")
            return
       
        embeddings = self.create_embeddings(all_chunks)
       
        # Create FAISS index
        faiss_index = self.create_faiss_index(embeddings)
       
        # Save everything
        self.save_vector_db(faiss_index, all_metadata, output_dir)
       
        logger.info(f"Vector database created successfully in {output_dir}")
        logger.info(f"Processed {len(all_chunks)} chunks from {len(file_paths)} files")
   
    def save_vector_db(self, index: faiss.Index, metadata: List[Dict], output_dir: str):
        """Save FAISS index and metadata"""
       
        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
       
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
       
        # Save model info
        model_info = {
            'model_name': self.model.get_sentence_embedding_dimension(),
            'total_vectors': index.ntotal,
            'dimension': index.d
        }
       
        info_path = os.path.join(output_dir, "db_info.pkl")
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
       
        logger.info(f"Saved FAISS index: {index_path}")
        logger.info(f"Saved metadata: {metadata_path}")
        logger.info(f"Saved DB info: {info_path}")


def main():
    # Initialize the vector DB creator
    db_creator = ExcelVectorDB()
   
    # List your Excel files here
    excel_files = [
        r'C:\AI_Project\New_proj\data\PC-Data_AON.xlsx',  # Replace with your file paths
        r'C:\AI_Project\New_proj\data\P&C_IND_Ops_Data.xlsx'
    ]
   
    # Check if files exist
    existing_files = []
    for file in excel_files:
        if os.path.exists(file):
            existing_files.append(file)
            logger.info(f"Found file: {file}")
        else:
            logger.warning(f"File not found: {file}")
   
    if not existing_files:
        logger.error("No Excel files found!")
        return
   
    # Create vector database
    db_creator.process_excel_files(existing_files)
   
    print("\n" + "="*50)
    print("VECTOR DATABASE CREATION COMPLETE!")
    print("="*50)
    print(f"Skipped sheets: {db_creator.skip_sheets}")
    print("Next step: Use the query system to search your data!")


if __name__ == "__main__":
    main()