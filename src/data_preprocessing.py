#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for E-commerce Chatbot
Handles cleaning, structuring, and preparing data for NLP models
"""

import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedData:
    """Container for processed data"""
    faq_data: pd.DataFrame
    product_data: pd.DataFrame
    policy_chunks: List[Dict[str, Any]]
    training_data: pd.DataFrame
    validation_data: pd.DataFrame
    test_data: pd.DataFrame

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self, data_dir: str = "data", documents_dir: str = "documents"):
        self.data_dir = Path(data_dir)
        self.documents_dir = Path(documents_dir)
        self.processed_dir = Path("processed_data")
        self.processed_dir.mkdir(exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        return text
    
    def process_faq_dataset(self) -> pd.DataFrame:
        """Process FAQ dataset into structured format with validation"""
        logger.info("Processing FAQ dataset...")
        faq_file = self.data_dir / "faq_dataset.json"
        if not faq_file.exists():
            raise FileNotFoundError(f"FAQ dataset not found: {faq_file}")
        with open(faq_file, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        processed_faqs = []
        issues = []
        seen = set()
        for category in faq_data['faqs']:
            category_name = category.get('category', '').strip()
            if not category_name:
                issues.append('Missing category name')
                continue
            for qa in category.get('questions', []):
                question = self.clean_text(qa.get('question', ''))
                answer = self.clean_text(qa.get('answer', ''))
                key = (category_name, question)
                if not question or not answer:
                    issues.append(f'Missing Q/A in category {category_name}')
                    continue
                if key in seen:
                    issues.append(f'Duplicate Q in category {category_name}: {question}')
                    continue
                seen.add(key)
                processed_faqs.append({
                    'category': category_name,
                    'question': question,
                    'answer': answer,
                    'question_length': len(question.split()),
                    'answer_length': len(answer.split()),
                    'type': 'faq'
                })
        if issues:
            logger.warning(f"FAQ validation issues: {issues}")
        df = pd.DataFrame(processed_faqs)
        logger.info(f"Processed {len(df)} FAQ entries (skipped {len(issues)})")
        return df
    
    def process_product_catalog(self) -> pd.DataFrame:
        """Process product catalog into structured format with validation"""
        logger.info("Processing product catalog...")
        product_file = self.data_dir / "product_catalog.json"
        if not product_file.exists():
            raise FileNotFoundError(f"Product catalog not found: {product_file}")
        with open(product_file, 'r', encoding='utf-8') as f:
            product_data = json.load(f)
        processed_products = []
        issues = []
        seen = set()
        for product in product_data['products']:
            pid = product.get('id', '').strip()
            name = self.clean_text(product.get('name', ''))
            if not pid or not name:
                issues.append('Missing product id or name')
                continue
            if pid in seen:
                issues.append(f'Duplicate product id: {pid}')
                continue
            seen.add(pid)
            features = ', '.join(product.get('features', []))
            specs = product.get('specifications', {})
            spec_text = ', '.join([f"{k}: {v}" for k, v in specs.items()])
            description = f"{product.get('description', '')} Features: {features}. Specifications: {spec_text}"
            description = self.clean_text(description)
            search_text = f"{name} {product.get('brand', '')} {product.get('category', '')} {product.get('subcategory', '')} {description}"
            search_text = self.clean_text(search_text)
            processed_products.append({
                'product_id': pid,
                'name': name,
                'category': product.get('category', ''),
                'subcategory': product.get('subcategory', ''),
                'brand': product.get('brand', ''),
                'price': product.get('price', 0.0),
                'description': description,
                'search_text': search_text,
                'features': features,
                'specifications': json.dumps(specs),
                'stock': product.get('stock', 0),
                'rating': product.get('rating', 0.0),
                'reviews_count': product.get('reviews_count', 0),
                'warranty': product.get('warranty', ''),
                'tags': ', '.join(product.get('tags', [])),
                'type': 'product'
            })
        if issues:
            logger.warning(f"Product catalog validation issues: {issues}")
        df = pd.DataFrame(processed_products)
        logger.info(f"Processed {len(df)} products (skipped {len(issues)})")
        return df
    
    def process_policy_documents(self) -> List[Dict[str, Any]]:
        """Process policy documents into chunks for RAG with validation"""
        logger.info("Processing policy documents...")
        policy_files = [
            "return_policy.txt",
            "shipping_policy.txt", 
            "privacy_policy.txt",
            "terms_of_service.txt",
            "customer_service_guidelines.txt"
        ]
        policy_chunks = []
        issues = []
        for filename in policy_files:
            file_path = self.documents_dir / filename
            if not file_path.exists():
                issues.append(f"Policy file not found: {file_path}")
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            sections = self._split_into_sections(content)
            for i, section in enumerate(sections):
                if len(section.strip()) > 50:
                    cleaned_section = self.clean_text(section)
                    policy_chunks.append({
                        'document': filename.replace('.txt', ''),
                        'section_id': i,
                        'content': cleaned_section,
                        'content_length': len(cleaned_section.split()),
                        'type': 'policy'
                    })
                else:
                    issues.append(f"Skipped short/empty section in {filename} (section {i})")
        if issues:
            logger.warning(f"Policy document validation issues: {issues}")
        logger.info(f"Processed {len(policy_chunks)} policy chunks (skipped {len(issues)})")
        return policy_chunks
    
    def _split_into_sections(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Split document content into manageable chunks"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new section (numbered or all caps)
            is_new_section = (
                re.match(r'^\d+\.', line) or
                (line.isupper() and len(line) < 100) or
                line.endswith(':') and len(line) < 50
            )
            
            # Start new chunk if section is too large or new section detected
            if (current_size > max_chunk_size and is_new_section) or current_size > max_chunk_size * 2:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = len(line.split())
            else:
                current_chunk.append(line)
                current_size += len(line.split())
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def create_training_datasets(self, faq_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create training, validation, and test datasets for fine-tuning"""
        logger.info("Creating training datasets...")
        
        # Create conversation pairs for training
        training_pairs = []
        
        for _, row in faq_df.iterrows():
            # Create multiple conversation formats
            formats = [
                # Direct Q&A format
                {
                    'instruction': 'Answer the following customer question about our e-commerce store.',
                    'input': row['question'],
                    'output': row['answer'],
                    'category': row['category']
                },
                # Conversational format
                {
                    'instruction': 'You are a helpful e-commerce customer service representative. Help the customer with their inquiry.',
                    'input': f"Customer: {row['question']}",
                    'output': f"Assistant: {row['answer']}",
                    'category': row['category']
                },
                # Context-aware format
                {
                    'instruction': f'As a customer service agent for an e-commerce store, answer this {row["category"]} related question.',
                    'input': row['question'],
                    'output': row['answer'],
                    'category': row['category']
                }
            ]
            
            training_pairs.extend(formats)
        
        # Create DataFrame
        training_df = pd.DataFrame(training_pairs)
        
        # Check if all categories have at least 2 samples for stratification
        cat_counts = training_df['category'].value_counts()
        can_stratify = (cat_counts >= 2).all()
        
        try:
            if can_stratify:
                train_df, temp_df = train_test_split(
                    training_df, test_size=0.3, random_state=42, stratify=training_df['category'])
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42, stratify=temp_df['category'])
            else:
                logger.warning("Not all categories have at least 2 samples. Falling back to non-stratified split.")
                train_df, temp_df = train_test_split(
                    training_df, test_size=0.3, random_state=42, shuffle=True)
                val_df, test_df = train_test_split(
                    temp_df, test_size=0.5, random_state=42, shuffle=True)
        except Exception as e:
            logger.error(f"Error during train/val/test split: {e}")
            logger.warning("Falling back to non-stratified split for all splits.")
            train_df, temp_df = train_test_split(
                training_df, test_size=0.3, random_state=42, shuffle=True)
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42, shuffle=True)
        
        # Reset index
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Log category distribution
        logger.info(f"Train category distribution: {train_df['category'].value_counts().to_dict()}")
        logger.info(f"Val category distribution: {val_df['category'].value_counts().to_dict()}")
        logger.info(f"Test category distribution: {test_df['category'].value_counts().to_dict()}")
        
        logger.info(f"Created training dataset: {len(train_df)} samples")
        logger.info(f"Created validation dataset: {len(val_df)} samples")
        logger.info(f"Created test dataset: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, processed_data: ProcessedData):
        """Save processed data to files"""
        logger.info("Saving processed data...")
        
        # Save FAQ data
        processed_data.faq_data.to_csv(self.processed_dir / "faq_processed.csv", index=False)
        
        # Save product data
        processed_data.product_data.to_csv(self.processed_dir / "products_processed.csv", index=False)
        
        # Save policy chunks
        with open(self.processed_dir / "policy_chunks.json", 'w') as f:
            json.dump(processed_data.policy_chunks, f, indent=2)
        
        # Save training datasets
        processed_data.training_data.to_csv(self.processed_dir / "train_data.csv", index=False)
        processed_data.validation_data.to_csv(self.processed_dir / "val_data.csv", index=False)
        processed_data.test_data.to_csv(self.processed_dir / "test_data.csv", index=False)
        
        # Save complete processed data object
        with open(self.processed_dir / "processed_data.pkl", 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info("All processed data saved successfully")
    
    def generate_data_summary(self, processed_data: ProcessedData) -> Dict[str, Any]:
        """Generate summary statistics of processed data"""
        summary = {
            'faq_data': {
                'total_entries': len(processed_data.faq_data),
                'categories': processed_data.faq_data['category'].nunique(),
                'avg_question_length': processed_data.faq_data['question_length'].mean(),
                'avg_answer_length': processed_data.faq_data['answer_length'].mean(),
                'category_distribution': processed_data.faq_data['category'].value_counts().to_dict()
            },
            'product_data': {
                'total_products': len(processed_data.product_data),
                'categories': processed_data.product_data['category'].nunique(),
                'brands': processed_data.product_data['brand'].nunique(),
                'price_range': {
                    'min': processed_data.product_data['price'].min(),
                    'max': processed_data.product_data['price'].max(),
                    'mean': processed_data.product_data['price'].mean()
                }
            },
            'policy_data': {
                'total_chunks': len(processed_data.policy_chunks),
                'documents': len(set(chunk['document'] for chunk in processed_data.policy_chunks)),
                'avg_chunk_length': np.mean([chunk['content_length'] for chunk in processed_data.policy_chunks])
            },
            'training_data': {
                'train_samples': len(processed_data.training_data),
                'val_samples': len(processed_data.validation_data),
                'test_samples': len(processed_data.test_data),
                'total_samples': len(processed_data.training_data) + len(processed_data.validation_data) + len(processed_data.test_data)
            }
        }
        
        return summary
    
    def run_preprocessing_pipeline(self) -> ProcessedData:
        """Run complete preprocessing pipeline"""
        logger.info("Starting data preprocessing pipeline...")
        
        # Process all data sources
        faq_data = self.process_faq_dataset()
        product_data = self.process_product_catalog()
        policy_chunks = self.process_policy_documents()
        
        # Create training datasets
        train_data, val_data, test_data = self.create_training_datasets(faq_data)
        
        # Create processed data object
        processed_data = ProcessedData(
            faq_data=faq_data,
            product_data=product_data,
            policy_chunks=policy_chunks,
            training_data=train_data,
            validation_data=val_data,
            test_data=test_data
        )
        
        # Save processed data
        self.save_processed_data(processed_data)
        
        # Generate and save summary
        summary = self.generate_data_summary(processed_data)
        with open(self.processed_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Data preprocessing pipeline completed successfully!")
        return processed_data

def main():
    """Main function to run preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_preprocessing_pipeline()
    
    # Print summary
    summary = preprocessor.generate_data_summary(processed_data)
    print("\n" + "="*50)
    print("DATA PREPROCESSING SUMMARY")
    print("="*50)
    print(f"FAQ Entries: {summary['faq_data']['total_entries']}")
    print(f"Products: {summary['product_data']['total_products']}")
    print(f"Policy Chunks: {summary['policy_data']['total_chunks']}")
    print(f"Training Samples: {summary['training_data']['total_samples']}")
    print("="*50)

if __name__ == "__main__":
    main() 