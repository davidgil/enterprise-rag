import argparse
from pprint import pprint
from typing import List, Dict, Any
import json
from decomposed_search import DecomposedSearchProcessor
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOP_K = 3

def main():
    """Main function that processes a single question."""
    
    processor = DecomposedSearchProcessor(top_k=TOP_K)
    file_path = "benchmark/test-dataset-2.csv"

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, delimiter=';')
            
            all_results = []
            for row in csv_reader:
                print(f"\n{'*' * 100}")
                print(f"ID: {row['ID']} | QUESTION: {row['Question']}")
                print(f"{'*' * 100}")
                
                result = processor.process_question(row['Question'])
                result['id'] = row['ID']

                

                all_results.append(result)
                pprint(all_results)
                
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logger.error(f"Error processing the file: {e}")


if __name__ == "__main__":
    main() 