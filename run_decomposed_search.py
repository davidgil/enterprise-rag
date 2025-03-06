import argparse
from pprint import pprint
from typing import List, Dict, Any
import json
from decomposed_search import DecomposedSearchProcessor
import csv
import logging
from summarize_responses import summarize_responses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOP_K = 3

def main():
    """Main function that processes a single question."""
    
    processor = DecomposedSearchProcessor(top_k=TOP_K)
    file_path = "benchmark/test-dataset.csv"

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
                result['Expected Answer'] = row['Expected Answer']


                # Check if result has multiple responses and summarize if needed
                if len(result.get('responses', [])) > 1:
                    summary = summarize_responses(result['original_question'], result['responses'])
                    result['summary'] = summary
                    print(f"\nSUMMARIZED RESPONSE:\n{summary}\n")
                else:
                    result['summary'] = result['responses'][0]['response']  

                all_results.append(result)
                
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        logger.error(f"Error processing the file: {e}")

    # Save all results to a CSV file
    output_file = "benchmark/output.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=all_results[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(all_results)

if __name__ == "__main__":
    main() 