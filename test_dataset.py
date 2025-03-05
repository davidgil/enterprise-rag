import csv
from search_docs import process_search_query

def read_test_dataset():
    """
    Lee el archivo test-dataset.csv y para cada fila muestra el ID, la pregunta
    y llama a process_search_query con la pregunta.
    """
    try:
        with open('benchmark/test-dataset.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, delimiter=';')
            
            print("-" * 80)
            for row in csv_reader:
                print(f"{row['ID']} | {row['Question']}")
                process_search_query(row['Question'], 3, True)
                
    except FileNotFoundError:
        print("Error: El archivo 'benchmark/test-dataset.csv' no fue encontrado.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

if __name__ == "__main__":
    read_test_dataset() 