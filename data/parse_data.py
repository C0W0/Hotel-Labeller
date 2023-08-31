import csv
import json

def store_to_local(data: dict[str, list[str]], directory: str) -> None:
    for (name, comments) in data.items():
        with open(f'./data/{directory}/{name}.json', 'w+') as json_file:
            json.dump(comments, json_file, indent=4)

def main():
    positive_data: dict[str, list[str]] = {}
    negative_data: dict[str, list[str]] = {}

    with open('./data/Hotel_Reviews.csv') as raw_data_file:
        reader = csv.DictReader(raw_data_file)
        for row in reader:
            hotel_name = row['Hotel_Name']

            if not hotel_name in positive_data:
                positive_data[hotel_name] = []
                negative_data[hotel_name] = []

            positive_data[hotel_name].append(row['Positive_Review'].strip())
            negative_data[hotel_name].append(row['Negative_Review'].strip())
    
    store_to_local(positive_data, 'positive')
    store_to_local(negative_data, 'negative')
    



if __name__ == '__main__': main()