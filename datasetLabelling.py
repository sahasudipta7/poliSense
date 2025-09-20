import csv

input_file = "C:\Users\sahas\Downloads\february_1.csv"
output_file = 'labeled_tweets.csv'

# Open the input and output files
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read the header, add a 'label' column, and write to the new file
    header = next(reader)
    header.append('label')
    writer.writerow(header)

    # Process each row
    for row in reader:
        # Assuming the tweet content is in the 7th column (index 6)
        tweet = row[6]
        print(f"\nTweet: {tweet}")
        label = input("Enter a label ('for', 'against', or 'neutral'): ")
        row.append(label)
        writer.writerow(row)

print("\nLabeling complete. The results have been saved to labeled_tweets.csv")