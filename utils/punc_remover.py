def remove_punctuation(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        text = f_in.read()

    # Remove punctuation characters
    punctuation = ",.?!;"
    for char in punctuation:
        text = text.replace(char, ' ')

    # Write the modified text to the output file
    with open(output_file, 'w', encoding="utf-8") as f_out:
        f_out.write(text)

# Example usage
input_file = r'C:\Users\Dell\OneDrive\Desktop\CBooks\testingDataset.txt'
output_file = r'C:\Users\Dell\OneDrive\Desktop\CBooks\processedDataset\test.test.txt'

remove_punctuation(input_file, output_file)
