input_file_path = r'C:\Users\Dell\OneDrive\Desktop\MAJOR\majorV3\punctuator2-BiDirectionalRNN\results\test.txt' 
output_file_path = r'C:\Users\Dell\OneDrive\Desktop\MAJOR\majorV3\punctuator2-BiDirectionalRNN\results\testPunced.txt'  

# Define the mapping of punctuation characters to their replacements
punctuation_mapping = {
    ',': 'COMMA',
    '.': 'PERIOD',
    '?': 'QUESTIONMARK',
    ';': 'SEMICOLON',
    '!': 'EXCLAMATIONMARK'
}

# Read the input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Process the text to add the replacements after punctuation characters
for char, replacement in punctuation_mapping.items():
    text = text.replace(char, f' {char}{replacement} ')

# Write the modified text to the output file
with open(output_file_path, 'w') as file:
    file.write(text)

print(f"Text has been processed and saved to {output_file_path}")
