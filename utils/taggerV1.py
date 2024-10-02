import os
import re
import random

def tag_text_with_silence(input_file):
    # Define punctuation and their corresponding silence tag ranges
    punctuation_ranges = {
        ".PERIOD": (0.900, 1.000),
        ",COMMA": (0.300, 0.650),
        "?QUESTIONMARK": (0.850, 1.000),
        ";SEMICOLON": (0.300, 0.650),
        "!EXCLAMATIONMARK": (0.850, 1.000)
    }

    # Read input text from file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Define regular expression to find punctuation
    punctuation_pattern = r'(\.PERIOD|,COMMA|\?QUESTIONMARK|;SEMICOLON|!EXCLAMATIONMARK)'

    # Split text by punctuation while preserving punctuation
    segments = re.split(punctuation_pattern, text)

    # Initialize tagged text
    tagged_text = []

    # Iterate over segments and add silence tags
    for segment in segments:
        # If segment is punctuation, just add it to the tagged text
        if re.match(punctuation_pattern, segment):
            tagged_text.append(segment)
        else:
            # Add text segment
            tagged_text.append(segment.strip())
            # Add silence tag if segment is not empty and not the last one
            if segment and segments.index(segment) != len(segments) - 1:
                punctuation = segments[segments.index(segment) + 1]
                silence_range = punctuation_ranges.get(punctuation, (0, 0))
                random_silence = random.uniform(*silence_range)
                tagged_text.append(f'<sil={random_silence:.3f}>')

    # Combine segments and tagged text
    tagged_text = ' '.join(tagged_text)

    return tagged_text

def process_files(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            tagged_text = tag_text_with_silence(input_file_path)
            with open(output_file_path, 'w', encoding="utf-8") as file:
                file.write(tagged_text)

# Input and output folder paths
input_folder = r"C:\Users\Dell\OneDrive\Desktop\CoursesDataset\PUNCtagged"
output_folder = r"C:\Users\Dell\OneDrive\Desktop\CoursesDataset\FINtagged"

# Process files
process_files(input_folder, output_folder)


