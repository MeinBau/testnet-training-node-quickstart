input_filepath="data/financial_news_clean.txt"
output_filepath="data/financial_news.txt"
num_lines=2000

with open(input_filepath, 'r', encoding='utf-8') as infile:
        lines = []
        for i, line in enumerate(infile):
            if i >= num_lines:
                break
            lines.append(line)

with open(output_filepath, 'w', encoding='utf-8') as outfile:
    outfile.writelines(lines)