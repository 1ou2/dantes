# dantes
Finetuning a model to act as Edmond Dantès

# Gutenberg file processing
- download the 4 files corresponding to the text
- remove UTF-8 BOM
- remove header and footer
- reformat text and remove unecessary line breaks around column 80

Output of this file is a clean formatted file in the data/gutenberg folder