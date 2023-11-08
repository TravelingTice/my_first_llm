# My First LLM

The code in this repo is all based on a 6-hour tutorial I followed on YouTube: https://www.youtube.com/watch?v=UU1WVnMk4E8

So all credits go to Elliot for creating this awesome tutorial to better understand how large language models work.

Work I've done myself to improve readability/usability:

- Separate out the different modules (model, vocab, getting training chunks, chatbot etc)
- Play with hyperparams

See plan.txt for some things I still want to do and experiment with this LLM. I only have 1 workstation (Mac) so I have to run most of the training experiments during the night, as I work on this laptop during the day.

## About the files

- **\*.ipynb:** Jupyter notebook playground files, used in toturial
- **llm:** Folder containing most of the end product GPT code
- **extract_text_corpus.py:** Script to extract text from `openwebtext` folder containing huge amount of text in xz compressed files, combines it to create a train and val split text file, used for generating random chunks in `train_data` folder
- **comment.txt:** Comment saved from python tutorial on some mac-specific settings
- **research.md:** Carrying out the things in `plan.txt` and sort of doing a bit of general research. Putting some results and notes in this markdown doc.
