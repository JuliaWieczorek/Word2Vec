import preprocessing
# Defining the window for context
window = 2

# Creating a placeholder for the scanning of the word list
word_lists = []
all_text = []

# open file and add text to the list
with open('sentences') as f:
    for text in f:
        text = preprocessing.clean_text(text)
        all_text.append(text)

# Context dictionary
for sentence in all_text:
    sentence = sentence.split()
    for i, word in enumerate(sentence):
        for next in range(window):
            if i+1+next < len(sentence):
                word_lists.append((word, sentence[i+1+next]))
            if i-1-next >= 0:
                word_lists.append((word, sentence[i-1-next]))

print(word_lists)