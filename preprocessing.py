import re

def clean_text(string):
    # a method to clean text

    # converting the text to lower
    string = string.lower()

    # cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()

    stop_words = ['the', 'a', 'and', 'is', 'be', 'will']
    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])

    return string