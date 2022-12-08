"""
To install spaCy and the medium-sized English language model:
pip3 install spacy
python3 -m spacy download en_core_web_md
"""

# Import spaCy
import spacy

# Load the spaCy medium-sized English language model
nlp = spacy.load('en_core_web_md')

print("-------------Similarities between single words--------------")

# Determine similarities between three nlp tokens
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))      # 0.5929930274321619
print(word3.similarity(word2))      # 0.40415016164997786
print(word3.similarity(word1))      # 0.22358825939615987

"""
From the results above, I can deduce that the similarity between cat and monkey at 59% -
is relatively high, since they're both animals

The similarity between banana and monkey at 40% is understandable compared to that of banana and cat at 22%
as the former is more likely to be associated with the fruit than the latter.
"""

print("-------------Similarities between series of words--------------")

# Compare series of words with one another
tokens = nlp('cat apple monkey banana tiger dog wolf feline omnivorous')

# Double for loops compares and prints out each iteration of token1 with that of token2
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


"""
From some of the results above, the similarity result between cat and dog at 82% can be likened to the fact
that they're both domestic animals

Comparing tiger vs wolf and wolf vs monkey at about 71% and 56% respectively -
For the former with a high similarity both animals are apex predators, while the same cannot be said for the latter.

We have 100% matches when comparing similar tokens

Running the example file with a simpler language model 'en_core_web_sm' returns a 'UserWarning: [W007]...' 
which results in less accurate similarity judgements for doc.similarity method
because the small models don't ship with word vectors tensors.
"""
