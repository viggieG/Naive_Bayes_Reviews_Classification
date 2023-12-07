import math
import re

class Bayes_Classifier:

    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}

    def train(self, lines):
        # preprocess data
        labels, reviews = preprocess_data(lines)

        positive_class_prob = 0
        negative_class_prob = 0
        # calculate each class's prob
        for label in labels:
            if label == 1:
                negative_class_prob += 1
            else:
                positive_class_prob += 1
        negative_class_prob = float(negative_class_prob/len(labels))
        positive_class_prob = float(positive_class_prob/len(labels))
        self.class_probs = {5: positive_class_prob, 1: negative_class_prob}

        # calculate each word's 1/5 prob
        # first count each word's positive/negative frequency
        # then divided by the total count
        word_counts = {}
        class_word_counts = {5: 0, 1: 0}
        for i in range(len(reviews)):
            for word in reviews[i]:
                if word not in word_counts:
                    word_counts[word] = {}
                    word_counts[word][labels[i]] = 0
                elif labels[i] not in word_counts[word]: 
                    word_counts[word][labels[i]] = 0
                word_counts[word][labels[i]] += 1
                class_word_counts[labels[i]] += 1

        for word in word_counts:
            self.word_probs[word] = {}
            for label in [1, 5]:
                self.word_probs[word][label] = (word_counts[word].get(label, 0) + 1) / (class_word_counts[label] + len(word_counts))
        

    def classify(self, lines):
        labels, reviews = preprocess_data(lines)
        predict = []

        # calculate prob by multiplying class prob and word's prob
        for review in reviews:
            log_probs = {label: math.log(self.class_probs[label]) for label in self.class_probs}
            for word in review:
                for label in self.class_probs:
                    log_probs[label] += math.log(self.word_probs.get(word, {}).get(label, 1e-10))
            predict.append(max(log_probs, key=log_probs.get))
        
        predict = [str(i) for i in predict]
        return predict
            

# process data and clean all unnecessary punctuation
# return corresponding labels and reviews
def preprocess_data(reviews):
    preprocessed_reviews = []
    labels = []
    for review in reviews:
        review_parts = review.split('|')
        rating = int(review_parts[0])
        text = review_parts[2]
        text = text.lower() 
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        preprocessed_reviews.append(tokens)
        labels.append(rating)
    return labels, preprocessed_reviews
