# POS-Tagging-Using-Hidden-Markov-Models

This project implements a Hidden Markov Model (HMM) for part-of-speech (POS) tagging in a text dataset, using machine learning techniques. The model uses a labeled training dataset to compute transition and emission probabilities, and then utilizes the Viterbi algorithm to predict the hidden states (POS tags) in a test dataset.

## HMM Model Description

The HMM model includes the following components:

- **Initial Probabilities**: The probabilities of a certain hidden state occurring at the beginning of a sequence.
- **Transition Matrix**: The probabilities of transitioning from one hidden state to another.
- **Emission Matrix**: The probabilities of a hidden state emitting a particular observation (word).
- **Viterbi Algorithm**: The algorithm used to predict the most probable hidden states for a given sequence of observations.

### Features:

1. **Model Training**:
   - Calculates the initial probabilities, transition matrix, and emission matrix based on a labeled training dataset.
   
2. **Model Testing**:
   - Uses the Viterbi algorithm to predict POS tags on a test dataset.
   
3. **Performance Evaluation**:
   - Calculates the accuracy of the model based on the predicted tags and the ground truth tags.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hmm-pos-tagger.git
   cd hmm-pos-tagger
   ```

2. Open and run the following Jupyter notebook on Google Colab: `HMM_Project_POS.ipynb`

## Example of Forward Algorithm:

Testing the `forward` method on a randomly generated sentence:

```python
# Test the forward method on a randomly generated sentence:
print(hmm.forward(["Rudolph", "said", "Kent", "cigarettes", "are", "recently", "diagnosed", "with", "cancer", "."])[0])
```

Output:
```
3.710526622952651e-34
```
This is an extremely small number, close to zero. In the context of the forward method of the Hidden Markov Model (HMM), it represents the probability of the given sequence of words (the sentence) occurring. The model is indicating that the probability of this specific sequence under its current parameters is **extremely low**. In simpler terms, the model suggests that the sequence of words is **highly improbable**.

## Example of Viterbi Algorithm:
Testing the Viterbi method on several random sentences:

```python
# Test the Viterbi method on random sentences:
print(hmm.Viterbi(["Rudolph", "Agnew", "said", "Kent", "cigarettes", "are", "recently", "diagnosed", "with", "cancer", "."], train_set))
print(hmm.Viterbi(["Jane", "will", "spot", "Will", "."], train_set))
print(hmm.Viterbi(["Will", "saw", "Mary", "."], train_set))
print(hmm.Viterbi(["Spot", "likes", "Jane", "."], train_set))
```

Outputs:
```
['NOUN', 'NOUN', 'VERB', 'NOUN', 'NOUN', 'VERB', 'ADV', 'VERB', 'ADP', 'NOUN', '.']
['NOUN', 'VERB', 'NOUN', 'NOUN', '.']
['NOUN', 'VERB', 'NOUN', '.']
['NOUN', 'VERB', 'NOUN', '.']
```

