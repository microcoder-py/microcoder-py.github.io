# Word Embeddings: The Magic That They Are, and Why GANs Can't Process Them
*Feb, 2021*

NLP is an astonishing field of research. Human language, as taught to us in schools, in college, even in theory of computational grammar, isn’t one that can simply be replicated by inanimate elements - it demands a sense of creativity, of romanticism, and yet, NLP has managed to overcome several of these challenges, rendering computational devices able to understand, generate, translate, and build new mechanisms for language modelling. 

All of this was possible because of a simple idea - word embeddings. Something that feels intuitive, something that makes sense, but when I tried to understand online why we use it, I could not find many satisfactory articles. Here, I will try and explain what I understand about word embeddings, why they are important, and why NLP still faces several important hurdles to be overcome, in part due to the structure of language itself, and in part because of the way these systems have been engineered. 

## What is distance?

Let’s think of a dataset of numbers. Say, the first ten natural numbers, i.e. [0, 9]. If I were to ask you the distance between 1 and 2, you might say it is 1. In a sense, independent of what distance metric you’re using, it is correct. The distance of 1 from 0 is also 1. There is a clear, continuous space with even divisions. 

In the case of images as well, assuming we have a single channel (black and white) image, every pixel has a fixed range of values, say [0, 255] and the distance from the origin (black being 0) is well defined. Therefore, the overall distance of an image, what we otherwise call the ‘loss’ of an image when compared with another image is a concept easier to grasp - every pixel is a certain distance away from what its true value should be, so if we add certain adjustments to each pixel, our source image distribution can be much more close to the real distribution. The same idea can be extended to multiple channel images as well, the concept remains the same, only the number of variables to deal with increases.

However, can you tell me the distance between the following statements? 

| Sentence 1 | Sentence 2 | 
|---|---| 
| This is a sentence | This is also a sentence | 

Despite the statements being rather similar, as I am sure you can tell from your several years of experience with language, how do you inform a computer how similar or dissimilar they are? What representation would make a computer understand what the sentences are?

## And our saviour, is the concept of word embeddings

To create a sense of distance, or similarity, we need to convert these sentences to a different, numerical space. This is partly because as engineers or mathematicians, numbers and quantification make sense to us - neural networks are mathematical machines, our loss functions need numbers, our optimisers need differentiable functions, who wants to re-engineer decades worth of research? - but also because it is one of the most concise, yet accurate way of ensuring that we do not need to build dedicated rules for grammar at each step for each word, the numbers themselves will do the magic. This feels a little like a magician’s conjuring, but bear with me, things are about to get interesting. 

Let’s consider the above sentences. If we were to consider all the unique words in the above (*in NLP terminology referred to as tokens*), we would have the following

| Token | Token  |Token |Token | Token  |
|:---:|:---:|:---:|:---:| :---:|
| This  | is |a|sentence| also |

If we were to assign a single value to each token on the number line, let’s say as below

| This  | is |a|sentence| also |
|:---:|:---:|:---:|:---:| :---:|
| 0 | 1| 2| 3| 4|

It wouldn’t make much sense. We still do not have any real relationship between the tokens. We cannot say that the token ‘this’ is at a distance of 2 units from the word ‘sentence’. Why? Because we simply projected the tokens onto a new, continuous space, without actually understanding what the distance between them is, or how these are even related to each other. 

Okay, so this clearly isn’t working. What we need to do is capture the relationship between the tokens, not just assign them random numbers and hope for the best. So how do we understand what relationship individual words have with each other? We turn to our good old friend, the neural network.

> *Please note that I will not be discussing specific techniques for word embedding such as [Word2Vec](https://arxiv.org/abs/1301.3781) or [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), I am only explaining the intuition behind word embeddings. You may read about different types of word embeddings online.* 

Our neural network in this case will assign some random numbers to each token. During the training process, it will somehow learn how to assign the right number on the number line to each token, continuously readjusting the values with every batch iteration until the representations somehow are able to optimise the target loss function. Now, the numbers being used represent a true relationship of relative distance between each other. Note that I have used the term ‘relative’, we will get back to this later. 

In this example, we have only considered word embeddings of unit dimension, i.e. along the number line. In practice, our databases will not contain 5 tokens, you are likely to deal with over 100k tokens even in the smaller datasets, and more than a million in production. It is foolhardy to assume a single number will be able to capture the essence of relationships between different words. 

So instead of projecting the tokens to a unit dimension, we project them onto a vector space of several dimensions, typically between 150 - 300 (based on empirical observations). What this means is that every word is represented as a vector of numbers. If the token ```this``` were to be represented as a vector of four dimensions, an example representation would be

| token   | embedding| 
|:---:|:---:| 
| this              | [0.1221, -4.563, 0.009, 3.789] | 

This might seem a little like trickery, and in a sense, it is, but here are some salient features about word embeddings, which will hopefully help you like them better.

### No need to model language grammar rules, the embeddings will help represent it

Once you have trained a network and obtained the embeddings, you do not need to inform the network what specific grammatical rules it must follow. Assuming that the dataset was large enough to capture almost all scenarios of grammatical possibility, and that the model itself was conditioned well, we can assume that the word embeddings are able to capture the essence of the grammatical rules of the input language. 


### A mathematical sense of distance

Now that the embeddings are at the appropriate approximate relative distances from each other, we can create any measure of distance to compare between any two vectors. Whether you choose cosine distance, ratio analysis or any other technique, the relative distances remain consistent, so we can now use these embeddings to calculate all that we wished to calculate

### This is not absolute distance, however

One point to note here, word embeddings do not capture an absolute sense of distance - the distances are relative to the other words in the vocabulary. What this means is that the distance from the origin for any token is not a concept with validity. It is not the same as saying that a pixel’s red value is 164 levels from the origin of 0. The concept of addition too is not a valid one here, the embeddings of two different tokens cannot be added to yield a third token that is sensible, until and unless we balance the equations with relative distance. 

For example, the embeddings for the tokens ```king``` and ```queen``` will have the same cosine distance as the terms ```him``` and ```her``` or ```male``` and ```female```. This is because the embeddings have been trained such that all male entities have the same distance from their female counterparts. 

The equation ```embedding(king) - embedding(queen) = embedding(him) - embedding(her)``` would also hold true, because the relative distances between them remain the same. However, the entity ```embedding(king) - embedding(queen)``` in itself holds no meaning. It is merely the relative distance between the token ```king``` and ```queen```

This third feature is extremely important to note, because it is also the reason we cannot use word embeddings directly in Generative Adversarial Networks. GANs need a continuous space to operate upon and adversarially train their network, which as I have already highlighted above is a simple operation to execute upon in the case of images and consequently video, also on audio, but not on text, since the concept of absolute distance does not work with word embeddings. This was also clearly outlined by the inventor of GANs, Ian Goodfellow in the following [reddit thread](https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/) 

Several interesting attempts have been made, however, by trying to convert discrete word embeddings into a continuous space using autoencoder networks such as the [TextKD Gan](https://arxiv.org/abs/1905.01976) among others. For anyone feeling adventurous, feel free to read this survey article on [usage of GANs in NLP](https://arxiv.org/pdf/1705.10929.pdf)  



