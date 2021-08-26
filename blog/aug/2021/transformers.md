## What is the hullabaloo with Transformers anyway?

One of the core concerns surrounding the typically used RNN models in NLP is the inability of the model to be parallelised. The architecture, by virtue of being sequential, cannot be distributed across multiple systems, thereby not allowing us to make the full use of heavily distributed computational machinery that we have access to. 

Then there’s also the concern of vanishing and exploding gradients. While not unique to RNNs, this has been a fairly prevalent problem even after introducing novel architectures and other methods for mitigation.

Third is the issue that sequential data works best with RNNs. This is true of many data types, audio, text, video, time series, pretty much anything that has anything to do with a sequence. 

A new architecture itself was needed that could resolve the above problems.

## Introducing the Transformer

What do you do when you need to build a new architecture, that solves the above problems? Well, as researchers at Google and University of Toronto found in their era-defining paper, [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017), what you need is a model that

Has massive capacity, but with only sequential operations. No, seriously. Massive.
Makes use of attention each time, every time, over multiple processing heads.
Looks simple at first glance, but every time you read the paper again reveals how nuanced and intricate the design is

Obviously, the above is an oversimplification from the eyes of a young man awestruck by how brilliantly designed this network is. Let’s dive a little deeper into the details. 

> *NOTE: All images used here have been picked up directly from the research paper cited above. These images were not generated by me*

## Alright, so I need Attention. What is attention, again?

RNNs were used because they knew how to remember what is important when. They knew how to remember that if we are looking at a word at a current timestep, there had to be a word in the past related to it. For example in the sentence ```His name is Adrian```, the RNN knew that ```his``` refers to ```Adrian```, and that ```name``` too refers to ```Adrian```. RNNs do this by maintaining an internal hidden state. 

![Img](https://lilianweng.github.io/lil-log/assets/images/sentence-example-attention.png)
*Example of Attention*

Of course, this became an issue once the RNN had provided its output to another RNN, particularly in encoder-decoder architectures. For instance, in the famous publication about the first encoder-decoder architecture, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (2014), the encoder-decoder architecture was able to translate languages very well, but the performance was still not optimal. They were passing the final state of the encoder to initialise the decoder states so that there would be some semantic information retained, but clearly it wasn’t enough. 

That was until the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014) came out. In this, in between the step where they pass the output of the encoder and decoder, an attention layer was used. While this attention layer isn’t the same as the one used in Transformers (more on this later), it proved that attention could definitely be a powerful tool to make use of in encoder-decoder architectures. 

> *Please note that, for our case, we only speak about self attention, and each time we say attention we mean self attention*

>*The major difference between attention and self attention is that attention is the general case that applies to two different entities, while self attention applies attention to the same entity*

So, what is attention? Attention is basically a matrix that remembers how much correlation every word in the sentence has with every other word. Like in the example above, ```adrian``` would be highly correlated to ```name``` and ```his``` but ```his``` would not be highly related to ```name```. Yeah, that’s pretty much it. This information is important since we cannot capture it via hidden states the way RNNs can. For more examples, you may refer to this amazing blog by [Jay Alammar](http://jalammar.github.io/illustrated-transformer/) or the one by [Lilian Weng](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#:~:text=The%20attention%20mechanism%20was%20born%20to%20help%20memorize%20long%20source,and%20the%20entire%20source%20input.)

## Can we move on to the transformer now please? 

Okay so like any other encoder-decoder architecture, the Transformer network too makes use of a similar idea - the input is passed through the encoder, the encoder generates some output which it passes to the decoder. The decoder starts with an empty string, i.e. the ```<START_TOKEN>```, passes it through the decoder, generates the next token, adds it to the sentence, and keeps predicting new tokens until it either hits the ```<END_TOKEN>``` or reaches maximum output sentence length. 

> *This property of the Transformer, that it generates one token at a time to predict the next statement is called the AUTOREGRESSIVE property - it predicts future statements basis its own predictions of past timesteps*

> *When reading the section on decoding, please pay particular attention. If you are TensorFlow programmers like me, you will more than likely run into the same confusion I had after reading the official documentation*  

Now, the encoder does not only pass the output - it also needs to pass other semantic information. In the case of Seq2Seq, they passed the initial RNN state. In the case of Seq2Seq with attention, they passed the initial RNN state for the decoder as well as the attention layer. We will discuss what kind of semantic information is passed via the encoder later, but remembering that there is some kind of extra information being passed is important - it is one of the key innovations in the transformer network.

## The Encoder Network