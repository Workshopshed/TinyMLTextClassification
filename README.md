# TinyMLTextClassification
## Experiments with Tensor Flow Lite

To process a machine learning model on a microcontroller it needs to be small and fast. Tensor flow processes numbers not words so the first thing to do is to convert the data input into an array of numbers. Conventionally this is done with a big lookup table where each word is mapped to a number. I realised on a microcontroller such as the SAMD21 Cortex-M0+ 32bit low power ARM MCU used in my target board would not be able to store that table as well as rest of the code and machine learning model.

![Machine Learning Text Classification](https://github.com/Workshopshed/TinyMLTextClassification/blob/master/Machine%20Learning%20Text%20Classification.png "Text Classification Pipeline")

## Tokenisation using a hash function

My first challenge was to hook up a fast hash function to [TensorFlow TextEncoder](https://www.tensorflow.org/tutorials/tensorflow_text/intro) so that the words were encoded without that need for the lookup table.

## Shrinking the model

For the model there do seem to be some slightly different approaches to building the classifier. So I've tried a couple of variations and have been tuning the parameters. I need to get the model size down small enough to fit into the little 256 KB flash memory with space left for the rest of my code.

## Building the model

I've been using tinymlgen to export the model but I may need to dig into that and produce my own variation with more optimisations. You can follow the build process using the Juypter notebook.

[Machine Learning Notebook](https://github.com/Workshopshed/TinyMLTextClassification/blob/master/text_classification_rnn_withCustomEncoder.ipynb)
