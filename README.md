# TinyMLTextClassification
## Experiments with Tensor Flow Lite

To process a machine learning model on a microcontroller it needs to be small and fast. Tensor flow processes numbers not words so the first thing to do is to convert the data input into an array of numbers. Conventionally this is done with a big lookup table where each word is mapped to a number. I realised on a microcontroller such as the SAMD21 Cortex-M0+ 32bit low power ARM MCU used in my target board would not be able to store that table as well as rest of the code and machine learning model.

So my thought was to try with a hash function that could be easily reproduced on both the Python training environment and over on the Arduino MKR Zero.

![Machine Learning Text Classification](https://github.com/Workshopshed/TinyMLTextClassification/blob/master/Machine%20Learning%20Text%20Classification.png "Text Classification Pipeline")

## Tokenisation using a hash function

My first challenge was to hook up a fast hash function to [TensorFlow TextEncoder](https://www.tensorflow.org/tutorials/tensorflow_text/intro) so that the words were encoded without that need for the lookup table. I used the [Super Fast Hash](http://www.azillionmonkeys.com/qed/hash.html) by Paul Hsieh which has ports for Python and C.

## Shrinking the model

For the model there do seem to be some slightly different approaches to building the classifier. So I've tried a couple of variations and have been tuning the parameters. I need to get the model size down small enough to fit into the little 256 KB flash memory with space left for the rest of my code.

## Building the model

I've been using tinymlgen to export the model but I may need to dig into that and produce my own variation with more optimisations. You can follow the build process using the Juypter notebook.

[Machine Learning Notebook](https://github.com/Workshopshed/TinyMLTextClassification/blob/master/text_classification_rnn_withCustomEncoder.ipynb)

## Problems running the model

Still a key outstaning issue is that this text classifier is failing to load onto the Arduino MKR board. I think the issue is that it is too big to run in memory.

## Optimisation

Did some experiments with the optimisations.

```
optimizers = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

results in 

```
Initialising...
Type FLOAT16 (10) not is not supported
Failed to initialize tensor 1
MicroAllocator: Failed to initialize.
AllocateTensors() failed
```

Tried also:

```
# From TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power
def representative_dataset_gen():
    for value in test_dataset:
        yield np.array(value,dtype=np.dtype((np.float32,8)),ndmin=2)

...

optimizers = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```
but could not work out how to get a generator to produce data in the right way

## New Model

A simpler model was created using the raw USB data rather than text. This avoids the issues of encoding and allows for simpler models to be tried.
