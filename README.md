# Final Project - Artist Generation

Brian Henriquez, bmhenriq@ucsd.edu

## Abstract Proposal

Given that I had the most luck when generating quotes from the r/Quotes subreddit, I thought I could retouch this idea but in a different approach. Originally, the intention was to mimic a user and simply generate quotes that other users may find interesting. However, given how we have already learned how to generate images, I felt that this project could use some visual aids. These visual aids would make more sense if we were to consider the people who uttered the quote instead. That is, instead of being interested in the user space, we are now more interested in the author space and what these quotes can tell us about them. Seeing that networks can generate iamges from text, I felt that a perfect use for such a network would be to generate images based on generated inspirational quotes. These images would then serve as a visual parallel to the quotes.

The style transfer portion of this project was actually part of the motivating factor. While AttnGAN does indeed generate images from any given input text (that has some embedding), it does not generate images that are stylistically interesting (unless trained on such images). Generating a regular dataset large enough to satisfy training the network was out of the question, so instead a neural style transfer algorithm was used in order to make the results more visually appealing. The finalized project was a success as the generated images are indeed interesting to observe. The outputs have been provided in this repository and will be presented in the form of a presentation in the delivery date.

Note that while a lot of these networks were covered in class, what I really sought was the novelty in the way that these networks were chained together. This pipeline can essentially create hundreds of "artists" and their respective artworks and a quote describing them. Instead of simply generating artwork, this network first generates the person and then the artwork. This is what makes the network more interesting than most others...

## Project Report

The report can be found [here](./ECE188Report.pdf).

## Model/Data

There are quite a few files in this repository due to it containing four different networks, but all networks should already contain their requisite weights. To touch on all networks briefly, the two RNNs used have already been trained and can have their weights loaded in their respective notebooks. The style transfer network needs no training as it is done during generation. AttnGAN does require some weights, but these are already provided and the proper configuration settings have been set so evaluation is possible.

Interestingly enough, there is no input data other than the style transfer seed images! These images can be found inside the [StyleTransferDefs directory](./StyleTransferDefs). These are all well known portraits and are all credited inside the PowerPoint presentation.

To summarize the pipeline, a char-RNN pretrained on the /r/Quotes subreddit generates quotes and their respective authors from scratch. This file is then parsed and the quotes are used for the generation of base images using AttnGAN. The quotes are then also passed into a word-RNN for sentiment analysis. These "moods" are conveyed by the seed images discussed earlier and are used as the style objective for the base image in a style transfer network. These outputs are then combined in order to show all of the results in a single image. 

## Code

In order to train the networks included here, there are various things that need to be done:
- SentimentRNN
  - Jupyter Notebook: [Sentiment Analysis Training](./Sentiment%20Analysis%20Training)
- AttnGAN
  - Cannot currently be trained in its form as this project uses only the evaluation API.
  - Consult the respective [respository](https://github.com/taoxugit/AttnGAN)
- charRNN (Quote Generation)
  - Jupyter Notebook: [Quote Generation using char-RNN](./Biased%20Quote%20Generation%20using%20GRU%20RNN.ipynb)
- Neural Style Transfer
  - Training is performed at the same time as evaluation.

In order to generate the original RNN outputs (examples of which can be found in ExampleRNNOut/), use the following Jupyter Notebook:
- [Quote Generation using char-RNN](./Biased%20Quote%20Generation%20using%20GRU%20RNN.ipynb)
  - Generation occurs in the second half of this notebook!

In order to generate text to image output, use the following Jupyter Notebook:
- [Text to Image Notebook](./Text%20to%20Image.ipynb)

## Results

All of the example results have already generated and can be found inside the Portraits directory [here](./Portraits). This includes the generated image along with the quote and author that were used in order to generate the image. Furthermore, a presentation made using PowerPoint that also includes the original seed style images can be found within the repository [here](./Artist%20Generation.pptx).

## Technical Notes

This code was run on my personal computer with a 1080Ti for training. The specs of this PC are close enough to the cluster that there should be no issue when training or evaluating in the cluster. However, there are a few packages that will be needed by the various networks:
- Torch & Torchvision (for AttnGAN)
- TensorFlow (for char-RNN & sentiment RNN)
- edict (for AttnGAN)
- tqdm (for progbars)
- scikit-image (for AttnGAN)
- sklearn (for sentiment RNN)
- PRAW / PSAW (for char-RNN) [This should be present within the notebook already]

## Reference

The following were used as resources for the completion of this project:
- [AttnGAN](https://github.com/taoxugit/AttnGAN)
- [AttnGAN non-Flask Evaluation](https://github.com/bprabhakar/text-to-image)
- [SentimentRNN](https://github.com/omarsar/nlp_pytorch_tensorflow_notebooks)
- [PSAW - Pushshift API](https://pushshift.io/)
- [Neural Style Transfer Implementation](https://github.com/roberttwomey/ml-art-bootcamp)
- [char-RNN Implementation](https://github.com/roberttwomey/ml-art-bootcamp)

A big thanks to Robert Twomey for nicely compiling many networks for ease of testing and use!