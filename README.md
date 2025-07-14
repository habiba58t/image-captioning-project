# Image Captioning Project

This project builds an automatic image captioning system that generates natural language descriptions for images. It combines a pretrained CNN (ResNet50) for image feature extraction with an LSTM-based RNN for caption generation.

## Dataset

- **Flickr8k** dataset from Kaggle: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Contains 8,000 images, each annotated with 5 captions.

## Project Overview

This project focuses on building an automatic image captioning system that generates natural language descriptions for given images. The model uses a deep learning pipeline combining Convolutional Neural Networks (CNNs) for visual feature extraction and Recurrent Neural Networks (RNNs) for caption generation.

## Steps

1. **Data Preprocessing**
   - Load raw captions.
   - Clean captions (lowercase, remove punctuation, add `<start>` and `<end>` tokens).
   - Tokenize and pad sequences to a fixed length (max length = 37).
   - Save tokenizer for reuse.

2. **Feature Extraction**
   - Use ResNet50 to extract 2048-dimensional features from images.
   - Save extracted features as pickle files for fast loading.

3. **Model Architecture**
   - Two input branches:  
     - Image features processed with Dense, BatchNorm, and Dropout layers.  
     - Caption sequences processed with Embedding, Dropout, and stacked LSTMs.  
   - Merge branches and pass through Dense layers to output word probabilities.

   Layer (type)            Output Shape        Param #     Connected to
----------------------------------------------------------------------
input_layer_6           (None, 2048)        0           
input_layer_7           (None, 37)          0           
dense_10 (Dense)        (None, 512)         1,049,088   input_layer_6[0][0]
embedding_3 (Embedding) (None, 37, 256)     2,149,888   input_layer_7[0][0]
batch_normalization     (None, 512)         2,048       dense_10[0][0]
dropout_7 (Dropout)     (None, 37, 256)     0           embedding_3[0][0]
...
dense_13 (Dense)        (None, 8398)        2,158,286   dropout_8[0][0]

Total params: 6,608,078
Trainable params: 6,606,542
Non-trainable params: 1,536




4. **Training**
   - Use a custom data generator to produce batches of image features, partial caption sequences, and next word labels.
   - Train model with categorical crossentropy loss and Adam optimizer.
   - Use TensorFlow Dataset API to handle large data efficiently.

5. **Inference & Caption Generation**
   - Generate captions for new images using beam search with the trained model.
   - Save generated captions in a CSV file.

## Results
- Sample generated caption:  
- Image ID: 544257613_d9a1fea3f7.jpg.jpg - Caption: a closeup of a woman in a black shirt with a nose
- Image ID: 2518508760_68d8df7365.jpg.jpg - Caption: a dog is running on a track
- Image ID: 2133650765_fc6e5f295e.jpg.jpg - Caption: a boy jumps into the pool
- Image ID: 2914737181_0c8e052da8.jpg.jpg - Caption: a man in a red shirt is snowboarding over a
- Image ID: 3203453897_6317aac6ff.jpg.jpg - Caption: two people stand on a city street
- Image ID: 3708172446_4034ddc5f6.jpg.jpg - Caption: a man in a blue shirt is jumping off a wooden platform
- Image ID: 1775029934_e1e96038a8.jpg.jpg - Caption: a brown dog is running through the grass
- Image ID: 3225226381_9fe306fb9e.jpg.jpg - Caption: two dogs play together in the grass

## Technologies Used

- Python, TensorFlow/Keras  
- NumPy, Pandas, pickle  
- ResNet50 pretrained model for image feature extraction  
- LSTM networks for caption generation  
- Beam search decoding for inference

## Author

Habiba Talha  
GitHub: [@habiba58t](https://github.com/habiba58t)
