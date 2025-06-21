Image Captioning using Deep Learning

Developed by: Preethi Verma
B.Tech in Data Science
Vidyashilp University

Project Overview:

This project focuses on generating meaningful natural language captions for images using deep learning. It uses a Convolutional Neural Network (CNN) as an encoder to extract image features and a Recurrent Neural Network (LSTM) or Transformer as a decoder to generate human-like descriptions.

Objective:

Train a model that can automatically generate captions for input images.

Use the MS COCO 2014 dataset to learn rich representations of visual scenes.

Build an end-to-end captioning system with a simple UI.

Tech Stack:

Programming Language: Python
Deep Learning: TensorFlow, Keras, PyTorch
Image Processing: OpenCV, PIL
Natural Language Processing: NLTK, Tokenizer
Interface: Streamlit or Flask
IDE: Jupyter Notebook

Dataset:

MS COCO 2014 Dataset
Over 330,000 images with more than 200,000 labeled
Each image has 5 human-written captions

Model Architecture:

Encoder - CNN like InceptionV3 or ResNet50 to extract image features
Decoder - LSTM or Transformer to generate captions based on the image context

Evaluation Metrics:

BLEU Score
CIDEr
ROUGE
METEOR

How to Run:

Clone the repository

Install required libraries using pip install -r requirements.txt

Download and prepare the COCO dataset

Run the training script (train.py)

Launch the application using streamlit run app.py

Results:

The model generates meaningful and contextually accurate captions for unseen images. It shows promising results on standard metrics and supports real-time testing via a web interface.

Learning Outcomes:

Gained understanding of deep learning for computer vision and NLP integration

Learned how to use CNNs, LSTMs, and Transformers effectively

Built and evaluated models using real-world data and metrics

Acknowledgements:

Special thanks to the Zidio Development team and mentors for their support during this internship.

License:

This project is shared for academic and learning purposes.
