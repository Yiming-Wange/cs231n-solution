# Recommended Open Course: Stanford University Computer Vision CS231n 2023
In 2023, I personally believe that if you know Python, investing some time in learning deep learning can yield significant returns. This CS231n course has been the best experience I've had among the self-studied courses from reputable institutions. The slides, notes, and assignment guides are all excellently crafted, showing the immense effort put in by the Feifei Li Lab team for this course.

This course primarily focuses on computer vision as an introductory course to deep learning. Visual inputs in deep learning, often represented as "tensors" (which might not be mathematically accurate), are mostly intuitive three-channel two-dimensional arrays. This format is both intuitive and natural for matrix operations, making it more conducive to teaching and guiding intuition, facilitating deeper learning in the future.

This article will describe the prerequisites, course content, assignment details, and recent changes over the years. It will also provide some personal insights into how to approach learning this course.

# 1. Prerequisites
1. Machine Learning: I believe having a foundation in machine learning is valuable for those planning to invest time in learning deep learning. Deep learning is a subset of machine learning, and understanding basic machine learning concepts will help you adapt to related mathematical concepts. For instance, knowing basic machine learning concepts will aid in understanding deep learning models like MLP and t-SNE. I recommend Andrew Ng's CS229 course for a comprehensive introduction to machine learning. It provides a general understanding in about 100 hours of study.

2. Python Basics: Since PyTorch is widely used for deep learning, learning Python is essential. The essence of CS231n lies in its three assignments, all of which are completed using Python. Having a foundational knowledge of programming is sufficient, and the course provides a quick introduction to Python basics.

# 2. Course Content
This course is divided into two parts. The first part covers fundamental topics, while the second part delves into what Justin refers to as "Researchy" content. It closely aligns with various topics in computer vision, offering a comprehensive and self-contained curriculum that is well-designed.

1. Fundamentals\
   - The course starts by introducing linear classification, differentiating between Nearest Neighbor and linear methods in the training process. The linear computation part (XW+bXW+b) is abstracted as a "score function." Linear classification losses (Hinge Loss and Cross-Entropy) are divided into multi-class SVM and softmax regression. This teaching approach is concise and connects the models learned in machine learning, providing a strong foundation for deep learning.
   - The course then covers optimization algorithms (order is different from 2017), explaining simple gradient descent and more advanced techniques like SGD with momentum, RMSprop, Adam, and their variants. It includes explanations of first- and second-order methods, making optimization and training concepts intuitive.
   - After introducing basic linear classification models and optimization, the course moves on to the simplest MLP. This model introduces non-linearity through activation functions like ReLU, expanding its expressive power. MLP is a natural progression from linear classification, and backpropagation is taught as a way to propagate gradients through composite functions.
   - Once MLP is covered, transitioning from linear layers (XW+bXW+b) to convolutional layers (X∗KX*K) completes the generalization from MLP to CNN. This section also provides a brief introduction to image features, including visualization techniques that highlight how CNNs extract low-level and high-level features from images. Following the basics, classic architectures like AlexNet, VGG, GoogLeNet, and ResNet are introduced, along with their parameters, complexity, and FLOPS analysis. The course also touches on transfer learning concepts.
   - Further topics include specific training techniques, covering practical aspects of neuron saturation, activation function choices, data preprocessing, parameter initialization, hyperparameter tuning, and data augmentation.

2. "Researchy" Part
   - The second part aims to use visual themes to introduce NLP-related models, starting with MLP and extending to more complex architectures like RNNs and Transformers. It introduces concepts of sequence modeling and covers tasks like image captioning using RNNs, LSTMs, and attention mechanisms.
   - NLP concepts are abstracted as "sequence" modeling, leading to the introduction of video understanding in Lecture 10. Video frame classification tasks, such as human pose detection, are covered, with main models including 3D-CNN, two-stream networks, and I3D. The course's 2023 version also includes some multi-modal content.
   - Object detection and image segmentation are discussed in one or two lectures, depending on the version. The course covers two-stage (R-CNN) and single-stage (SSD, YOLO, RetinaNet) methods. While these sections might seem challenging, they provide an overview of the topics without diving too deep into the details.
   - The visualization section is intriguing, covering weight visualization, t-SNE for feature visualization, saliency maps, adversarial attacks, DeepDream, and style transfer.

3. Extensions
   - The official extension section, "Generative and Interactive Visual Intelligence," covers modern topics, simplified for intuitive understanding. It includes concepts like self-supervised learning, robotic learning, generative models, and 3D vision.

# 3. Assignment Overview
CS231n's core lies in its three assignments. They provide guided hands-on experience, covering KNN, linear classifiers, neural networks, RNNs, LSTMs, Transformers, self-supervised learning, GANs, and more. Completing these assignments is enjoyable and helps solidify theoretical knowledge through practical application.

1. Assignment 1: Covers numpy-based implementations of KNN, SVM, Softmax, and a simple neural network. Also includes image feature tasks using HOG and histograms.

2. Assignment 2: Focuses on building neural network components using numpy, then transitioning to PyTorch. Implementing layers like convolution, pooling, normalization, and dropout. Includes a PyTorch introduction and image classification task.

3. Assignment 3: Involves building RNN, LSTM, Transformer, and GAN models using PyTorch. Implements image captioning, self-supervised learning, SimCLR, and GANs.

# 4. Conclusion and Recommendations
Approach this course centered around assignments. Watch the lectures while referencing slides, and then read slides thoroughly for a deeper understanding. References and extra readings can enhance your knowledge. Spend about 100 hours within a month to complete the course, and focus on practical coding more than extensive reading. This approach is suited for beginners like me, while experts may find different methods more suitable.
