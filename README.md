# DeepLearning_CS677
Deep Learning codes in python during CS677 course period
-in kong server - NJIT datasci node
https://web.njit.edu/~usman/courses/cs677_spring20/index.html

Assignment1:
CUDA program for computing the dot product of a vector in parallel with each row of a matrix, each thread access consecutive memory locations (coalescent memory access), with inputs as number of rows, number of columns, a data matrix, a vector file (one row), cuda device, number of threads.

Assignment2:
Convert the CUDA program that you wrote for assignment one into an OpenMP program.

Assignment3:
Python program to train a single layer neural network with sigmoid activation, use numpy, input in dense liblinear format which means you exclude the dimension and include 0's, and n is the number of nodes in the single hidden layer. Implement gradient descent. Answer the following.
1. Does your network reach 0 training error? 
2. Can you make your program into stochastic gradient descent (SGD)?
3. Does SGD give lower test error than full gradient descent?
4. What happens if change the activation to sign? Will the same algorithm work? If not what will you do?

Assignment4:
Implement stochastic gradient descent in the back propagation program (assignment 3). Perform the mini-batch SGD search. Same input and output as assignment3. batch size k as input, leave the offset for the final layer to be zero.
1. Test the program on breast cancer and ionosphere given on the website. Is the mini-batch faster or the original one? accuracy?
2. Is the search faster or more accurate if you keep track of the best objective in the inner loop?

Assignment5:
Python program that trains a neural network with a single 2x2 convolutional layer with stride 1 and global average pooling. Use sigmoid activation function. The input are 3x3 images. Images for training and testing are in seperate directories.
1. What is the convolutional kernel learnt by your program? 

Assignment6:
Build a convolutional network in Keras to train the Mini-ImageNet dataset. Create a network that achieves at least 80% test accuracy. Take three inputs: the input training data, training labels, and a model file name to save the model to. Save a Keras model to the file The output is the test error of the data which is the number of misclassifications divided by size of the test set.

Assignment7:
Write a convolutional network in Keras to train the Mini-ImageNet dataset. Use transfer learning to achieve above 90% accuracy on the test/validation datasets. Same input and output as assignment6.

Assignment8:
Classify images in the three Kaggle datasets with convolutional networks. Use transfer learning to achieve above 85% accuracy on the test/validation datasets. Same input and output as assignment6.

Assignment9:
Implement a simple GAN in Keras to generate MNIST images. Use the GAN https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f as your discriminator and generator. Train the generator to produce images of numbers between 0 and 9. Take one input: the generator model file. The output should be images resembling MNIST digits saved to the output file.

Assignment10:
Implement a simple black box attack in Keras to attack a pretrained ResNet18 model from Keras. Use a two hidden layer neural network with each layer having 100 nodes for the substitute model. Gnerate adversaries to decieve a simple single layer neural network with 20 hidden nodes into misclassifying data from a test set provided, consisting of examples  from classes 0 and 1 from CIFAR10. Target model should have at least 85% accuracy on the test set withoutadversaries. Successful attack should have a classification accuracy of at most 10%on the test. Take three inputs: the test data, the target model to attack (in our case this is the network with 20 hidden nodes), and a model file name to save the black box model file to. Output the accuracy of the target model on the test data without adversaries as the first step, verify model accuracy on the test data without adversaries (Note-harder to attack if lower accuracy). Output the accuracy of the target model on the adversaries generated from the test data after each epoch. Take three inputs: test set, target model, and the black box model.
The output should be the accuracy of adversarial examples generated with epsilon=0.0625.

Assignment11:
Learn a word2vec model from fake news dataset and a real news dataset, make use of the Python Gensim library. Output the top 5 most similar words to the following ones from each representation.
1. Hillary
2. Trump
3. Obama
4. Immigration
First normalize all vector representations (set them to Euclidean length 1). Consider the vector x for a given word w. We compare the cosine similarity between x and the vectors x' for each word w' in the fake news dataset first. We then output the top 5 words with highest similarity. We then do the same for the real news and then see if the top similar words differ considerably. Take two inputs: the text dataset on which to learn the words and a model file name to save the word2vec model to.
Are the most similar words to the queries considerably different from the fake and real news datasets? 
