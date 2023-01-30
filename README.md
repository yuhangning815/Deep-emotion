Facial Emotion Recognition by Attentional Convolutional Network

Zhixuan Chen Haochen Yang Yuhang Ning        University of Michigan 

shczx@umich.edu yanghc@umich.edu dlning@umich.edu


Abstract

Facial expression recognition is a challenging task in the field of computer vision due to the subtle and dynamic na- ture of human emotions. In this paper, we propose the use of an attentional convolution network (ACN) to tackle the problem of facial expression recognition, based on a related paper. ACN is a deep learning architecture that incorpo- rates an attention mechanism to selectively focus on rele- vant features in the input, leading to improved performance and robustness. We trained and evaluated the proposed ar- chitecture with ACN on a dataset specificallyused for facial expression recognition. In this paper, we will discuss our improvements over an existing non-official implementation of the architecture mentioned in the paper we found.

1. Introduction

Emotions are an essential aspect of our daily life. We generate emotions when interacting with other people both consciously and subconsciously. Sometimes the expression is subtle enough not to be caught by the naked eye. How- ever, since they are inevitably expressed in one way or an- other, there will be a way to detect and recognize even the mostsubtlechangeinone’smind. Thedemandofthepoten- tial market for this application has been steadily increasing as technology keep filtratingpeople’s daily life.

ACN is a deep learning architecture that incorporates an attention mechanism to selectively focus on relevant fea- tures in the input, leading to improved performance and ro- bustness. The attention mechanism allows the network to learn to focus on different regions of the face depending on the facial expression, making the recognition process more efficientand accurate.

Thetestresultsofthearchitecturethatweintendtoreim- plement show that ACN gives a relatively high accuracy, demonstrating its effectiveness in facial expression recog- nition.

![](Aspose.Words.e3386b45-e28f-4efe-8087-94edc1c3522b.001.png)

Figure 1. Spacial Transformation

2. Related Work

In one of the recent works by Zeng et al.[1], they dis- covered a robust algorithm in case of an annotation error. To address the inconsistency, they proposed an Inconsis- tent Pseudo Annotations to Latent Truth (IPA2LT) frame- work to train a facial expression recognition model from multiple inconsistently labeled data sets and large scale un- labeled data. In IPA2LT, they assigned each sample more than one labels with human annotations or model predic- tions. In our project, we removed part of the controversial data points. Since people have a neutral expression for most of the time, we downsized the disproportionally large set of neutral labels to avoid biases. Zeng et al. also brought up and addressed a problem with facial expression recognition methods, which arose when training sets are enlarged by combining multiple datasets because of the subjectivity of annotating facial expressions. We took note of this problem and modified our training data accordingly to avoid such issue.

Another novel framework that yields accurate detection results was proposed by Zeng et al.[2] This model achieves a staggering accuracy of 95.79 percent on the extended Cohn–Kanade (CK+) database for seven facial expressions including neutral, beating majority of the similar existing models. This model takes use of the deep sparse auto- encoders (DSAE) to compose a high-dimensional feature composed by the combination of the facial geometric and appearance features.



![](Aspose.Words.e3386b45-e28f-4efe-8087-94edc1c3522b.002.png)

Figure 2. Architecture

2

The work of Cowie et al.[4] and Edwards et al.[5] high- lighted significant improvements on emotion recognition methodologies via multichannel detection including facial expression. This paper is based on and modified from the research of Minaee et al.[3] They first proposed the use of attentional networks to extract key feature points with small number of hidden layers and visualized the part of face that the attentional neural network that is important to facial ex- pression detection. We found a primitive implementation of the paper on GitHub. However, there are several significant flaws in its implementation, which resulted in a very low test accuracy. We took part of its implementation and mod- ified majority part of it and got a better result on both our training set and testing set.

3. Method

A common way to improve accuracy of a neural network is to add more layers and increase the number of neurons in a hidden layer. However, due to the small dimensionality of the output layer, it is possible to achieve impressive re- sult using less than 10 layers with our proposed framework based on attentional convolutional neural network.

Key feature abstraction is especially crucial for facial ex- pression recognition since only a small set of positions on someone’s face is responsible for making a facial expres- sion.

Based on this assumption, we added spatial transfor- mation (Figure 1) to our network architecture to train our model to recognize key feature points. According to the pa- per of Minaee et al., the architecture design is as shown in Figure 2 above. This design is composed of two main parts: the localization network (spacial transformer) and the fea- ture extraction. A spatial transformer layer is a type of layer in a neural network that enables the model to perform spa- tial transformations on the input data. It consists of three components: a localization network, a grid generator, and a sampler. The localization network produces transforma- tionparametersbasedontheinputdata, whicharethenused by the grid generator to generate a grid of output locations. The sampler uses the generated grid and the input data to

produce the output of the spatial transformer layer. This layer allows the model to learn to perform spatial transfor- mations in an end-to-end manner, which can make it more flexible and better able to adapt to different tasks and input data.

In the original paper, the feature extraction part includes four convolutional layers followed by a max-pooling layer and a rectified linear unit (ReLU) activation function. To improve accuracy, we added an additional layer to the fea- ture extraction part, followed by a dropout layer and two fully-connected layers. The spatial transformer (localiza- tionnetwork)consistsoftwoconvolutionallayers(eachfol- lowed by a max-pooling layer and a ReLU), and two fully- connected layers. After estimating the transformation pa- rameters, the input is transformed onto the sampling grid T (θ) to produce the warped data. The spatial transformer module focuses on the most relevant part of the image by estimating a sample over the attended region. In this case, an affine transformation is used for the warping and crop- ing, as it is commonly used in many applications. For more information about the spatial transformer network, please see the original paper.

This model is then trained by optimizing a loss function using stochastic gradient descent approach (i.e. Adam opti- mizer). The loss function in this work is simply the classi- ficationloss (cross-entropy).

Also, we manually adjusted the hyperparameters includ- ing batch size, the number of epochs and the learning rate to prevent overfitting.

4. Experiment

In this section we will discuss detailed implementation of our model structure, experiment input data set and result- ing accuracy. We will then provide an analysis of the per- formance and possible room for improvement. Our training and test data come from FER2013 (Figure 3), a database of 35,887 facial images created by using Google image search API. The FER2013 dataset includes images with 7 emo- tions in total: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral. Compared with other datasets, FER2013 con-



![](Aspose.Words.e3386b45-e28f-4efe-8087-94edc1c3522b.003.png)

Figure 3. FER2013

tains more variations in its data, including partial faces and low-contrast images.

To further improve our result, we made several adjust- ments for both the model in the existing GitHub implemen- tation and the dataset. Regarding the model architecture, we corrected the order of the spatial transformer and the feature extraction network. We also added a hidden con- volutional layer in the feature extraction network, as men- tioned above. For the dataset, we removed controversial in- put images such as the ones that are blurry, and the images that has ambiguous meanings. Additionally, since the num- ber of images for disgusting, surprised, and fear was very low compared with others, we decided to remove the im- ages with these expressions to avoid biases. This means the model we implemented focuses on classificationof the four remaining emotions. After data cleaning, we divided the dataset into three parts: a training set with around 16500 images, a validation set with around 3500 images and a test set with around 3000 images. Finally, with 120 epochs, a batch size of 128 and a learning rate of 10−5, we managed to achieve a test accuracy of 64.3 percent.

4.1. Analysis

Initially, we trained and tested the model directly using the GitHub non-officialimplementation. However, we only got a test accuracy of 48.7 percent. After comparing the model in the GitHub implementation with the model in the original paper, We found a flaw in the non-official imple- mentation. Specifically, in the non-officialimplementation, the spatial transformer and the feature extraction network run sequentially instead of parallelly. After fixing this is- sue, the accuracy was improved to 57.8 percent, which is still not high enough. By adding an extra convoluntional layerinthefeatureextractionnetworkanddoingdataclean- ing, we achieved a test accuracy of 64.3 percent. We wrote codes to visualize the prediction results, as shown in Figure

4. This accuracy is lower than the accuracy of 70 percent in the original paper. We tried different approaches such as adding more layers or adjusting hyperparameters to im- prove the result, but it turned out that we were unable to achieve a higher accuracy. As the authors of the original paper did not provide more details about the model, there must be something we missed in our implementation. We hope that we can improve the model in the future.

![](Aspose.Words.e3386b45-e28f-4efe-8087-94edc1c3522b.004.png)

Figure 4. Predictions

5. Conclusion

In this paper, we reimplemented an attentional model for facial expression recognition. We managed to get a test accuracy relatively close to the accuracy mentioned in the original paper. We improved the performance of the exist- ing non-officialimplementation on GitHub by changing the sequential structure into parallel, adding one extra convolu- tional layer, and cleaning the dataset.

Facial expression recognition is a technology that can be used in a variety of applications. For the future use of facial expression recognition technology, people need to be cau- tious of its use. The technology is powerful but also dan- gerous. Government handling this technique could use this power to guess the mentalities of people which would help them to do righteous works, but villians can also use this technology to harm the society. Therefore, people should still set their own standard of identifying a facial expres- sion instead of solely relying on this technology. The re- searchers should also be aware of the ethical aspects of the facial expression recognition technology. The database im- ages should achieve equality in race, gender, and emotion, and the technology should be inherently non-biased.


6. Reference
1. Jiabei Zeng, Shiguang Shan, and Xilin Chen. EECV 2018, “Facial Expression Recognition with Inconsistently Annotated Datasets”.
1. Nianyin Zeng, Hong Zhang, Baoye Song, Weibo Liu, Yurong Li, Abdullah M. Dobaie. “Facial expression recognition via learning deep sparse autoencoders” Neurocomputing, Volume 273, 643-649, 2018.
1. Minaee, Shervin, Mehdi Minaei, and Amirali Abdolrashidi. “Deep-emotion: Facial expression recognition using attentional convolutional net- work.” Sensors 21.9 (2021): 3046.
1. Cowie, Roddy, Ellen Douglas-Cowie, Nico- las Tsapatsoulis, George Votsis, Stefanos Kollias, Winfried Fellenz, and John G. Taylor. “Emo- tion recognition in human-computer interaction.” IEEE Signal processing magazine 18, no. 1: 32- 80, 2001.

3

[5]Edwards, Jane, HenryJ.Jackson, andPhilippa E. Pattison. “Emotion recognition via facial ex- pression and affective prosody in schizophrenia: a methodological review.” Clinical psychology review 22.6: 789-832, 2002.
4
