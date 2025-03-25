# Melanoma-Skin-Cancer-Detection
## Problem Statement
In this assignment, you will build a multiclass classification model using a custom convolutional neural network in TensorFlow. 

 

Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


### The data set contains the following diseases:

Actinic keratosis
Basal cell carcinoma
Dermatofibroma
Melanoma
Nevus
Pigmented benign keratosis
Seborrheic keratosis
Squamous cell carcinoma
Vascular lesion

### Data Sample Visulisation

![image](https://github.com/user-attachments/assets/686153f8-b4cd-439b-b48e-8a5d4a69069e)


### The first Model has below mentioned results :

The model training accuracy increased from 20 to 87 % but validation accuracy remains low about 50%
As training accuracy is high it seem there is mdoel overfit.
To manage overfitting , let;s use augumentation tech.

![image](https://github.com/user-attachments/assets/f7b1af5d-4d14-40d4-a4be-c35b4bd3261e)


By using Augementation technique we were able to manage overfitting , the validation accuracy increased a bit but the training accuracy decreased drastically signifies underfitting.

![image](https://github.com/user-attachments/assets/b6bd6675-0013-44c1-972d-d38eccd22f2d)

We identified  and removed data imblances from training sample.

![image](https://github.com/user-attachments/assets/6967b053-883d-46e1-b017-c18d02eaa848)

We used Augementor for same.

	count
Label	
pigmented benign keratosis	962
melanoma	948
basal cell carcinoma	876
nevus	857
squamous cell carcinoma	681
vascular lesion	639
actinic keratosis	614
dermatofibroma	595
seborrheic keratosis	577

dtype: int64


We than ran model with normalisation.We increased epoch to 50

![image](https://github.com/user-attachments/assets/4ee09853-53d3-448c-a3aa-0a7235a072d7)

Now after removing imbalance and using Normalisation in model , the training accuracy increases gradually but the validation accuracy is not stable ven though on overall graph moves with training accuracy. Now we can see there is reducing in overfitting but there is still scope of improvment.





