# MSc_Thesis-Classify-Objects-Using-PREMONN-Technique-and-small-Neural-Networks
The first step is to create the Neural Networks that we are going to use. You can use your own NN if you want. At this part we are training NN to learn the curves that exists on the shapes. The corresponing file is completeTrain

After we create the NN we are using them to define the curves that appear on an object and thus to classify it. The experiments were done using the COIL20 database which you can find here: http://deeplearning.net/datasets/

We are using two techinues: By comparing the Histogramm and the Strings

#1
We use all the NN to predict the next value of the curv. The one that makes the best prediction wins and we note this info. Finally we calculate the Histogram that shows the percentage of the NN that won.
HistTrain: We group some images and calculate their mean-histogram. This represents this group. We crate all the basis-hists this way
HistTest: To classify an image we calculate its histogram and compare it with all the basis-histograms. This part it the classification

#2
Instead of calculating the histogram we can compare directly the string that shows which NN wons. To do it efficiently for every test image we need to calculate all it's possible string-shifts and compare all of them with the basis-strings
stringTrain: We first create the basis-strings
stringTest: This is the classification. We calculate the string for the image to classify and then compare it with all the basis strings.

The above files are using the functions that are shown below:

FUNcalcKampParametriki2ou: extract the curvature

FUNfindContour: image processing and contour extraction

FUNpremonPrediction: using NN to predict and note the best one in every step
