%This is the part that we use to classify a new object. To do this we use
%the existing NN and we calculate the histogram of the image to classify.
%We compare this histogramm to the existing base histograms and the image is 
%classified to the object that gives the smallest histogram-error.

clear all
load nnNum
load netMat
load netType
load poseStats

PicNum=1;
coilPics=[1 4 5 8 11 13 14 15 16 18]; %THIS SHOULD BE AT THE SAME ORDER AS THIS ON THE completeTestTrain
poses=[0 10 18 26 36 45 53 60];
%poses=0:71;
totalPoses=8; %We consider the 8 different poses for an object from the train stage

objClass=zeros(1,length(coilPics)*length(poses));
trueFalse=zeros(1,length(coilPics)*length(poses));


for objNum=coilPics
    
    for poseNum=poses         
        image=imread( sprintf('coil/obj%d__%d.png', objNum,poseNum)  );
        [x,y]=FUNfindContour(image);
        kamp=FUNcalcKampParametriki2ou(x,y,0,0);
        [bestNN, statisticMat]=FUNpremonPrediction(kamp, nnNum,netMat,netType );
        
        [~, poseClass] =min( mean( gsubtract(statisticMat,poseStats).^2 ) ); %evresi pozas
        objClass(PicNum)=ceil(poseClass/totalPoses); %evresi obj 
        
        if ceil(PicNum/length(poses))==objClass(PicNum)
            trueFalse(PicNum)=1;
        else 
            trueFalse(PicNum)=0;
        end           
        
        PicNum=PicNum+1;        
        
    end
        
end
Accuracy=100*sum(trueFalse)/length(trueFalse);

