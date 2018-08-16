%After we create the basis-strings we use this code to classify a new
%object. To do this we compare its string to all the existing
%basis-strings. For a new string we need to create all its possible shifts.
%The object will be classified to the string that gives the smallest
%distance from all its shifts. We compare the strings using the LevDist
%function (we calculate the levadian distance)
clear all
load nnNum
load netMat
load netType
load bestNNMat

PicNum=1;
coilPics=[1 4 5 8 11 13 14 15 16 18]; %THIS SHOULD BE AT THE SAME ORDER AS THIS ON THE stringDistTrain
poses=[0 10 18 26 36 45 53 60];
%poses=0:70;
totalPoses=8; %We consider the 8 different poses for an object from the train stage

objClass=zeros(1,length(coilPics)*length(poses));
poseClass=zeros(1,length(coilPics)*length(poses));
trueFalse=zeros(1,length(coilPics)*length(poses));
distMatrix=zeros(1,numel(bestNNMat));

for objNum=coilPics
    
    for poseNum=poses         
        image=imread( sprintf('coil/obj%d__%d.png', objNum,poseNum)  );
        [x,y]=FUNfindContour(image);
        kamp=FUNcalcKampParametriki2ou(x,y,0,0);
        [bestNN, ~]=FUNpremonPrediction(kamp, nnNum,netMat,netType );
        for i=1:numel(bestNNMat)
            distMatrix(i)=levdist( bestNN,bestNNMat{i} );            
        end
      
        [~, poseClass(PicNum)] =min( distMatrix ); %define pose
        objClass(PicNum)=ceil(poseClass(PicNum)/totalPoses); %define obj 
        
        if ceil(PicNum/length(poses))==objClass(PicNum)
            trueFalse(PicNum)=1;
        else 
            trueFalse(PicNum)=0;
        end           
        
        PicNum=PicNum+1        
        
    end
        
end
Accuracy=100*sum(trueFalse)/length(trueFalse);
