% After we create the Neural Networks that we will use, we run this
% code. Its adjusted to the COIL20 database. We group the pics at poses
% and we calculate the pose histograms. Every histogramm represends a pose.
%For every object we consider 8 poses. We have chosen the 10 objects
%from the database that give the best contour
clear all
load nnNum
load netMat
load netType

PicNum=1;
coilPics=[1 4 5 8 11 13 14 15 16 18];
totalPoses=[70 71 1 2  8 9 11 12  16 17 19 20  24 25 27 28  34 35 37 38  43 44 46 47  51 52 54 55  58 59 61 62];

for objNum=coilPics
    
    for poseNum=totalPoses         
        image=imread( sprintf('coil/obj%d__%d.png', objNum,poseNum)  );
        [x,y]=FUNfindContour(image);
        kamp=FUNcalcKampParametriki2ou(x,y,0,0);
        [~, statisticMat(:,PicNum)]=FUNpremonPrediction(kamp, nnNum,netMat,netType );
        PicNum=PicNum+1;        
        
    end
        
end

i=1;
j=1;
poseStats=zeros(nnNum,80);
while i<=317
    poseStats(:,j)= mean(statisticMat(:,i:i+3),2);
    j=j+1;
    i=i+4;
end
save poseStats poseStats
