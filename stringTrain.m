%We use the existing NN and for every pic we calculate a representative
%string. Here we group the pics and we create the basis strings. To
%classify a new image we will compare its string to this basis strings

clear all
load nnNum
load netMat
load netType

coilPics=[1 4 5 8 11 13 14 15 16 18];
%totalPoses=[70 71 1 2  8 9 11 12  16 17 19 20  24 25 27 28  34 35 37 38  43 44 46 47  51 52 54 55  58 59 61 62];
totalPoses=[1 11 19 27 37 46 54 61];
%totalPoses=[71 1  9 11  17 19  25 27  35 37  44 46  52 54  59 61];



PicNum=1;
bestNNMat=cell(1,numel(coilPics)*numel(totalPoses));

for objNum=coilPics
    
    for poseNum=totalPoses         
        image=imread( sprintf('coil/obj%d__%d.png', objNum,poseNum)  );
        [x,y]=FUNfindContour(image);
        kamp=FUNcalcKampParametriki2ou(x,y,0,0);
        [bestNN, ~]=FUNpremonPrediction(kamp, nnNum,netMat,netType );
        bestNNMat(PicNum)={bestNN};
        PicNum=PicNum+1;        
        
    end
        
end


save bestNNMat bestNNMat
