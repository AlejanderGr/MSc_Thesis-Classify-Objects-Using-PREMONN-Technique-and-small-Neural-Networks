%This functions uses all the existing Neural Networks and a calculated 
%curvature of an image. We use NAR NN
%In every step we predict the next curvature value using all the NN and
%calculate their errors. The NN with the lowest error wins and we note this
%information. At the end we will have a string of numbers. Every number
%indicates the NN that won at that step. We also calculate the histogram of
%this string, which shows the percentage of the use of each NN 


function [ bestNN, statisticMat ] = FUNpremonPrediction( kamp, nnNum,netMat,netType )

testData=tonndata(kamp,true,false); 
[x,xi,ai,t] = preparets(netType,{},{},testData);

errorsMat=ones(nnNum,length(kamp)-length(xi) );
statisticMat=ones(nnNum,1);
for nn=1:nnNum 
        net=nncell2mat(netMat(nn));
        provlepsi = net(x,xi,ai);
        errors= gsubtract(t,provlepsi);
        errorsMat(nn,:)=abs(fromnndata(errors,1,1,0));

end
[~,bestNN]=min(errorsMat);

for nn=1:nnNum
    statisticMat(nn,1)=100*sum(bestNN==nn)/length(bestNN);
end
 

end

