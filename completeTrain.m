%Given some pictures we use the PREMONN technique. We first calculate the
%curvature for a pic. Then we create small Neural Networks that learn this
%curvature topical and we predict the next values of this curvature. 
%When all the existing NN give error higher than <thres> we
%train a new network at this region. We continue this procces until we test
%all the images. Thus we learn the shapes that compose this images. We
%test all the pictures and at the end we will have a set of NN. Every
%one of this Nets represends a shape.
%Afterwards we use a mergin techique. To do this we use all the NN to 
%predict the train data of all the other NN. For every train-data-set we 
%keep the Network that gives the lowest error
%The pics should have name trainX.png and be in total <totalPicsTrain>
%At the end we need the Neural Networks <netMat>, the networks Type 
%<netType> and the total networks number <nnNum>

tic
clear all
load totalPicsTrain;
trainNum=50; 
trainNum2=3;
checkNum=2;
thres=0.0005;
delayNum=3;

%NN parameters 
feedbackDelays = 1:delayNum;
hiddenLayerSize = 7;
netType = narnet(feedbackDelays,hiddenLayerSize);
netType.divideParam.trainRatio = 70/100;
netType.divideParam.valRatio = 30/100;
netType.divideParam.testRatio = 0/100;        
%netType.trainParam.goal = 1e-4;
%netType.trainParam.min_grad = 1e-10;
netType.trainParam.max_fail=6;


nnNum=1;


PicNum=1;
p_old=trainNum-delayNum;
image=imread( sprintf('trainData/train%d.png', PicNum)  );
[x,y]=FUNfindContour(image);
kamp=FUNcalcKampParametriki2ou(x,y,0,0);
kamp=tonndata(kamp,true,false);  
trainData=kamp(1:trainNum);
trainDataMat(:,nnNum)=cell2mat(trainData);


 
%Train NN   
[x,xi,ai,t] = preparets(netType,{},{},trainData);
[net,tr] = train(netType,x,t,xi,ai);
netMat(nnNum)={net};



[x,xi,ai,t] = preparets(netType,{},{},kamp);
provlepsi = net(x,xi,ai);
errors= gsubtract(t,provlepsi);
errors=abs(fromnndata(errors,1,1,0));
bestNN=nnNum*ones(1,length(errors));

    while 1
        newNet=0;
        for p=p_old:length(errors)-trainNum+trainNum2 
            if (sum(  errors(p:p+checkNum)>thres )==checkNum+1)
                newNet=1;
                p_old=p;   
                break
            end
        end
        
        if newNet==0 
            break
        end
        
        %create new NN
        nnNum=nnNum+1;
        trainData=kamp(p+delayNum-trainNum2:p+delayNum+trainNum-trainNum2-1);
        [x,xi,ai,t] = preparets(netType,{},{},trainData);
        [net,tr] = train(netType,x,t,xi,ai);
        trainDataMat(:,nnNum)=cell2mat(trainData);
        netMat(nnNum)={net};
        
        %new prediction
        [x,xi,ai,t] = preparets(net,{},{},kamp(p:end));
        provlepsi = net(x,xi,ai);
        tempErrors= gsubtract(t,provlepsi);
        tempErrors=abs(fromnndata(tempErrors,1,1,0));
        bestNN(p-1+find(tempErrors<errors(p:end)))=nnNum;
        errors(p:end)=min(errors(p:end),tempErrors);
    end
    
    bestNNMatTrain(PicNum)={bestNN};
   
%finish with first Pic


for PicNum=2:totalPicsTrain
    clear errorsMat
    image=imread( sprintf('trainData/train%d.png', PicNum)  );
    [x,y]=FUNfindContour(image);
    kamp=FUNcalcKampParametriki2ou(x,y,0,0);
    kamp=tonndata(kamp,true,false);  
    

    for nn=1:nnNum 
        net=nncell2mat(netMat(nn));
        [x,xi,ai,t] = preparets(netType,{},{},kamp);
        provlepsi = net(x,xi,ai);
        errorsMat(nn,:)=abs(fromnndata(gsubtract(t,provlepsi),1,1,0));
    end
    

if length(errorsMat(:,1))>1
    [errors,bestNN]=min(errorsMat);
else 
    errors=errorsMat;
    bestNN=ones(1,length(errors));
end


p_old=trainNum2;
    while 1
        newNet=0;
        for p=p_old:length(errors)-trainNum+trainNum2 
            if (sum(  errors(p:p+checkNum)>thres )==checkNum+1)
                newNet=1;
                p_old=p;  
                break
            end
        end
        
        if newNet==0
            break
        end
        
        nnNum=nnNum+1;
        trainData=kamp(p+delayNum-trainNum2:p+delayNum+trainNum-trainNum2-1);
        [x,xi,ai,t] = preparets(netType,{},{},trainData);
        [net,tr] = train(netType,x,t,xi,ai);
        trainDataMat(:,nnNum)=cell2mat(trainData);
        netMat(nnNum)={net};        

        
        [x,xi,ai,t] = preparets(netType,{},{},kamp(p:end));
        provlepsi = net(x,xi,ai);
        tempErrors= gsubtract(t,provlepsi);
        tempErrors=abs(fromnndata(tempErrors,1,1,0));
        bestNN(p-1+find(tempErrors<errors(p:end)))=nnNum;
        errors(p:end)=min(errors(p:end),tempErrors);        
    end
    
    bestNNMatTrain(PicNum)={bestNN};
    PicNum


       
end
clearvars -except nnNum netMat bestNNMatTrain trainDataMat netType

% MERGIN----------------------------------
for testNN=1:nnNum
    clear errorsMat
    testData=tonndata(trainDataMat(:,testNN)',true,false); 
    [x,xi,ai,t] = preparets(netType,{},{},testData);
    
    for compareNN=1:nnNum
        net=nncell2mat(netMat(compareNN));
        provlepsi = net(x,xi,ai);
        errors= gsubtract(t,provlepsi);
        errorsMat(compareNN)=mean( abs(fromnndata(errors,1,1,0)) );
    end
    [~, bestPerformance(testNN)]=min(errorsMat);
 
end
bestPerformance=unique(bestPerformance); %best Nets
nnNumFirst=nnNum;
nnNum=length(bestPerformance);
for i=1:length(bestPerformance)
    netMatFinal(i)=netMat(bestPerformance(i));
    
end
netMat=netMatFinal;
clearvars -except nnNum nnNumFirst netMat bestNNMatTrain netType bestPerformance

save nnNum nnNum                    
save netType netType                
save netMat netMat                   
%save bestNNMatTrain bestNNMatTrain   

toc
        
