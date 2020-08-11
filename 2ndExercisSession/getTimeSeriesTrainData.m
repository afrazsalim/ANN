function [TrainData,TrainTarget,mu,sig,nn_train_target]=getTimeSeriesTrainData(trainset, p,data,mu,sig)
trainset = importdata(trainset);
if data == 0
    mu = mean(trainset);
    sig = std(trainset);
end

dataTrainStandardized = (trainset - mu) / sig;
TrainMatrix=[];
for i=1:p
    TrainMatrix=[TrainMatrix,dataTrainStandardized(i:end-p+i)];
end
TrainData=TrainMatrix(1:end-1,:)';
TrainTarget=dataTrainStandardized(p+1:end)';
nn_train_target = trainset(1:end-p)';