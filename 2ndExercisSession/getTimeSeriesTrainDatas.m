function [TrainData,TrainTarget,full_train_data]=getTimeSeriesTrainDatas(trainset, p)

TrainMatrix=[];
for i=1:p
    TrainMatrix=[TrainMatrix,trainset(i:end-p+i)];
end
TrainData=TrainMatrix(1:end-1,:)';
TrainTarget=trainset(p+1:end)';
full_train_data = importdata(trainset);