%First we get the train and test data.
p = 1;
[~,~,full_train_data] = getTimeSeriesTrainDatas('lasertrain.dat',p);
[~,~,full_test_data] = getTimeSeriesTrainDatas('laserpred.dat',p);
%Standardize the data.
%mu = mean(full_train_data);
%sig = std(full_train_data);
%full_normalize_data = (full_train_data - mu) / sig;
%Create train and target data.
data_copy = full_train_data;
mu = mean(full_train_data);
sig = std(full_train_data);
dataTrainStandardized = (full_train_data - mu) / sig;
train_data = dataTrainStandardized(1:end-p,:)';
train_target=dataTrainStandardized(p+1:end)';


dataTestStandardized = (full_test_data - mu) / sig;
test_data = dataTestStandardized(1:end-p)';
test_target = dataTestStandardized(p+1:end)';
test_target = full_test_data(1:end-p)';
%First we plot a figure.
figure
plot(test_data)
xlabel("x")
ylabel("Y")
title("Time series test data")
%Build an LSTM
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 150;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    lstmLayer(numHiddenUnits)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.1, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(train_data,train_target,layers,options);
analyzeNetwork(net);
%Now read the test dataset.
net = predictAndUpdateState(net,train_data);
[net,YPred] = predictAndUpdateState(net,train_target(end));
numTimeStepsTest = numel(test_data);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','gpu');
end
YPred = sig*YPred + mu;

full_test_data = test_target;
rmse = sqrt(mean((YPred-full_test_data).^2));

train_data = data_copy(1:end-p,:)';
numTimeStepsTrain = numel(train_data);
figure
plot(train_data(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[train_data(numTimeStepsTrain) YPred],'.-')
hold off
xlim([0,1100])
xlabel("X")
ylabel("Y")
title("Forecast & RMSE = " + rmse)
legend(["Observed" "Forecast"])
%We plot test data against forecasted.

display(size(YPred));
figure
subplot(2,1,1)
plot(full_test_data)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Y")
xlabel("X")
title("Forecast & RMSE = " + rmse)

subplot(2,1,2)
stem(YPred - full_test_data)
xlabel("X")
ylabel("Error")
title("RMSE = " + rmse)



