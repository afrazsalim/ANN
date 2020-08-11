p = 3;
[train_data,train_target,mu,sig,nn_train_target] = getTimeSeriesTrainData('lasertrain.dat',p,0,0,0);
[test_data,test_target,mu,sig,nn_test_target] = getTimeSeriesTrainData('laserpred.dat',p,1,mu,sig);

numFeatures = 3;
numResponses = 1;
numHiddenUnits = 300;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',400, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(train_data,train_target,layers,options);


net = predictAndUpdateState(net,train_data);

init = [];
init_pred = [];
for i = p-1:-1:0
   [net,pred] = predictAndUpdateState(net,train_data(:,end-i));
   init_pred = [init_pred,pred];
end
init = init_pred';
numTimeStepsTest = numel(test_target);
YPred = [];
YPred = [YPred,pred];
display("Loop size " + size(init));
[rows,cols] = size(init);
for i = 2:numTimeStepsTest
    [net,pred] = predictAndUpdateState(net,init(:,end),'ExecutionEnvironment','gpu');
    if rows > 1
       for k = 1:rows-1
           init(k,end) = init(k+1,end);
       end
    end
    display(size(init));
    init(end,end) = pred;
    YPred = [YPred,pred];
end


YPred = sig*YPred + mu;
un_train_data = sig*train_data+mu;

numTimeStepsTrain = numel(train_target);


figure
plot(un_train_data(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[train_target(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecastsss")
legend(["Observed" "Forecast"])


%%Test+Observed
YTest = nn_test_target;
rmse = sqrt(mean((YPred-YTest).^2));

display("Sizes");
display(size(nn_test_target));
display("Sizes");
display(size(YPred));

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Y")
title("Forecast & RMSE = " + rmse)

subplot(2,1,2)
stem(YPred - YTest)

