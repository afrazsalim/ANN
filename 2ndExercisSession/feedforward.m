 p = 1;
[train_data,train_target,mu,sig,nn_train_target] = getTimeSeriesTrainData('lasertrain.dat',p,0,0,0);
[test_data,test_target,mu,sig,nn_test_target] = getTimeSeriesTrainData('laserpred.dat',p,1,mu,sig);
disp(size(train_data));
disp(size(train_target));

 layers = 300;
 net = feedforwardnet(layers);
 net.trainParam.epochs=1000;
 net.trainParam.max_fail = 200;
 net.divideParam.trainRatio = 1; 
 net.divideParam.valRatio = 0;
 net.divideParam.testRatio = 0;  
 net = train(net,train_data,train_target);
 view(net);
 init = train_data(:,end);
 numTimeStepsTest = numel(test_target);
 y = net(init);
 YPred = y;
 [rows,cols] = size(init);
 for i = 2:numTimeStepsTest
     pred = net(init);
     if rows > 1
       for k = 1:rows-1
           init(k,end) = init(k+1,end);
       end
     end
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
display(size(train_target));


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
xlabel("Month")
ylabel("Error")

 
 
 
 
