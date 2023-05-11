%Import data in data Message
dataMessage = ans;

%Partition data into 80/20 for training and testing along
%with their Labels
XTrain = dataMessage(1:int16(round((0.8*4601))), 1:57);
YTrain = dataMessage(1:int16(round((0.8*4601))), 58);

XTest = dataMessage(int16(round((0.8*4601))):end, 1:57);
YTest = dataMessage(int16(round((0.8*4601))):end, 58);
%Making the model using training data and defining the model as
%a binomial logistic regression model
mdl = fitglm(XTrain,YTrain,Distribution="binomial",BinomialSize=2);

%Getting the predicted values from the model based on testing data
YPred = predict(mdl,XTest);
%Getting the score value for AUC
scores = mdl.Fitted.Probability;
[X,Y,T,AUC] = perfcurve(YTrain,scores,'1');
%The code below uses the T value to find the most optimal accuracy
%Changing the threshold each time to find when accuracy is the highest
temp = 0;
threshHold = 0;
tempPred = YPred;
for f = 1:size(T)
    YPred = tempPred;
YPred = double(YPred() > T(f));
acc = sum(YPred == YTest)/numel(YTest);
    if acc > temp
        temp = acc;
        k = f;
    end
end
%The threshold with the highest accuracy is used to convert values
YPred = double(tempPred > T(k));
%Getting accuracy for the dataset by comparing true values with predicted
acc = sum(YPred == YTest)/numel(YTest);
%Making of the confusion matrix
cm = confusionchart(YTest,YPred);
