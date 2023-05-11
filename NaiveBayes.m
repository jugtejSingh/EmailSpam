%Import data in data Message
dataMessage = ans;

%Partition data into 80/20 for training and testing along
%with their Labels
XTrain = dataMessage(1:int16(round((0.8*4601))), 1:57);
YTrain = dataMessage(1:int16(round((0.8*4601))), 58);

XTest = dataMessage(int16(round((0.8*4601)))+1:end, 1:57);
YTest = dataMessage(int16(round((0.8*4601)))+1:end, 58);
%Train Model
mdl = fitcnb(XTrain,YTrain);

%Get predicted values and scores for accuracy and AUC
[YPred,scores] = predict(mdl,XTest);
%Converting scores for positive value
scores = double(scores(1:920,2));
%Getting the AUC values for when the data is positive
[X,Y,T,AUC] = perfcurve(YTest,scores,'1');
%Getting accuracy for the dataset by comparing true values with predicted
acc = sum(YPred == YTest)/numel(YTest);
%Making of the confusion matrix
cm = confusionchart(YTest,YPred);