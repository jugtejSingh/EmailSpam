%Import data in data Message
tbl = ans;

%Partition table into 60/20/20 for training,testing and validation
tblTrain = tbl(1:int16(round((0.6*4601))), 1:end);
tblValidate = tbl(int16(round((0.6*4601)))+1:int16(round((0.8*4601))),:); 
tblTest = tbl(int16(round((0.8*4601)))+1:end, 1:end);

%Getting number of features in the dataset
numFeatures = size(tblTrain,2) - 1;
%The number of classes which is set to 2 as there are only two binary values
numClasses = 2;
%Putting label values into labelName
labelName = categorical(tblTrain(:,58));
%Splitting the validation table
labelNameValidate = categorical(tblValidate(:,58));
tblValidate = tblValidate(:,1:57);

%Layer makeup of the perceptron
layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
miniBatchSize = 16;
%Option values used by the neural network along with validation data
%which was previously split
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{tblValidate,labelNameValidate},...
    'Plots','training-progress', ...
    'Verbose',false);
%Cutting the training data to remove the label values
tblTrain = tblTrain(:,1:57);
%training the neural network
net = trainNetwork(tblTrain,labelName,layers,options);
%Finding the predicted values from test data and also the scores
[YPred,scoresSmallNet] = classify(net,tblTest(:,1:end-1),'MiniBatchSize',miniBatchSize);
%Converting YTest to categorical array.
YTest = categorical(tblTest(:,58));
%Finding accuracy by testing the predictive values against the true values
acc= sum(YPred == YTest)/numel(YTest);
%Making the confusion chart
confusionchart(YTest,YPred)
%Putting in class names into classNames for rocmetrics
classNames = net.Layers(end).Classes;
%Running the rocmetrics to find out the auc value of the neural network
rocSmallNet = rocmetrics(YTest,scoresSmallNet,classNames);