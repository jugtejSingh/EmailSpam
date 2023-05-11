function dataMessage = importMessage()
%Clear command window and workspace
clc;
clear;
%Import table
dataMessageTemp = readtable("spambase.data","FileType","text");
%convert to array
dataMessageTemp = table2array(dataMessageTemp);
%Randomizing the data
[NumRow,NumCol] = size(dataMessageTemp);
% Randomize the row
index = randperm(NumRow);
dataMessage = dataMessageTemp(index,:);
end

