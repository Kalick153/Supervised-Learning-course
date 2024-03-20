% Train a stacked classifier starting from the 5 level-1 classifiers available in the skeleton of the code.
%% load dataset
load dataset.mat

%% %% stratified sampling
rng('default'); % for reproducibility
idx_f1=[];
idx_f2=[];
for nclass=1:2
    u=find(labels_tr==nclass);
    idx=randperm(numel(u));
    idx_f1=[idx_f1; u(idx(1:round(numel(idx)/2)))];
    idx_f2=[idx_f2; u(idx(1+round(numel(idx)/2):end))];
end
labels_f1=labels_tr(idx_f1);
labels_f2=labels_tr(idx_f2);
data_f1=data_tr(idx_f1,:);
data_f2=data_tr(idx_f2,:);

%% train level-1 classifiers on fold1
mdl={};

% SVM with gaussian kernel
rng('default');
mdl{1}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'gaussian', 'KernelScale', 5);

% SVM with polynomial kernel
rng('default');
mdl{2}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'polynomial', 'KernelScale', 10);

% decision tree
rng('default');
mdl{3}=fitctree(data_f1, labels_f1, 'SplitCriterion', ...
    'gdi','MaxNumSplits',20);

% Naive Bayes
rng('default');
mdl{4}=fitcnb(data_f1, labels_f1);

% Ensemble of decision trees
rng('default');
mdl{5}=fitcensemble(data_f1, labels_f1);

%% make the predictions on fold2 (to be used to train the meta-learner)
N=numel(mdl);
Predictions=zeros(size(data_f2,1),N);
Scores=zeros(size(data_f2,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_f2);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
end

%% Compare the performance of the meta-classifier when trained on Predictions (i.e. predicted classes) instead of Scores.

%% train stack models
rng('default');
stackedModel_on_Scores = fitcensemble(Scores, labels_f2, 'Method',...
    'Bag');
stackedModel_on_Predictions = fitcensemble(Predictions, labels_f2, 'Method',...
    'Bag');


stac_mdl = {};
stac_mdl{1} = stackedModel_on_Scores;
stac_mdl{2} = stackedModel_on_Predictions;

%% predictions
rng('default');

ACC=[];
ACC_stack = [];
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

% predictions of the stacked classifier
predictions = predict(stackedModel_on_Scores, Scores);
ACC_stack(1)=numel(find(predictions==labels_te))/numel(labels_te);
predictions = predict(stackedModel_on_Predictions, Predictions);
ACC_stack(2)=numel(find(predictions==labels_te))/numel(labels_te);



%% Compare the performance of the meta-classifier when the training split 
% is not performed and the same data is used to train the level-1 classifiers and the meta-classifier.

%% train level 1 on the whole train dataset

%% train level-1 classifiers on fold1
mdl={};

% SVM with gaussian kernel
rng('default');
mdl{1}=fitcsvm(data_tr, labels_tr, 'KernelFunction', ...
    'gaussian', 'KernelScale', 5);

% SVM with polynomial kernel
rng('default');
mdl{2}=fitcsvm(data_tr, labels_tr, 'KernelFunction', ...
    'polynomial', 'KernelScale', 10);

% decision tree
rng('default');
mdl{3}=fitctree(data_tr, labels_tr, 'SplitCriterion', ...
    'gdi','MaxNumSplits',20);

% Naive Bayes
rng('default');
mdl{4}=fitcnb(data_tr, labels_tr);

% Ensemble of decision trees
rng('default');
mdl{5}=fitcensemble(data_tr, labels_tr);

%% train stack models
rng('default');
stackedModel_on_whole_train = fitcensemble(data_tr, labels_tr, 'Method',...
    'Bag');


%% predictions
rng('default');

ACC_on_whole=[];
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    ACC_on_whole(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

% predictions of the stacked classifier
predictions = predict(stackedModel_on_whole_train, data_te);
ACC_stack_on_whole =numel(find(predictions==labels_te))/numel(labels_te);


%% same but only on fold-1
rng('default');
stackedModel_on_fold1 = fitcensemble(data_f1, labels_f1, 'Method',...
    'Bag');
predictions = predict(stackedModel_on_fold1, data_te);
ACC_stack_on_fold1 =numel(find(predictions==labels_te))/numel(labels_te);
