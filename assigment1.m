accuracy_5x2=[];
rng('default');

for ndataset=1:4
    switch ndataset
        case 1, load dataset1.mat
        case 2, load dataset2.mat
        case 3, load dataset3.mat
        case 4, load dataset4.mat
        otherwise
    end

    accuracy_times=[];
    for ntimes=1:5
        % stratified sampling
        idx_tr=[];
        idx_te=[];
        for nclass=1:2
            u=find(labels==nclass);
            idx=randperm(numel(u));
            idx_tr=[idx_tr; u(idx(1:round(numel(idx)/2)))];
            idx_te=[idx_te; u(idx(1+round(numel(idx)/2):end))];
        end
        labels_tr=labels(idx_tr);
        labels_te=labels(idx_te);
        data_tr=data(idx_tr,:);
        data_te=data(idx_te,:);

        % training classifier(s)
        % train on training split, test on test split

        SVM_LIN_tr=fitcsvm(data_tr,labels_tr,'KernelFunction',...
            'linear','KernelScale',1);
        SVM_RBF_tr=fitcsvm(data_tr,labels_tr,'KernelFunction',...
            'gaussian','KernelScale',0.1);
        KNN_tr = fitcknn(data_tr,labels_tr,'Distance','Euclidean',...
            'NumNeighbors',10);
        TREE_tr = fitctree(data_tr,labels_tr,'SplitCriterion','gdi',...
             'MaxNumSplits', 15);
        models_tr={SVM_LIN_tr, SVM_RBF_tr, KNN_tr, TREE_tr};


        % reversing role of training and test:
        % train on test split, test on train split
        SVM_LIN_te=fitcsvm(data_te,labels_te,'KernelFunction',...
            'linear','KernelScale',1);
        SVM_RBF_te=fitcsvm(data_te,labels_te,'KernelFunction',...
            'gaussian','KernelScale',0.1);
        KNN_te = fitcknn(data_te,labels_te,'Distance','Euclidean',...
            'NumNeighbors',10);
        TREE_te = fitctree(data_te,labels_te,'SplitCriterion','gdi',...
             'MaxNumSplits', 15);
        models_te={SVM_LIN_te, SVM_RBF_te, KNN_te, TREE_te};

        for nmodel=1:4
            prediction1=predict(models_tr{nmodel},data_te);
            accuracy1= numel(find(prediction1==labels_te))/numel(labels_te);
            prediction2=predict(models_te{nmodel},data_tr);
            accuracy2= numel(find(prediction2==labels_tr))/numel(labels_tr);

            accuracy = (accuracy1+accuracy2)/2;
            accuracy_times(ntimes,nmodel)=accuracy;
        end
    end
    for nmodel=1:4
        accuracy_5x2(ndataset,nmodel)=mean(accuracy_times(:, nmodel));
    end
end

%% rankings
ranks = [
    1.5, 3, 1.5, 4; 
    1, 4, 2, 3; 
    4, 2, 1, 3; 
    4, 2, 3, 1  
];

% Compute the average rank for each model
avg_ranks = [];
for i=1:4 
    avg_ranks(i) = mean(ranks(:, i));
end


