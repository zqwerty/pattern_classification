function res = pattern_classification()
    rng default  % For reproducibility
    [train_set, test_set] = gauss_sample;
%     knn_classify(train_set, test_set);
%     linear_classify(train_set, test_set);
%     quad_classify(train_set, test_set);
%     kmeans_res = K_means(train_set, test_set);
%     mle_classify(train_set, test_set);
%     nn_classify(train_set, test_set);
%     pause
%     nn2_classify(train_set, test_set);
end

function nn_res = nn2_classify(train_set, test_set)
    input = train_set{1,1}';
    ind = train_set{1,2}';
    output = full(ind2vec(ind));
    net = patternnet(10);
    net.divideParam.trainRatio = 0.9;
    net.divideParam.valRatio = 0.1;
    net.divideParam.testRatio = 0;
    net = train(net,input,output);
    view(net);
    x = test_set{1,1}';
    true_label = test_set{1,2};
    y = net(x);
    [s y] = max(y);
    res_mat = zeros(3);
    n = size(x,2);
    for k=1:n
        res_mat(y(k),true_label(k)) = res_mat(y(k), true_label(k)) + 1;
    end
    nn_res = res_mat
    nn_acc = sum(diag(nn_res))/300.0
end

function nn_res = nn_classify(train_set, test_set)
    input = train_set{1,1}';
    label = train_set{1,2};
    n = size(label,1)
    output = zeros(n,2);
    y1 = [1 1];
    y2 = [2 2];
    y3 = [3 3];
    for k=1:n
        if label(k) == 1
            output(k,:) = y1;
        elseif label(k) == 2
            output(k,:) = y2;
        elseif label(k) == 3
            output(k,:) = y3;
        end
    end
    output = output';
    net = feedforwardnet(10);
    net.divideParam.trainRatio = 0.9;
    net.divideParam.valRatio = 0.1;
    net.divideParam.testRatio = 0;
    net = train(net,input,output);
%     view(net);
    y = net(input);
%     perf = perform(net,y,output)
    
    x = test_set{1,1}';
    true_label = test_set{1,2};
    y = net(x)';
    n = size(x,2);
    res_mat = zeros(3);
    for k=1:n
        yk = y(k,:);
        dist = [norm(y1-yk), norm(y2-yk), norm(y3-yk)];
        [d, l] = min(dist);
        res_mat(l, true_label(k)) = res_mat(l, true_label(k)) + 1;
    end
%     perf = perform(net,y,output)
    nn_res = res_mat
    nn_acc = sum(diag(nn_res))/300.0
end

function kmeans_res = K_means(train_set, test_set)
    true_label = train_set{1,2};
    n = size(true_label,1);
%     cluster train set
    [idx,C,sumd,D] = kmeans(train_set{1,1},3,'Distance','cityblock');
    C
    res_mat = zeros(3);
    for k=1:n
        res_mat(idx(k), true_label(k)) = res_mat(idx(k), true_label(k)) + 1;
    end
    train_set_res = res_mat
    
%     cluster test set
    true_label = test_set{1,2};
    n = size(true_label,1);
    [idx,C,sumd,D] = kmeans(test_set{1,1},3,'Distance','cityblock');
    C
    res_mat = zeros(3);
    for k=1:n
        res_mat(idx(k), true_label(k)) = res_mat(idx(k), true_label(k)) + 1;
    end
    test_set_res = res_mat
end

function mle_res = mle_classify(train_set, test_set)
    x = test_set{1,1};
    true_label = test_set{1,2};
    n = size(true_label,1);
    c = 3;
    res_mat = zeros(3);
    
    [prior, mu, sigma,  inv_sigma, coef] = MLE(train_set);
    prior
    mu
    for i=1:c
        sigma((i-1)*c+1:i*c,:)
    end
    
    for k=1:n
%         calculate unnormalize Posterior_i = Prior_i * p(x|\omega_i)
        post = zeros(1,3);
        for i=1:c
            post(i) = coef(i) * exp(-1/2*(x(k,:)-mu(i,:))* inv_sigma((i-1)*c+1:i*c,:)*(x(k,:)-mu(i,:))');
        end
        [max_post, max_post_index] = max(post);
        res_mat(max_post_index, true_label(k)) = res_mat(max_post_index, true_label(k)) + 1;
    end
    mle_res = res_mat
    mle_acc = sum(diag(mle_res))/300.0
end

function [prior_est, mu_est, sigma_est, inv_sigma_est, coef_est] = MLE(train_set)
%     estimate p(omega_i) & p(x|omega_i)
    x = train_set{1};
    n = size(x,1);
    d = size(x,2);
    c = 3;
%     init
%     prior(i)
    prior = [1/3 1/3 1/3];
%     mu(i,:)
    mu = [[1 1 1];[2 2 2];[3 3 3]];
%     sigma((i-1)*c+1:i*c,:)
    sigma = [diag([1,1,1]);diag([1,1,1]);diag([1,1,1])];
%     inv of sigma
    inv_sigma = [];
    coef = [];
    for i=1:c
        inv_sigma = [inv_sigma; inv( sigma((i-1)*c+1:i*c,:))];
%     |sigma_i|^(-1/2) * P(omega_i)
        coef = [coef; det( sigma((i-1)*c+1:i*c,:))^(-1/2)*prior(i)];
    end
%     post_ik = P(omega_i|x_k,theta)
    post = zeros(c,n);

    max_iter = 100;
    for iter=1:max_iter
%         update post prob
        for i=1:c
            for k=1:n
                post(i,k) = exp(-1/2*(x(k,:)-mu(i,:))* inv_sigma((i-1)*c+1:i*c,:)*(x(k,:)-mu(i,:))');
            end
            post(i,:) = post(i,:) * coef(i);
        end
        for k=1:n
            post(:,k) = post(:,k)/sum(post(:,k));
        end
        
%         update prior
        prior = sum(post,2)/n;
%         update sigma
        for i=1:c
            temp = zeros(d);
            for k=1:n
                temp = temp + post(i,k) * (x(k,:)-mu(i,:))' * (x(k,:)-mu(i,:));
            end
            sigma((i-1)*c+1:i*c,:) = temp/(n*prior(i));
        end
%         update mu
        mu = post * x;
        for i=1:c
            mu(i,:) = mu(i,:)/(n*prior(i));
        end
%         update inv_sigma, coef
        for i=1:c
            inv_sigma((i-1)*c+1:i*c,:) = inv( sigma((i-1)*c+1:i*c,:) );
            coef(i) = det( sigma((i-1)*c+1:i*c,:))^(-1/2)*prior(i);
        end
    end
    prior_est = prior;
    mu_est = mu;
    sigma_est = sigma;
    inv_sigma_est = inv_sigma;
    coef_est = coef;
end

function linear_res = linear_classify(train_set, test_set)
    linearModel = fitcdiscr(train_set{1,1},train_set{1,2});
    res_mat = test_mdl(linearModel, test_set);
    linear_res = res_mat
    linear_acc = sum(diag(linear_res))/300.0
end

function quad_res = quad_classify(train_set, test_set)
    quadModel = fitcdiscr(train_set{1,1},train_set{1,2},...
                            'DiscrimType','quadratic');
    res_mat = test_mdl(quadModel, test_set);
    quad_res = res_mat
    quad_acc = sum(diag(quad_res))/300.0
end

function knn_res = knn_classify(train_set, test_set)
    %     auto fit model
    knnModel_auto = fitcknn(train_set{1,1},train_set{1,2},...
                       'OptimizeHyperparameters','auto');
    knn_auto_res = test_mdl(knnModel_auto, test_set)
    knn_auro_acc = sum(diag(knn_auto_res))/300.0
    
    euc_max = 0;
    cty_max = 0;
    euc_res = [];
    cty_res = [];
%     for each k
    for k = 3:9
%         euclidean distance
        knnModel = fitcknn(train_set{1,1},train_set{1,2},...
                       'NumNeighbors',k,'Distance','euclidean');
        res_mat = test_mdl(knnModel, test_set);
        crt = sum(diag(res_mat));
        euc_res = [euc_res, crt];
        if(crt>euc_max)
            euc_max = crt;
            knn_euclid_res = res_mat;
        end
%         cityblock distance
        knnModel = fitcknn(train_set{1,1},train_set{1,2},...
                       'NumNeighbors',k,'Distance','cityblock');
        res_mat = test_mdl(knnModel, test_set);
        crt = sum(diag(res_mat));
        cty_res = [cty_res, crt];
        if(crt>cty_max)
            cty_max = crt;
            knn_cityblock_res = res_mat;
        end
    end
    
%     plot accuracy for both
    pause
    plot(3:9,euc_res/300.0,3:9,cty_res/300.0);
    xlabel('k')
    ylabel('accuracy')
    legend('euclid dist','cityblock dist')
    knn_euclid_res
    knn_cityblock_res
end

function res_mat = test_mdl(mdl, test_set)
    td = test_set{1,1};
    true_label = test_set{1,2};
    label = predict(mdl, td);
    tot = size(label);
    tot = tot(1,1);
    res_mat = zeros(3);
    for i=1:tot
        res_mat(label(i), true_label(i)) = res_mat(label(i), true_label(i)) + 1;
    end
end

function [train_set, test_set] = gauss_sample()
    muA = [1 1 1];
    muB = [3 3 3];
    muC = [7 8 9];
    sigA = [1 1 1];
    sigB = [2 3 4];
    sigC = [6 6 9];
    train_set{1} = [mvnrnd(muA, sigA, 1000);
                    mvnrnd(muB, sigB, 600);
                    mvnrnd(muC, sigC, 1600)];
    cls = zeros(3200,1);
    cls(1:1000)=1;
    cls(1001:1600)=2;
    cls(1601:3200)=3;
    train_set{2} = cls;
    
    test_set{1} =  [mvnrnd(muA, sigA, 100);
                    mvnrnd(muB, sigB, 100);
                    mvnrnd(muC, sigC, 100)];
    cls = zeros(300,1);
    cls(1:100)=1;
    cls(101:200)=2;
    cls(201:300)=3;
    test_set{2} = cls;
end