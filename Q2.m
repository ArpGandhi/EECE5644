clc;
close all;
    
mu1 = [-1;8];
Cov1 = [1, 0.3; 0.3, 1];

mu2 = [-4;1];
Cov2 = [1, -0.2; -0.2, 1];

mu3 = [1;6];
Cov3 = [1, 0.1; 0.1, 1];

mu4 = [0;0];  
Cov4 = [0.8, 0; 0, 0.8];

p1 = 0.25; 
p2 = 0.25;
p3 = 0.25; 
p4 = 0.25;

N = 10000; 
labels = zeros(N,1);

for i = 1:N
    r = rand();
    if r < 0.25
        labels(i) = 1;
    elseif r < 0.5
        labels(i) = 2;
    elseif r < 0.75
        labels(i) = 3;
    else
        labels(i) = 4;
    end
end

N1 = sum(labels==1); 
N2 = sum(labels==2); 
N3 = sum(labels==3); 
N4 = sum(labels==4);

fprintf('Decision distribution in part A \n');
fprintf('Class 1: %d samples\n',N1);
fprintf('Class 2: %d samples\n',N2);
fprintf('Class 3: %d samples\n',N3);
fprintf('Class 4: %d samples\n\n',N4);

X = zeros(N,2);
for i = 1:N
    if labels(i) == 1
        X(i,:) = mvnrnd(mu1, Cov1, 1);
    elseif labels(i) == 2
        X(i,:) = mvnrnd(mu2, Cov2, 1);
    elseif labels(i) == 3
        X(i,:) = mvnrnd(mu3, Cov3, 1);
    else
        X(i,:) = mvnrnd(mu4, Cov4, 1);
    end
end

p_x_1 = mvnpdf(X, mu1', Cov1); 
p_x_2 = mvnpdf(X, mu2', Cov2);
p_x_3 = mvnpdf(X, mu3', Cov3);
p_x_4 = mvnpdf(X, mu4', Cov4);

posterior = [p_x_1*p1, p_x_2*p2, p_x_3*p3, p_x_4*p4];
[~, decisions] = max(posterior, [], 2);
Conf_mat = zeros(4, 4);

for i = 1:N
    Conf_mat(decisions(i), labels(i)) = Conf_mat(decisions(i), labels(i)) + 1;
end

Conf_mat = Conf_mat ./ repmat(sum(Conf_mat, 1), 4, 1);
fprintf('Confusion Matrix P(D=i|L=j) in part A:\n');
fprintf('       L=1      L=2      L=3      L=4\n');
for i = 1:4
    fprintf('D=%d  ', i);
    for j = 1:4
        fprintf('%7.4f  ', Conf_mat(i,j));
    end
    fprintf('\n');
end
fprintf('\n');

P_error = sum(decisions ~= labels) / N;

figure(1);
hold on;
markers = {'.', 'o', '^', 's'};

for c = 1:4
    idx_class = find(labels == c);
    for i = 1:length(idx_class)
        sample_idx = idx_class(i);
        if decisions(sample_idx) == labels(sample_idx)
            plot(X(sample_idx,1), X(sample_idx,2), ...
                ['g', markers{c}], 'MarkerSize', 6);
        else
            plot(X(sample_idx,1), X(sample_idx,2), ...
                ['r', markers{c}], 'MarkerSize', 6);
        end
    end
end

h1 = plot(nan, nan, 'g.', 'MarkerSize', 12);
h2 = plot(nan, nan, 'go', 'MarkerSize', 8);
h3 = plot(nan, nan, 'g^', 'MarkerSize', 8);
h4 = plot(nan, nan, 'gs', 'MarkerSize', 8);

xlabel('X_1', 'FontSize', 12); 
ylabel('X_2', 'FontSize', 12);
title(sprintf('Part A: MAP Classification (P(error)=%.4f)', P_error), 'FontSize', 13);
legend([h1 h2 h3 h4], {'Class 1', 'Class 2', 'Class 3', 'Class 4'}, 'Location', 'best');
grid on;
axis equal;

lambda = [0  10  10 100;
          1   0  10 100;
          1   1   0 100;
          1   1   1   0];

posterior = posterior./repmat(sum(posterior, 2), 1, 4);
R = zeros(N, 4);

for d = 1:4 
    for l = 1:4
        R(:, d) = R(:, d) + lambda(d, l) * posterior(:, l);
    end
end

[~, decisions] = min(R, [], 2);

Conf_mat = zeros(4, 4);
for i = 1:N
    Conf_mat(decisions(i), labels(i)) = Conf_mat(decisions(i), labels(i)) + 1;
end

Conf_mat = Conf_mat./repmat(sum(Conf_mat, 1), 4, 1);

fprintf('Confusion Matrix P(D=i|L=j) in part B:\n');
fprintf('       L=1      L=2      L=3      L=4\n');
for i = 1:4
    fprintf('D=%d  ', i);
    for j = 1:4
        fprintf('%7.4f  ', Conf_mat(i,j));
    end
    fprintf('\n');
end
fprintf('\n');

total_risk = 0;
for i = 1:N
    l = labels(i);
    d = decisions(i);
    total_risk = total_risk + lambda(d, l);
end
avg_risk = total_risk / N;

fprintf('Part A P(error): %.4f\n', P_error);
fprintf('Part B Avg Risk: %.4f\n\n', avg_risk);

fprintf('Decision Distribution in part B:\n');
for d = 1:4
    count = sum(decisions == d);
    fprintf('  Decision %d: %5d samples (%.1f%%)\n', d, count, 100*count/N);
end
fprintf('\n');

figure(2);
hold on;

for c = 1:4
    idx_class = find(labels == c);
    for i = 1:length(idx_class)
        sample_idx = idx_class(i);
        if decisions(sample_idx) == labels(sample_idx)
            plot(X(sample_idx,1), X(sample_idx,2), ...
                ['g', markers{c}], 'MarkerSize', 6);
        else
            plot(X(sample_idx,1), X(sample_idx,2), ...
                ['r', markers{c}], 'MarkerSize', 6);
        end
    end
end

h1 = plot(nan, nan, 'g.', 'MarkerSize', 12);
h2 = plot(nan, nan, 'go', 'MarkerSize', 8);
h3 = plot(nan, nan, 'g^', 'MarkerSize', 8);
h4 = plot(nan, nan, 'gs', 'MarkerSize', 8);

xlabel('X_1', 'FontSize', 12);
ylabel('X_2', 'FontSize', 12);
title(sprintf('Part B: ERM Classification (Avg Risk=%.4f)', avg_risk), 'FontSize', 13);
legend([h1 h2 h3 h4], {'Class 1', 'Class 2', 'Class 3', 'Class 4'}, ...
       'Location', 'best', 'FontSize', 10);
grid on;
axis equal;