clc;
close all;

N = 10000; 
pL0 = 0.65; 
pL1 = 0.35;
u = rand(1,N)>=pL0; 
N0 = length(find(u==0)); 
N1 = length(find(u==1));
mu0 = [-1/2;-1/2;-1/2]; 
Cov0 = [1,-0.5,0.3;
        -0.5,1,-0.5;
        0.3,-0.5,1];
r0 = mvnrnd(mu0,Cov0,N0);
figure(1)
plot3(r0(:,1),r0(:,2),r0(:,3),'.b'); 
axis equal; hold on;
mu1 = [1;1;1]; 
Cov1 = [1,0.3,-0.2;
        0.3,1,0.3;
        -0.2,0.3,1];
r1 = mvnrnd(mu1,Cov1,N1);
figure(1) 
plot3(r1(:,1),r1(:,2),r1(:,3),'.r'); 
axis equal; 
hold on;

X = [r0; r1];
Labels = [zeros(N0,1); 
         ones(N1,1)];
theo_gamma = pL0/pL1;
% disp(theo_gamma);
% fprintf('Det of cov0 is: %f\n', det(Cov0));
% fprintf('inverse of cov0 is:\n');
% disp(inv(Cov0));
% fprintf('Det of cov1 is: %f\n', det(Cov1));
% fprintf('inverse of cov1 is:\n');
% disp(inv(Cov1));

px_L0 = mvnpdf(X,mu0',Cov0);
px_L1 = mvnpdf(X,mu1',Cov1);
LikeRatio = px_L1./px_L0;
gammas = logspace(-3, 3, 200);
TPR = zeros(size(gammas)); 
FPR = zeros(size(gammas)); 
Perr = zeros(size(gammas));

for k = 1:length(gammas)
    gamma = gammas(k);
    Decisions = (LikeRatio>gamma);
    TP = sum(Decisions==1 & Labels==1);
    FP = sum(Decisions==1 & Labels==0);
    FN = sum(Decisions==0 & Labels==1);
    TN = sum(Decisions==0 & Labels==0);
    % P(D=1|L=1)
    TPR(k) = TP/N1;
    % P(D=1|L=0)
    FPR(k) = FP/N0;
    %p(err) = p(D=1|L=0)*p(0)+p(D=0|L=1)*p(1) 
    Perr(k) = FPR(k)*pL0+(1-TPR(k))*pL1;
end

[Pe_min, idx_min] = min(Perr);
% disp([Pe_min, idx_min]);
emp_gamma = gammas(idx_min);
TPR_min = TPR(idx_min);
FPR_min = FPR(idx_min);
fprintf('Theoretical Y = %.4f\n', theo_gamma);
fprintf('Empirical Y = %.4f\n', emp_gamma);
fprintf('Minimum empirical P(error) = %.4f\n', Pe_min);
fprintf('At this Y: TPR = %.4f, FPR = %.4f\n', TPR(idx_min), FPR(idx_min));

figure(2);
plot(FPR, TPR, 'b-', 'LineWidth', 1.5);
grid on; 
hold on;
plot(FPR(idx_min), TPR(idx_min), 'gs', 'MarkerSize', 15, ...
     'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
text(FPR(idx_min) + 0.08, TPR(idx_min), ...
     {sprintf('Min P(error) = %.4f', Pe_min), ...
      sprintf('\\gamma = %.4f', emp_gamma), ...
      sprintf('TPR_{min} = %.4f', TPR(idx_min)), ...
      sprintf('FPR_{min} = %.4f', FPR(idx_min))}, ...
     'FontSize', 10, 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'LineWidth', 1);

xlabel('False Positive Rate, P(D=1|L=0)', 'FontSize', 12);
ylabel('True Positive Rate, P(D=1|L=1)', 'FontSize', 12);
title('ROC Curve of ERM Classifier', 'FontSize', 13);
legend('ROC Curve', 'ERM Classifier', 'Min P(error) point', ...
       'Location', 'SouthEast', 'FontSize', 10);

figure(3);
semilogx(gammas, Perr, 'b-', 'LineWidth', 2);
hold on;
semilogx(emp_gamma, Pe_min, 'ro', 'MarkerSize', 12, ...
         'MarkerFaceColor', 'r', 'LineWidth', 2);
semilogx([theo_gamma, theo_gamma], [0, max(Perr)], 'g--', 'LineWidth', 2);
grid on;
xlabel('\gamma (threshold)', 'FontSize', 12);
ylabel('Probability of Error', 'FontSize', 12);
title('P(error) vs \gamma', 'FontSize', 14);
legend('P(error)', '\gamma_{Empirical}', '\gamma_{Theoretical}', ...
       'Location', 'best');
text(emp_gamma*1.5, Pe_min, ...
sprintf('\\gamma_{emp}=%.4f\nP(err)_{min}=%.4f', emp_gamma, Pe_min), ...
'FontSize', 10);
