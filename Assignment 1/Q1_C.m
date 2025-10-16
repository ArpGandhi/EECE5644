% https://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_LDA09.pdf
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
r0 = mvnrnd(mu0, Cov0, N0);

figure(1)
plot3(r0(:,1),r0(:,2),r0(:,3),'.b'); 
axis equal; 
hold on;

mu1 = [1;1;1]; 
Cov1 = [1,0.3,-0.2;
        0.3,1,0.3;
        -0.2,0.3,1];
r1 = mvnrnd(mu1, Cov1, N1);

figure(1) 
plot3(r1(:,1),r1(:,2),r1(:,3),'.r'); 
axis equal;
hold on;

X = [r0; r1];
Labels = [zeros(N0,1); 
          ones(N1,1)];

X0 = X(Labels==0,:);
X1 = X(Labels==1,:);
est_mu0 = mean(X0)'; 
est_mu1 = mean(X1)';  
% fprintf('Mu0:= %f \n', est_mu0);
% fprintf('Mu1:= %f\n', est_mu1);
est_Cov0 = cov(X0);
est_Cov1 = cov(X1);
Sw = est_Cov0+est_Cov1;
Sb = (est_mu0-est_mu1)*(est_mu0-est_mu1)';
w_LDA = Sw\(est_mu1-est_mu0); % inv(Sw)*(est_mu1-est_mu0)
y = X*w_LDA;
y0 = y(Labels==0);
y1 = y(Labels==1); 
taus = linspace(min(y)-1, max(y)+1, 200);
TPR = zeros(size(taus));
FPR = zeros(size(taus));
Perr = zeros(size(taus));
for k = 1:length(taus)
    tau = taus(k);
    Decisions = (y>tau);
    TP = sum(Decisions==1 & Labels==1);
    FP = sum(Decisions==1 & Labels==0);
    FN = sum(Decisions==0 & Labels==1);
    TN = sum(Decisions==0 & Labels==0);
    % P(D=1|L=1)
    TPR(k) = TP/N1;
    % P(D=1|L=0)
    FPR(k) = FP/N0;
    %p(err) = p(D=1|L=0)*p(0)+p(D=0|L=1)*p(L=1) 
    Perr(k) = FPR(k)*pL0+(1-TPR(k))*pL1;
end

[Pe_min, idx_min] = min(Perr);
emp_tau = taus(idx_min);
TPR_min = TPR(idx_min);
FPR_min = FPR(idx_min);

figure(2);
h1 = plot(FPR, TPR, 'b-', 'LineWidth', 2); 
grid on; 
hold on;
h2 = plot(FPR_min, TPR_min, 'gs', 'MarkerSize', 15, ...
     'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

text(FPR_min + 0.08, TPR_min, ...
     {sprintf('P(error)_{min} = %.4f', Pe_min), ...
      sprintf('\\tau = %.4f', emp_tau), ...
      sprintf('TPR = %.4f', TPR_min), ...
      sprintf('FPR = %.4f', FPR_min)}, ...
     'FontSize', 10, 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'LineWidth', 1);

xlabel('False Positive Rate, P(D=1|L=0)', 'FontSize', 12);
ylabel('True Positive Rate, P(D=1|L=1)', 'FontSize', 12);
title('ROC Curve - Fisher LDA Classifier', 'FontSize', 13);
legend([h1,h2], {'Fisher LDA', 'Min P(error) point'}, ...
       'Location', 'SouthEast', 'FontSize', 10);

figure(3);
histogram(y0, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); 
hold on;
histogram(y1, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
ylims = ylim;
plot([emp_tau, emp_tau], ylims, 'g--', 'LineWidth', 2);
xlabel('Projected value: w^T x', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
title('Fisher LDA: Projected Data Distribution', 'FontSize', 13);
legend('Class 0', 'Class 1', 'Optimal threshold', 'Location', 'best');
grid on;

figure(4);
plot3(X0(:,1), X0(:,2), X0(:,3), '.b', 'MarkerSize', 8); hold on;
plot3(X1(:,1), X1(:,2), X1(:,3), '.r', 'MarkerSize', 8);

[xx,yy] = meshgrid(linspace(min(X(:,1)), max(X(:,1)), 20), ...
                    linspace(min(X(:,2)), max(X(:,2)), 20));
zz = (emp_tau-(w_LDA(1)*xx)-(w_LDA(2)*yy))/w_LDA(3);
surf(xx, yy, zz, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'g');

xlabel('X_1'); ylabel('X_2'); zlabel('X_3');
title('Fisher LDA Decision Boundary in 3D', 'FontSize', 13);
legend('Class 0', 'Class 1', 'Decision Boundary', 'Location', 'best');
grid on;