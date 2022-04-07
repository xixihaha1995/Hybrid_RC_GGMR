
function KL  = SKLD (mean1,cov1,mean2,cov2)
% Find the symetrized Kullback-Leibler divergence
d = length(mean1); % Dimention of gausian component
KLD_12 = log(abs(det(cov2)/(det(cov1)+realmin)))+trace(inv(cov2)*cov1)+ (mean1-mean2)'*inv(cov2)*(mean1-mean2)-d;
KLD_21 = log(abs(det(cov1)/(det(cov2)+realmin)))+trace(inv(cov1)*cov2)+ (mean2-mean1)'*inv(cov1)*(mean2-mean1)-d;
KL = 0.5*(KLD_12+KLD_21);
end