
 %% functions   
function Md = Mahal_dis (Data,Mu,Cov)

% Measure the Mahalanobis distance
   Md = sqrt(abs((Data-Mu)'*inv(Cov)*(Data-Mu)));

end