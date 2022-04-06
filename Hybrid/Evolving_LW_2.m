%function [Priors, Mu, Sigma, expData] = Evolving_LW(Priors, Mu, Sigma, Data_Test, L_rate, T_mrg, T_split, T_sigma, sum_beta)
function [Priors, Mu, Sigma, expData] = Evolving_LW_2(Priors, Mu, Sigma, Data_Test,SumPosterior)
L1 = size(Data_Test,1);
%[m_best_Ts,Post_pr_Ts] = BMC(Data_Test(1:L1,1),Priors,Mu,Sigma);

% variables to store the model parameters during adaptation
Priors_all = [];
Mu_all = [];
Sigma_all = [];
M_comp_all = [];
PostPr_all = [];
dis_comp_all = [];
MDdis_all = [];
%PostPr_all = Post_pr_Ts';
Priors_all(:,:,1) = Priors;
Mu_all(:,:,1) = Mu;
Sigma_all(:,:,:,1) = Sigma;
M_comp_all(1) = length(Priors);


%% Adaptive GMM algorithm
% length(Data_Test)
Rou_all = [];
Beta_all = [];
mbest_all = [];
%mbest_all(1) = m_best_Ts;
sc = [];
mc=[];
gc=[];
mvl = [];
mvl_ind = [];
Pr_Sum =0;
Mrg_Mat_all =[];
rej = [];
buf = 0;
%C_mat = ones(length(Priors),1); % sum of expected posterior for each component
C_mat=SumPosterior';
L_rate = 0.01;%0.001;%0.001; %learning rate 0.01
pumax = 0.09;
M_min = 2; % Minimum number of mixture components
M_max =50; %20; % Maximum number of mixture components
Beta_min = 0.03; % Minimum learning factor
Beta_max = 0.07; % Maximum learning factor
T_sigma=2;
eps = 1e-2;
T_mrg = 0.000001;%.00000100; % Merge threhsold need to be determined 395
T_split = 100;%10; % Split threshold  3.8695e-03
Spl_fac = 0.8; % split factor
sinit = 300; % initial determinan for genertaed component
pr_init = mean(Priors); % initial prior for genertaed component
Sigma = Sigma+0.00000002*eye(size(Data_Test(1:L1,:),1));% Add small white noise to prevent singularity

yest2 = [];
nbVar = L1; 
nes=L1-1;

er_mat = [];
Total_er_mat = [];
Fault_ID = [];
% t = 2:(size(Data_Test,2))
n=0;
for t = 2:(size(Data_Test,2))  
%% AFDD algorithm
    t
    [expData(t,1), cof1]=GMR(Priors, Mu, Sigma,  Data_Test(1:L1-1,t), [1:nes], [nes+1:nbVar]);
    
    n=n+1;
%% Model Update algorithm
    com_dis = [];
    com_cand = [];
    Post_pr = [];
    M_comp = length(Priors); %Number of mixture components
    
    % Record parameters of last GMM before starting adaptation
    Priors_L = Priors; 
    Mu_L = Mu;
    Sigma_L = Sigma;
    
    
    
% Update the best match component for current observation
[m_best,Post_pr] = BMC(Data_Test(1:L1,t),Priors,Mu,Sigma);
Mu_BMC = Mu(:,m_best);
Sigma_BMC = Sigma(:,:,m_best);



% find the distance of new observation from all available mixture components
for m=1:size(Priors,2)
    %com_dis(m,1) = norm(Data_Test(1:L1,t) - Mu(:,m))/abs(det(Sigma(:,:,m)));
    com_MD(m,1)= Mahal_dis(Data_Test(1:L1,t),Mu(:,m),Sigma(:,:,m));
end

% find the candidates to update the model


for m=1:size(Priors,2)   
%     if (com_dis(m,1)< mean(com_dis))&&(Post_pr (m)> eps)
%         com_cand(m,1) = 1;   
%     else
%         com_cand(m,1) = 0;
%         Post_pr(m) = 0; 
%         
%     end  
    % if (com_MD(m_best,1)< mean(com_dis))&&(Post_pr (m_best)> eps)
%     if (com_MD(m_best,1)< T_sigma)&&(Post_pr (m_best)> eps)
%         com_cand(m_best,1) = 1;   
%         updateFlag=1;
%     else
%         com_cand(m_best,1) = 0;
%         Post_pr(m_best) = 0; 
%         updateFlag=1;
%     end 
updateFlag=1;
existFlag=0;
    if (com_MD(m_best,1)< T_sigma)&&(Post_pr (m_best)> eps)
    existFlag=1; %doing nothing
    end
    
end

% Find the learning factor given the new observation and the previous best
% match component
%  Beta = Beta_max - (Beta_max - Beta_min)*(exp(-Mahal_dis(Data_Test(1:5,t),Mu_BMC,Sigma_BMC)/Avg_Mahdis_Tr));


%PostPr_all(t,1:size(Priors,2)) = Post_pr';
%dis_comp_all(t,1:size(Priors,2)) = com_dis';
% MDdis_all(t,1:size(Priors,2)) =com_MD';
mbest_all(t) = m_best;


% Update the model parameters for selected componenets if there is any. If
% there is no GMM componenet matches the new observation change the GMM
% structure by means of split operation.
 
%     for i=1:size(Priors,2)
%     
%     Pr_Sum = Pr_Sum+gaussPDF(Data_Test(1:L1,t),Mu_L(:,i),Sigma_L(:,:,i));
%     end


% ind_upd = find(com_cand == 1); % Update all matched components
ind_upd = find(Post_pr == max(Post_pr)& max(Post_pr)~=0 &updateFlag==1); % Update the
% best mach components

% if (sum(Post_pr) ~= 0)
if (length(ind_upd)~= 0 & existFlag~=1) % Update the model parameters

    %% Update parameters

    Rout=[];
    q=[];
    for i=1:length(ind_upd)
        
        q = Post_pr(ind_upd(i))./sum(Post_pr);
        
        C_mat(ind_upd(i)) = C_mat(ind_upd(i))+q;
        pu=(1 - L_rate) * Priors(ind_upd(i)) + L_rate * q ;
        Priors(ind_upd(i)) = min(pu,pumax);
        Rou = q * (((1-L_rate)/C_mat(ind_upd(i)))+ L_rate);
        Mu(:,ind_upd(i)) = (1-Rou) * Mu(:,ind_upd(i)) + Rou * Data_Test(1:L1,t);
        Sigma(:,:,ind_upd(i)) = (1 - Rou)*Sigma(:,:,ind_upd(i)) + Rou*(Data_Test(1:L1,t) - Mu(:,ind_upd(i)))*(Data_Test(1:L1,t) - Mu(:,ind_upd(i)))';
        
    
%         Rou = Beta * gaussPDF(Data_Test(:,t),Mu_L(:,ind_upd(i)),Sigma_L(:,:,ind_upd(i)));
%         Priors(ind_upd(i)) = (1 - Beta) * Priors(ind_upd(i)) + Beta;
%         Mu(:,ind_upd(i)) = (1-Rou) * Mu(:,ind_upd(i)) + Rou * Data_Test(:,t);
%         Sigma(:,:,ind_upd(i)) = (1 - Rou)*Sigma(:,:,ind_upd(i)) + Rou*(Data_Test(:,t) - Mu(:,ind_upd(i)))*(Data_Test(:,t) - Mu(:,ind_upd(i)))';
        Rout(1,i) = Rou;
%         Beta_all = [Beta_all Beta];
    end
    Rou_all(t,1:size(Rout,2)) = Rout;
    Priors = Priors/sum(Priors); % Normalize the priores after updationg operation
    
    %% Generate a gaussian component 
    elseif (size(Priors,2) < M_max && buf ==0 & existFlag~=1)
        buf = 0;
        gc(t)=1;
        Priors_j = pr_init;
%     Mu_j = mean (Data_Test(:,1:t),2);
        Mu_j = Data_Test(1:L1,t);
        ns = size(Sigma(:,:,1),1); 
        Sigma_j = sinit*eye(ns);   
    % Append the new component information
        Priors(size(Priors,2)+1) = Priors_j;
        Mu(:,size(Priors,2)) =  Mu_j;
        Sigma(:,:,size(Priors,2)) = Sigma_j;
        Priors = Priors/sum(Priors); % Normalize the priores after split operation
        C_mat(size(Priors,2))=1;
        
else
    buf = 0;
            
   
end
% Sigma = Sigma-0.0011*eye(size(Data_Train,1));
%% Split operation
Vol = [];
for i=1:length(Priors)
    Vol(i) = det(Sigma(:,:,i));
end

if max(Vol) > T_split && size(Priors,2) < M_max 
    
    mvl(t) = max(Vol);
    vi = find (Vol == max(Vol));
    mvl_ind(t) = vi(1);
    sc(t)=1;
    ind_Spl = find (Vol == max(Vol));
    ind_Spl = ind_Spl(1); % check that

    [COEFF,latent] = pcacov(Sigma(:,:,ind_Spl));
    E_vec = COEFF(:,1)/norm(COEFF(:,1));
    E_val = latent(1);
    
    DeltaV = sqrt(Spl_fac*E_val)* E_vec;
    
    Mu1 = Mu(:,ind_Spl) + DeltaV ;
    Mu2 = Mu(:,ind_Spl) - DeltaV ;
    Priors(ind_Spl) = Priors(ind_Spl)/2;
    Mu(:,ind_Spl) = Mu1;
    Sigma(:,:,ind_Spl) = Sigma(:,:,ind_Spl)-DeltaV*DeltaV';
    
    % Append the new component information
    Priors(size(Priors,2)+1) = Priors(ind_Spl)/2;
    Mu(:,size(Priors,2)) =  Mu2;
    Sigma(:,:,size(Priors,2)) = Sigma(:,:,ind_Spl);
    Priors = Priors/sum(Priors); % Normalize the priores after merge operation
    %C_mat(size(Priors,2))=C_mat(ind_Spl)/2;
    %C_mat(ind_Spl)=C_mat(ind_Spl)/2;
    C_mat(size(Priors,2))=1;
    C_mat(ind_Spl)=1;  
end

%     
% 





%% Merge Operation

Mi = tril(ones(length(Priors)),-1);
[Mi_r Mi_c] = find(Mi==1);
Skld_tot = [];
for i=1:length(Mi_r)
    Skld_gm = SKLD (Mu(:,Mi_r(i)),Sigma(:,:,Mi_r(i)),Mu(:,Mi_c(i)),Sigma(:,:,Mi_c(i)));
    Skld_tot = [Skld_tot Skld_gm];
end

if min(abs(Skld_tot)) < T_mrg
    mc(t)=1;
    Skld_MinInd = find (Skld_tot==min(Skld_tot));
    Max_r = Mi_r(Skld_MinInd);
    Max_c = Mi_c(Skld_MinInd);
    Max_r = Max_r(1); 
    Max_c = Max_c(1);
        
    Priors_m = Priors(Max_r)+Priors(Max_c);
    Mu_m = (Priors(Max_r)*Mu(:,Max_r)+Priors(Max_c)*Mu(:,Max_c))/Priors_m;
    Sigma_m = (Priors(Max_r)*(Sigma(:,:,Max_r)+(Mu(:,Max_r)-Mu_m)*(Mu(:,Max_r)-Mu_m)')+Priors(Max_c)*(Sigma(:,:,Max_c)+(Mu(:,Max_c)-Mu_m)*(Mu(:,Max_c)-Mu_m)'))/Priors_m;
        
    min_mrg_ind = min(Max_r,Max_c);
    max_mrg_ind = max(Max_r,Max_c);
    
    Priors(min_mrg_ind) = Priors_m;
    Mu(:,min_mrg_ind) = Mu_m;
    Sigma(:,:,min_mrg_ind) = Sigma_m;
    %C_mat(min_mrg_ind)=C_mat(min_mrg_ind)+C_mat(max_mrg_ind);
    C_mat(min_mrg_ind)=1;
    
    Priors(max_mrg_ind) = [];
    Mu(:,max_mrg_ind) = [];
    Sigma(:,:,max_mrg_ind) = [];
    Priors = Priors/sum(Priors); % Normalize the priores after merge operation
    C_mat(max_mrg_ind) = [];   
% [m_best,Post_pr] = BMC(Data_Test(:,t),Priors,Mu,Sigma);
% Post_pr_mat = Post_pr*Post_pr';
% Mrg_Mat_all(1:size(Priors,2),1:size(Priors,2),t) = Post_pr_mat;
% 
% ind_nondg = eye(length(Post_pr_mat));
% max_nondg = max(Post_pr_mat(~ind_nondg));
% [Max_r Max_c] = find(Post_pr_mat == max_nondg); % find the index of maximu of nondiagonal elements
% Max_r = Max_r(1); 
% Max_c = Max_c(1);

% .00001 (max_nondg > 0.001)&&   
% if (abs(Post_pr(Max_r)-Post_pr(Max_c))<.00001)&&(size(Priors,2)>8)
    
    
    
    
end

%% Acceptance probability
    % Compute the likelihood of the last GMM. Data_comb contains all training data
    % and all new observations 
% Data_comb = [Data_Train(1:8,1:min(t,size(Data_Train,2))) Data_Test(1:8,1:t)];
% Data_comb = [Data_Train(1:8,:) Data_Test(1:8,1:t)];
% Data_comb = [Data_Train(1:8,1:t) Data_Test(1:8,1:t)];
% Pr_Data_L = [];
% Pr_Data_U = [];
% for i = 1:size(Priors_L,2)
%     Pr_Data_L(:,i) = gaussPDF(Data_comb,Mu_L(:,i),Sigma_L(:,:,i));   
% end
% Pr_Data_LVec = Pr_Data_L * Priors_L';
% Pr_Data_LVec(find(Pr_Data_LVec == 0))= realmin;
% LogLik_L = sum(log(Pr_Data_LVec)); %likelihood of the last GMM
% 
%     % Compute the the likelihood of the updated GMM
% for i = 1:size(Priors,2)
%     Pr_Data_U(:,i) = gaussPDF(Data_comb,Mu(:,i),Sigma(:,:,i)); 
% end
% Pr_Data_UVec = Pr_Data_U * Priors';
% Pr_Data_UVec(find(Pr_Data_UVec == 0))= realmin;
% LogLik_U = sum(log(Pr_Data_UVec)); %likelihood of the last GMM
% 
% Pr_accp = 1/exp(LogLik_L - LogLik_U);
%Check the update rejection condition
% if Pr_accp < rand(1) 
%     Priors = Priors_L;
%     Mu = Mu_L;
%     Sigma = Sigma_L;
%     rej(t) = 1;
% end

M_comp = size(Priors,2);

% Store the model parameters 
M_comp_all(t) = M_comp;
Priors_all(t,1:M_comp) = Priors;
Mu_all(:,1:M_comp,t) = Mu;
Sigma_all(:,:,1:M_comp,t) = Sigma;

end


% % Fault_ID_mod = Fault_ID(1010:2500);
% Fault_ID_mod = Fault_ID;
% % Fault_ID_mod(find(Data_Test(11,:)==1))=[];
% % Fault_ID_mod(find(Data_Test(9,:)~=1))=[];
% 
% 
% Unknown_class = find(Fault_ID_mod == 11);
% Unknown_class_Pr = length(Unknown_class)/length(Fault_ID_mod)*100
% Unknown_class_Pr = round(Unknown_class_Pr,2)
% 
% True_class = find(Fault_ID_mod == R_class);
% % True_class_Pr = length(True_class)/(length(Fault_ID_mod)-length(Unknown_class))*100
% True_class_Pr = length(True_class)/(length(Fault_ID_mod))*100
% True_class_Pr = round(True_class_Pr,2)
% 
% Undetected = find(Fault_ID_mod == 0);
% Undetected_Pr = length(Undetected)/(length(Fault_ID_mod))*100
% Undetected_Pr = round(Undetected_Pr,2)
% 
% 
% MisDiag = 100-(Unknown_class_Pr+True_class_Pr+Undetected_Pr)
% 
% [True_class_Pr  Undetected_Pr MisDiag Unknown_class_Pr]






