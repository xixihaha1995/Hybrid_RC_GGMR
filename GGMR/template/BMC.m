
function [m_best,Post_pr] = BMC(Data_in,Priors_in,Mu_in,Sigma_in)
% Find the best match component
    for m=1:size(Priors_in,2)
        Post_pr(m,1) = Priors_in(m)*gaussPDF(Data_in,Mu_in(:,m),Sigma_in(:,:,m));
    end
    m_best = find(Post_pr == max(Post_pr)); % index of best match component
    %Data_in
    %Sigma_in
    %Mu_in
    %Priors_in
    %Post_pr
    %max(Post_pr)
    %m_best
    m_best = m_best(1);
end 