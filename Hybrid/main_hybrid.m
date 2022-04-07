clear
potential_nbstates = [2,10,40];
input_case = 3;

nb_row = 1;
for nbStates = potential_nbstates
    [cvrmse_gmr, cvrmse_ggmr] = gmr_ggmr_rs(nbStates,input_case);
    all_cvrmse(nb_row, :) = [cvrmse_gmr, cvrmse_ggmr];
    nb_row = nb_row + 1;
end
subplot(1,2,1);
plot(potential_nbstates,all_cvrmse(:,1))
xlabel('Number of Gaussian Components') 
ylabel('CVRMSE(%)') 
title('GMR')

subplot(1,2,2);
plot(potential_nbstates,all_cvrmse(:,2))
xlabel('Number of Gaussian Components') 
ylabel('CVRMSE(%)') 
title('GGMR')

sgtitle("case "+input_case); 
savefig("data/"+input_case +"input_case.fig")