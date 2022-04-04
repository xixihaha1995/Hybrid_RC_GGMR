clear
potential_nbstates = [8, 16, 32];
nb_row = 1;
for nbStates = potential_nbstates
    [cvrmse_gmr, cvrmse_ggmr] = ggmr_rs(nbStates);
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