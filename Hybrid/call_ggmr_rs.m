clear
potential_nbstates = [15, 20];
input_case = 4;

nb_row = 1;
for nbStates = potential_nbstates
    [cvrmse_gmr, cvrmse_ggmr] = ggmr_rs(nbStates,input_case);
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


switch (input_case)
    case 1
        sgtitle_name = "case 1";
    case 2
        sgtitle_name = "case 2";
    case 3
        sgtitle_name = "case 3";
    case 4
        sgtitle_name = "case 4";
end
sgtitle(sgtitle_name); 
savefig("data/"+input_case +"input_case.fig")