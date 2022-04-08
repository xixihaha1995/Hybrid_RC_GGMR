clear
potential_nbstates = [2,5,10,15];
input_case = 2;

nb_row = 1;
for nbStates = potential_nbstates
    [nrmse_gmr, cvrmse_gmr, mae_gmr, mape_gmr, ...
    nrmse_ggmr, cvrmse_ggmr, mae_ggmr, mape_ggmr] = gmr_ggmr_rs(nbStates,input_case);
    all_performance(nb_row, :) = [nrmse_gmr, cvrmse_gmr, mae_gmr, mape_gmr, ...
            nrmse_ggmr, cvrmse_ggmr, mae_ggmr, mape_ggmr];
    nb_row = nb_row + 1;
end

subplot(1,2,1);
yyaxis left
hold on
p1 = plot(potential_nbstates,all_performance(:,1));
p2 = plot(potential_nbstates,all_performance(:,2));
p4 = plot(potential_nbstates,all_performance(:,4));
ylabel("Error (%)")
yyaxis right
p3 = plot(potential_nbstates,all_performance(:,3));
xlabel('Number of Gaussian Components') 
ylabel("Watts")
title('GMR')

subplot(1,2,2);
yyaxis left
hold on
p5 = plot(potential_nbstates,all_performance(:,5));
p6 = plot(potential_nbstates,all_performance(:,6));
p8 = plot(potential_nbstates,all_performance(:,8));
ylabel("Error (%)")
yyaxis right
p7 = plot(potential_nbstates,all_performance(:,7));
xlabel('Number of Gaussian Components') 
ylabel("Watts")
title('GGMR')

sgtitle("case "+input_case); 
legend([p1,p2,p3,p4,p5,p6,p7,p8],{'NRMSE-GMR','CVRMSE-GMR','MAE-GMR', ...
    'MAPE-GMR','NRMSE-GGMR','CVRMSE-GGMR','MAE-GGMR','MAPE-GGMR'},'Orientation','horizontal');
savefig("data/"+input_case +"input_case.fig")