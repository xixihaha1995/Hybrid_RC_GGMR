clear
potential_nbstates = [2,5,10,15];

input_case = 4;
potential_L_rate = [5e-3, 8e-3];


nb_row = 1;
for L_rate = potential_L_rate
    rc_ggmr_hybrid_rs(15,input_case, L_rate);
    nb_row = nb_row + 1;
end

% yyaxis left
% hold on
% p5 = plot(potential_L_rate,all_performance(:,5));
% p6 = plot(potential_L_rate,all_performance(:,6));
% p8 = plot(potential_L_rate,all_performance(:,8));
% ylabel("Error (%)")
% yyaxis right
% p7 = plot(potential_L_rate,all_performance(:,7));
% xlabel('Learning Rates') 
% ylabel("Watts")
% title("case "+input_case)
% 
% legend([p5,p6,p7,p8],{'NRMSE-GGMR','CVRMSE-GGMR','MAE-GGMR','MAPE-GGMR'},'Orientation','horizontal');
% savefig("data/"+input_case +"input_case.fig")