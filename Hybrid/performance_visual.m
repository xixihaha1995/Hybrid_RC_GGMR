% % rc_t=readtable('data/_rc_saved_6_measured_modeled.csv');
% % rc_pre = rc_t{4033:end,2};
% % rc_pre = rc_pre.';
% % 
% % 
% % test_hours = fix(size(rc_pre,2) / 12);
% % rc_pre_t = reshape(rc_pre, 12, test_hours );
% % rc_pre_t = sum(rc_pre_t);
% % 
% % hourly_rc = rc_pre_t;
% % save('data/hourly_rc','hourly_rc');
% 
% load('data/hourly_measure.mat');
% load('data/hourly_rc.mat');
% load('data/hourly_ggmr_flow_no_rc.mat');
% load('data/hourly_ggmr_flow_with_rc.mat');
% load('data/hourly_hybrid_flow_with_rc.mat');
% 
% hourly_act_ggmr2_hybrid = [hourly_measure;hourly_rc;hourly_ggmr_flow_no_rc;hourly_ggmr_flow_with_rc;hourly_hybrid_flow_with_rc]./1000;
% 
% hold on
% plot(hourly_act_ggmr2_hybrid(1,:),'-o')
% plot(hourly_act_ggmr2_hybrid(2,:),'--s')
% plot(hourly_act_ggmr2_hybrid(4,:),':x','Color',[0 0 0])
% plot(hourly_act_ggmr2_hybrid(5,:),'-.^')
% legend({'Measurement','RC','GGMR','Hybrid'},'Location','southwest')
% xlabel('Time Steps, one hour interval')
% ylabel('Radiant Slab Heating Load (kW)')
% savefig("data/hourly_comparison.fig")
% 
% save("outputs/_saved"+outputs,"comments","x_axis","y_label","title_str", ...
%     "p","legend_str");

hold on;
ylabel(y_label)

p1 = plot(x_axis, p(1), '-o', LineWidth=2);
p2 = plot(x_axis, p(2),':x',LineWidth=2);
p3 = plot(x_axis, p(3),'--s',LineWidth=2);
p4 = plot(x_axis, p(4),'-.^',LineWidth=2);
p1.Color = '#614124';
p2.Color = '#CC704B';
p3.Color = '#E8C07D';
p4.Color = '#9FC088';

set(gca,'FontSize',20)

title(title_str,fontsize=15)
legend(legend_str,FontSize=15)

hold off;
pause;