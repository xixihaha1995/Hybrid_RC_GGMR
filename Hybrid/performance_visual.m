% rc_t=readtable('data/_rc_saved_6_measured_modeled.csv');
% rc_pre = rc_t{4033:end,2};
% rc_pre = rc_pre.';
% 
% act = rc_t{4033:end,1};
% act = act.';
load('data/act_rc_ggmr_hybrid.mat');
hold on
plot(act_rc_ggmr_hybrid(1,:))
plot(act_rc_ggmr_hybrid(2,:))
plot(act_rc_ggmr_hybrid(3,:))
plot(act_rc_ggmr_hybrid(4,:))
legend({'Measurement','RC','GGMR','Hybrid'},'Location','southwest')
xlabel('Time Steps, 5 mins interval')
ylabel('Radiant Slab Load (W)')