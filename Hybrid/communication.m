function [res] = communication(target_time)
% 'Type the following in Matlab Command Window' 
%     pyenv('Version', 'C:\Users\[Your User Name]\AppData\Local\Programs\Python\Python38\python.exe')
%     res = pyrunfile("GGMR_Call_RC.py","res",target_time_idx=782)
res = pyrunfile("GGMR_Call_RC.py","res",target_time_idx=target_time);
end