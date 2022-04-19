function [res] = matlab_client(client_said)
% 'Type the following in Matlab Command Window' 
%     pyenv('Version', 'C:\Users\[Your User Name]\AppData\Local\Programs\Python\Python38\python.exe')
%     Example: pyenv('Version', 'C:\Users\wulic\AppData\Local\Programs\Python\Python38\python.exe')
%     res = pyrunfile("python_server.py","res",client_msg="Hello Python server")
res = pyrunfile("python_server.py","res",client_msg=client_said);
end