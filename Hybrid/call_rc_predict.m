% load abcd
fname = 'data/abcd.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
abcd = jsondecode(str);

% load u_RC_needed_measure


y_model = np.zeros((u_arr.shape[1],))
x_discrete = np.array([[0], [10], [22], [21], [23], [21]])
for i in range(u_arr.shape[1]):
    y_model[i] = (self.abcd['c'] @ x_discrete + self.abcd['d'] @ u_arr[:, i])[0, 0]
    x_discrete = self.abcd['a'] @ x_discrete + (self.abcd['b'] @ u_arr[:, i]).reshape((6, 1))
return y_model[-1]