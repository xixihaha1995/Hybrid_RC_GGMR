function [res] = RC_PredictedRealTime(u_arr, abcd)
    x_discrete = [0;10;22;21;23;21];
    for idx = 1 : size(u_arr,2)
      y_all(idx) = abcd.c * x_discrete + abcd.d * u_arr(:, idx);
      x_discrete = abcd.a * x_discrete + abcd.b * u_arr(:, idx);
    end
    res =y_all(end);
end