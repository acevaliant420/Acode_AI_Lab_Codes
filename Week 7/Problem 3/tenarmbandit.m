function [value, m] = bandit_nonstat(action, m)
v = normrnd(0,0.01, [1, 10]);
m = m + v;
value = m(action);
end

Q = zeros(1, 10);
N = zeros(1, 10);
R = zeros(1, 10000);
epsilon = 0.1;
m = ones(1, 10);
RR = 0;
for i = 1:10000
if rand > epsilon
 [a, id] = max(Q);
 A = id;
else
 temp = randperm(10);
 A = temp(1);
end
 [RR, m] = bandit_nonstat(A, m);
 N(A) = N(A) + 1;
 Q(A) = Q(A) + (RR - Q(A)) / N(A);
if i == 1
 R(i) = RR;
else
 R(i) = ((i - 1) * R(i - 1) + RR) / i;
end
end
i = 1:10000;
plot(i, R, 'r');
title('Average Reward Over Time');
xlabel('Steps');
ylabel('Average Reward');