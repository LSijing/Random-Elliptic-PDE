function res = Legendre1D(p,x)
% Compute Legendre polynomials 1D
% P_POLYNOMIAL evaluates the Legendre polynomials P(n,x).
%    P(n,1) = 1.
%    P(n,-1) = (-1)^N.
%    | P(n,x) | <= 1 in [-1,1].
%    The N zeroes of P(n,x) are the abscissas used for Gauss-Legendre
%    quadrature of the integral of a function F(X) with weight function 1
%    over the interval [-1,1].
%    The Legendre polynomials are orthogonal under the inner product defined
%    as integration from -1 to 1:
%      Integral ( -1 <= X <= 1 ) P(I,X) * P(J,X) dX 
%        = 0 if I =/= J
%        = 2 / ( 2*I+1 ) if I = J.
%    Except for P(0,X), the integral of P(I,X) from -1 to 1 is 0.
%    A function F(X) defined on [-1,1] may be approximated by the series
%      C0*P(0,x) + C1*P(1,x) + ... + CN*P(n,x)
%    where
%      C(I) = (2*I+1)/(2) * Integral ( -1 <= X <= 1 ) F(X) P(I,x) dx.
%    The formula is:
%      P(n,x) = (1/2^N) * sum ( 0 <= M <= N/2 ) C(N,M) C(2N-2M,N) X^(N-2*M)
%  Differential equation:
%    (1-X*X) * P(n,x)'' - 2 * X * P(n,x)' + N * (N+1) = 0
%  First terms:
%    P( 0,x) =      1
%    P( 1,x) =      1 X
%    P( 2,x) = (    3 X^2 -       1)/2
%    P( 3,x) = (    5 X^3 -     3 X)/2
%    P( 4,x) = (   35 X^4 -    30 X^2 +     3)/8
%    P( 5,x) = (   63 X^5 -    70 X^3 +    15 X)/8
%    P( 6,x) = (  231 X^6 -   315 X^4 +   105 X^2 -     5)/16
%    P( 7,x) = (  429 X^7 -   693 X^5 +   315 X^3 -    35 X)/16
%    P( 8,x) = ( 6435 X^8 - 12012 X^6 +  6930 X^4 -  1260 X^2 +   35)/128
%    P( 9,x) = (12155 X^9 - 25740 X^7 + 18018 X^5 -  4620 X^3 +  315 X)/128
%    P(10,x) = (46189 X^10-109395 X^8 + 90090 X^6 - 30030 X^4 + 3465 X^2-63)/256
%  Recursion:
%    P(0,x) = 1
%    P(1,x) = x
%    P(n,x) = ((2*n-1)*x*P(n-1,x)-(n-1)*P(n-2,x)) / n
%  Parameters:
%    Input, integer M, the number of evaluation points.
%    Input, integer N, the highest order polynomial to evaluate.
%    Note that polynomials 0 through N will be evaluated.
%    Input, real X(M,1), the evaluation points.
%    Output, real V(M,1:N+1), the values of the Legendre polynomials 
%    of order 0 through N at the points X.
%% compute
% % Compute  Legendre polynomials 1D
N        = length(x);  
pmax     = 20; 
% use the recursion formula to construct Legendre polynomials
L        = [ones(N,1)  x];
for n = 2:pmax 
    temp = (2*n-1)/n*x.*L(:,n) - (n-1)/n*L(:,n-1); 
    L    = [L temp];
end
% normalize the Hermite Polynomials
for n = 1:pmax 
    L(:,n) = L(:,n)/sqrt(1/(2*n-1));  
end
% return the result
res        = L(:,p+1);