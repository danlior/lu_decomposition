#include <Eigen/Dense>
#include <cassert>

// returns matrices P,U,D,L,Q, of the same scalar type as A, such that
// A = P*L*D^{-1}*U*Q, where
// P, Q are permutation matrices
// D is an invertible diagonal matrix with entries nondecreasing in absolute value
// L is lower triangular
// U is upper triangular


// todo: detect overflow (especially in D with integral scalars)

template<typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
std::array<Matrix<T>, 5> PLDUQ(const Matrix<T>& A)
{
    assert(!A.isZero());

    const uint64_t m = A.rows(); 
    const uint64_t n = A.cols(); 

    assert(m>0);
    assert(n>0);

    // "complete" pivoting strategy: 
    // pivot on the entry with the largest absolute value
    uint64_t i=-1;
    uint64_t j=-1;
    double maxa = 0;
    for (auto i_ = 0; i_<m; ++i_)
        for (auto j_=0; j_<n; ++j_)
            if (abs(A(i_,j_)) > maxa)
            {
                i = i_;
                j = j_;
                maxa = abs(A(i,j));
            }


    assert(A(i,j) != 0);

    Matrix<T> tau_P = Matrix<T>::Identity(m,m); // the prefix tau indicates transposition
    Matrix<T> tau_Q = Matrix<T>::Identity(n,n); // the prefix tau indicates transposition

    if (i != 0)
    {
        // transpose rows 0 and i of P
        tau_P(0,0) = 0;
        tau_P(i,i) = 0;
        tau_P(0,i) = 1;
        tau_P(i,0) = 1;
    }
    if (j != 0)
    {
        // transpose cols 0 and j of Q
        tau_Q(0,0) = 0;
        tau_Q(j,j) = 0;
        tau_Q(0,j) = 1;
        tau_Q(j,0) = 1;
    }

    auto A0 = tau_P * A * tau_Q;
    assert(A0(0,0) == A(i,j)); // pivot
    assert(A0(0,0) != 0);

    // Caution: if m or n are 1, v or w may be "degenerate"
    Matrix<T> d = A0.topLeftCorner(1,1);
    Matrix<T> a = A0.bottomRightCorner(m-1,n-1);
    Matrix<T> v = A0.bottomLeftCorner(m-1,1);
    Matrix<T> w = A0.topRightCorner(1, n-1);
    assert(!d.isZero());

    const Matrix<T> my_det = d(0,0)*a-v*w; // degenerate if m==1 or n==1
    const bool is_rank_one = (m==1) || (n==1) || my_det.isZero();

    std::array<Matrix<T>, 5> rtn;

    if (is_rank_one) // base case
    {
        Matrix<T> L = Matrix<T>::Identity(m,1);
        L(0,0) = d(0,0);

        Matrix<T> U = Matrix<T>::Identity(1,n);
        U(0,0) = d(0,0);

        // make sure to avoid "degenerate" cases
        if (m>1)
            L.block(1,0,m-1,1) = v;

        if (n>1)
            U.block(0,1,1,n-1) = w;

        rtn = std::array<Matrix<T>, 5>({tau_P,L,d,U,tau_Q});
    }
    else  // rank A > 1, recursive case
    {
        auto recursive_decomposition = PLDUQ(my_det);

        auto P1 = recursive_decomposition[0];
        auto L1 = recursive_decomposition[1];
        auto D1 = recursive_decomposition[2];
        auto U1 = recursive_decomposition[3];
        auto Q1 = recursive_decomposition[4];

        assert(P1.rows() == m-1);
        assert(P1.cols() == m-1);
        Matrix<T> P = Matrix<T>::Identity(m,m);
        P.bottomRightCorner(m-1, m-1) = P1;
        P = tau_P * P;

        assert(Q1.rows() == n-1);
        assert(Q1.cols() == n-1);
        Matrix<T> Q = Matrix<T>::Identity(n,n);
        Q.bottomRightCorner(n-1, n-1) = Q1;
        Q = Q * tau_Q;

        assert(D1.isDiagonal());
        uint64_t rank = 1 + D1.rows();
        Matrix<T> D = Matrix<T>::Zero(rank,rank);
        D(0,0) = d(0,0);
        D.bottomRightCorner(rank-1, rank-1) = D1*d(0,0);

        assert(L1.isLowerTriangular());
        assert(L1.rows() == m-1);
        assert(L1.cols() == rank-1);
        Matrix<T> L = Matrix<T>::Zero(m,rank);
        L(0,0) = d(0,0);
        L.bottomLeftCorner(m-1,1) = P1.transpose()*v;
        L.bottomRightCorner(m-1, rank-1) = L1;

        assert(U1.isUpperTriangular());
        assert(U1.rows() == rank-1);
        assert(U1.cols() == n-1);
        Matrix<T> U = Matrix<T>::Zero(rank,n);
        U(0,0) = d(0,0);
        U.topRightCorner(1, n-1) = w*(Q1.transpose());
        U.bottomRightCorner(rank-1,n-1) = U1;

        rtn = std::array<Matrix<T>, 5>({P,L,D,U,Q});
    }

    return rtn;
}

