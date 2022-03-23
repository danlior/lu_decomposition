#include <iostream>
#include <Eigen/Dense>
#include "lu.h"


namespace
{    
    using Eigen::MatrixXd;

    // casts matrices to type MatrixXd so that D can be inverted
    template<typename T>
    bool basic_test(const Matrix<T>& A)
    {
        auto ans = PLDUQ(A);      

        MatrixXd P = (ans[0]).template cast<double>();
        MatrixXd L = (ans[1]).template cast<double>();
        MatrixXd D = (ans[2]).template cast<double>();
        MatrixXd U = (ans[3]).template cast<double>();
        MatrixXd Q = (ans[4]).template cast<double>();

        MatrixXd A_decomposed = P*L*D.inverse()*U*Q;
        MatrixXd AA = A.template cast<double>();

        bool result = AA.isApprox(A_decomposed);
        
        if (!result)
        {
            std::cerr << "unexpected result: " << std::endl;
            std::cerr << A << std::endl << std::endl;
            std::cerr << A_decomposed << std::endl;
            std::cerr << "________________" << std::endl;
            for (auto a : ans)
                std::cerr << a << std::endl << std::endl;
        }
 
        return result;
    }

    template<typename T>
    bool test_identities()
    {
        if (!basic_test<T>(Matrix<T>::Identity(1,1))) return false;
        if (!basic_test<T>(Matrix<T>::Identity(3,1))) return false;
        if (!basic_test<T>(Matrix<T>::Identity(1,3))) return false;

        if (!basic_test<T>(Matrix<T>::Identity(2,2))) return false;
        if (!basic_test<T>(Matrix<T>::Identity(3,3))) return false;
        if (!basic_test<T>(Matrix<T>::Identity(4,4))) return false;

        if (!basic_test<T>(Matrix<T>::Identity(2,3))) return false;
        if (!basic_test<T>(Matrix<T>::Identity(3,2))) return false;

        return true;
    }

    template<typename T>
    bool test_assorted_stuff()
    {
        {
            Matrix<T> A(1,1);
            A(0,0) = -1;
            if (!basic_test<T>(A)) return false;
        }

        {
            Matrix<T> A(3,1);
            A(0,0) = 3;
            A(1,0) = 2;
            A(2,0) = -2;
            if (!basic_test<T>(A)) return false;
        }
        {
            Matrix<T> A(1,3);
            A(0,0) = 3;
            A(0,1) = 2;
            A(0,2) = -2;
            if (!basic_test<T>(A)) return false;
        }    
        {
            Matrix<T> A(2,2);
            A(0,0) = 1;
            A(0,1) = 1;
            A(1,0) = 1;
            A(1,1) = 1;
            if (!basic_test<T>(A)) return false;
        }
        {
            Matrix<T> A(2,2);
            A(0,0) = 1;
            A(0,1) = 1;
            A(1,0) = 1;
            A(1,1) = 0;
            if (!basic_test<T>(A)) return false;
        }
        {
            Matrix<T> A(2,2);
            A(0,0) = 1;
            A(0,1) = -1;
            A(1,0) = 1;
            A(1,1) = 0;
            if (!basic_test<T>(A)) return false;
        }
        {
            Matrix<T> A(2,2);
            A(0,0) = 1;
            A(0,1) = 2;
            A(1,0) = 5;
            A(1,1) = 7;
            if (!basic_test<T>(A)) return false;
        }
        {
            Matrix<T> A(2,2);
            A(0,0) = 2;
            A(0,1) = 0;
            A(1,0) = 0;
            A(1,1) = 1;
            if (!basic_test<T>(A)) return 1;
        }
        {
            Matrix<T> A(2,3);
            A(0,0) = 2;
            A(0,1) = 2;
            A(0,2) = -3;
            A(1,0) = 4;
            A(1,1) = 0;
            A(1,2) = 6;
            if (!basic_test<T>(A)) return false;
        }

        {
            Matrix<T> A(5,4);
            A(0,0) = 5;
            A(0,1) = 10;
            A(0,2) = 15;
            A(0,3) = 20;
            A(1,0) = -1;
            A(1,1) = -6;
            A(1,2) = -19;
            A(1,3) = -16;
            A(2,0) = 1;
            A(2,1) = 5;
            A(2,2) = 15;
            A(2,3) = 19;
            A(3,0) = 5;
            A(3,1) = 6;
            A(3,2) = -1;
            A(3,3) = -12;
            A(4,0) = 4;
            A(4,1) = 9;
            A(4,2) = 16;
            A(4,3) = 29;
            if (!basic_test<T>(A)) return false;
        }
        return true;
    }
}

int main()
{
// example from "LU Factoring of non-invertible matrices", D.J. Jeffery

    Matrix<int> A(5,4);
    A(0,0) = 5;
    A(0,1) = 10;
    A(0,2) = 15;
    A(0,3) = 20;
    A(1,0) = -1;
    A(1,1) = -6;
    A(1,2) = -19;
    A(1,3) = -16;
    A(2,0) = 1;
    A(2,1) = 5;
    A(2,2) = 15;
    A(2,3) = 19;
    A(3,0) = 5;
    A(3,1) = 6;
    A(3,2) = -1;
    A(3,3) = -12;
    A(4,0) = 4;
    A(4,1) = 9;
    A(4,2) = 16;
    A(4,3) = 29;

    auto ans = PLDUQ(A);      

    MatrixXd P = (ans[0]).template cast<double>();
    MatrixXd L = (ans[1]).template cast<double>();
    MatrixXd D = (ans[2]).template cast<double>();
    MatrixXd U = (ans[3]).template cast<double>();
    MatrixXd Q = (ans[4]).template cast<double>();

    MatrixXd A_decomposed = P*L*D.inverse()*U*Q;
    MatrixXd AA = A.template cast<double>();

    bool result = AA.isApprox(A_decomposed);
    
    if (!result)
        std::cerr << "unexpected result: " << std::endl;

    std::cerr << A << std::endl << std::endl;
    std::cerr << A_decomposed << std::endl;
    std::cerr << "________________" << std::endl;
    for (auto a : ans)
        std::cerr << a << std::endl << std::endl;


    if (!test_identities<int>()) return 1;
    if (!test_identities<double>()) return 1;
    if (!test_assorted_stuff<int>()) return 1;
    if (!test_assorted_stuff<double>()) return 1;

	return 0;
}
