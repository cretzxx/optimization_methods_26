#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

double f(double x) {
    return pow(x - 1.0, 2.0) * pow(x + 1.0, 4.0) * pow(x - 2.0, 3.0);
}

int main() {
    double a = -0.95;
    double b = 1.05;
    double sigma = 0.001;
    double epsilon = 0.0001;
    int k = 1;

    cout << "k\tx1\t\tx2\t\tf(x1)\t\tf(x2)\t\t[ak, bk]\t\tLk" << endl;
    cout << "--------------------------------------------------------------------------------------------------" << endl;
    cout << "0\t-\t\t-\t\t-\t\t-\t\t[" << a << "; " << b << "]\t" << (b - a) << endl;

    while (true) {
        double x1 = (a + b) / 2.0 - epsilon / 2.0;
        double x2 = x1 + epsilon;
        double f1 = f(x1);
        double f2 = f(x2);

        if (f1 < f2) {
            b = x2;
        } else if (f1 > f2) {
            a = x1;
        } else {
            a = x1;
            b = x2;
        }

        double L = b - a;

        cout << k << "\t" << fixed << setprecision(4) << x1 
        << "\t" << x2 << "\t" << f1 << "\t" << f2 << "\t[" << a << "; " << b << "]\t" << L << endl;

        if (L <= sigma) {
            double x_star = (a + b) / 2.0;
            cout << "--------------------------------------------------------------------------------------------------" << endl;
            cout << "x* = " << x_star << endl;
            break;
        }
        k++;
    }

    return 0;
}