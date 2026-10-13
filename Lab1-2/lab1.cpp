#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

double f(double x) {
    return pow(x - 1.0, 2.0) * pow(x + 1.0, 4.0) * pow(x - 2.0, 3.0);
}

int main() {
    double x0 = 0.05;
    double delta = 1.0;
    
    double f0 = f(x0);
    
    cout << "k\tDelta\tx_k\t\tf(x_k)\t\tInterval [a0, b0]" << endl;
    cout << "-------------------------------------------------------------" << endl;
    cout << 0 << "\t" << delta << "\t" << fixed << setprecision(4) << x0 << "\t\t" << f0 << "\t\t-" << endl;
    
    double x_right = x0 + delta;
    double f_right = f(x_right);
    cout << 1 << "\t" << delta << "\t" << x_right << "\t\t" << f_right << "\t\t-" << endl;
    
    if (f_right >= f0) {
        double x_left = x0 - delta;
        double f_left = f(x_left);
        cout << 2 << "\t" << -delta << "\t" << x_left << "\t\t" << f_left << "\t\t-" << endl;
        
        if (f_left >= f0) {
            double a0 = x0 - delta;
            double b0 = x0 + delta;
            cout << "-------------------------------------------------------------" << endl;
            cout << "Initial interval [a0, b0] = [" << a0 << "; " << b0 << "]" << endl;
        }
    }

    return 0;
}