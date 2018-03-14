class IncorrectArgumentException(Exception):
    def __init__(self, error):
        self.error = error
    def __str__(self):
        return repr(self.error)

empty_characters = [" ", "", "\n", "\t"]

class NewtonRaphson():
    def get_inputs(self):
        self.degree = int(input("Enter the degree of the polynomial\n"))
        self.coef = input("Enter the coefficients separated by space\n")
        self.coef = [int(x) for x in self.coef.split(" ") if x not in empty_characters]
        if(len(self.coef) != (self.degree+1)):
            raise IncorrectArgumentException("Incorrect number of coefficients passed!")
        return self.degree, self.coef

    def f(self, x):
        function = 0
        for i in range(self.degree+1):
            function = function + self.coef[self.degree-i]*(x**i)
        return function

    def df(self,x):
        derivative_function = 0
        for i in range(self.degree + 1 ):
            derivative_function = derivative_function + i*self.coef[self.degree-i]*(x**(i-1))
        return derivative_function

    def errorfunction(self, x):
        return abs(0 - self.f(x))

    def find_roots(self, x0, error = 0):
        while(self.errorfunction(x0) > error):
            x0 = x0 - self.f(x0)/self.df(x0)
        return x0

n = NewtonRaphson()
n.get_inputs()
print(n.find_roots(x0=100, error=0))