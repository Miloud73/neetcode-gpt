class Solution:
    def format_number(self,num):
        if num == 0:
            return round(num,1)
        elif num % 1 == 0:
            return int(num)
        else:
            return round(num,5)

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = float(init)
        for _ in range(iterations):
            gradient = 2 * x
            if abs(gradient) <= 0.001:
                break
            x -= learning_rate * gradient
        return self.format_number(x) 