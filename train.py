import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, data, learning_rate=0.01):
        """
            Constructor
        """
        self.learning_rate = learning_rate
        self.theta0: float = 0.0
        self.theta1: float = 0.0
        self.denormalizedTheta0: float = 0.0
        self.denormalizedTheta1: float = 0.0
        self.normalizedKm = []
        self.normalizedPrice = []
        self.km = data['km']
        self.price = data['price']
        self.length: int = len(self.km)
        self.normalizeData()
        self.theta0tmp: float = 0.0
        self.theta1tmp: float = 0.0
        self.historyTheta0 = []
        self.historyTheta1 = []
        self.historyPrecision = []
        self.train()
        self.plot()

    def normalizeData(self):
        """
            Normalize the data
                Get the values of km and price and truncate them to the range [0, 1]
        """

        for i in range(self.length):
            self.normalizedKm.append((float(self.km[i]) - min(self.km)) / (max(self.km) - min(self.km)))
            self.normalizedPrice.append((float(self.price[i]) - min(self.price)) / (max(self.price) - min(self.price)))

    def denormalize(self):
        """
            Denormalize the data
        """
        self.denormalizedTheta1 = (max(self.price) - min(self.price)) * self.theta1 / (max(self.km) - min(self.km))
        self.denormalizedTheta0 = min(self.price) + ((max(self.price) - min(self.price)) * self.theta0) + self.denormalizedTheta1 * (-min(self.km))

    def estimatePrice(self, mileage):
        """
            Estimate the price of the car
            y = θ0 + θ1 * x
            y = b + (w*x)
            ^price = b + (w * km)
        """
        return (self.theta0 + (self.theta1 * float(mileage)))

    def updateTheta0(self):
        """
            Update the value of theta0
                This is actually the partial derivative of the cost function with respect to theta0
                theta0 = theta0 - learning_rate * (1/2n) * Σ( -2 * price[i] - estimatePrice(km[i])
                theta0 = theta0 - learning_rate * (-1/n) * Σ((price[i] - (theta0 + (theta1 * float(km[i])))
                theta0 = theta0 - learning_rate * (1/n) * Σ(price[i] - (theta0 + (theta1 * float(km[i])))
        """
        sum = 0.0

        for i in range(self.length):
            sum += float(self.normalizedPrice[i]) - self.estimatePrice(self.normalizedKm[i])
        self.theta0tmp = self.learning_rate * (-sum / self.length)

    def updateTheta1(self):
        """
            Update the value of theta1
                theta1 = theta1 - learning_rate * (-1/2n) * Σ( -2 * (km[i] * (price[i] - (theta0 + (theta1 * float(km[i])))))
                theta1 = theta1 - learning_rate * (-1/n) * Σ( km[i] * (price[i] - (theta0 + (theta1 * float(km[i]))))
        """
        sum = 0.0

        for i in range(self.length):
            sum += float(self.normalizedKm[i]) * (float(self.normalizedPrice[i]) - self.estimatePrice(self.normalizedKm[i]))
        self.theta1tmp = self.learning_rate * (-sum / self.length)

    def updateTheta(self):
        """
            Update the values of theta0 and theta1
        """
        self.updateTheta0()
        self.updateTheta1()
        self.theta0 -= self.theta0tmp
        self.theta1 -= self.theta1tmp

    def meanSquareError(self):
        """
            Calculate the mean square error
            1/2n * Σ(yᵢ - ȳ)²
        """
        sum = 0.0

        for i in range(self.length):
            sum += (self.estimatePrice(self.normalizedKm[i]) - float(self.normalizedPrice[i])) ** 2
        return (sum / (self.length*2))

    def getPrecision(self):
        """
            Get the precision of the model using the R² method
            SSres = Σ(yᵢ - ŷ)² = Σ(yᵢ - (θ0 + θ1 * xᵢ))²
            SStot = Σ(yᵢ - ȳ)²
        """
        totalPredict = sum([abs((self.denormalizedTheta0 + self.denormalizedTheta1 * self.km[i]) - self.price[i]) for i in range(self.length)])
        return 1 - totalPredict / sum(self.price)

    def saveWeights(self):
        """
            Save the weights of the model
        """
        self.denormalize()
        self.historyTheta0.append(self.denormalizedTheta0)
        self.historyTheta1.append(self.denormalizedTheta1)
        self.historyPrecision.append(self.getPrecision())

    def saveModel(self):
        save = open('model.csv', 'w')
        save.write(str(self.denormalizedTheta0) + "," + str(self.denormalizedTheta1))

    def train(self):
        """
            Train the model
            Add the weights to the history
        """
        mse = self.meanSquareError()
        sharpness = mse
        while abs(sharpness) > mse * 0.00001:
            self.saveWeights()
            self.updateTheta()

            tmp = mse
            mse = self.meanSquareError()
            sharpness = mse - tmp

        self.denormalize()
        self.saveModel()

    def plot(self):
        """
            Plot the data
        """
        for i in range(len(self.historyTheta1)):
            plt.cla()

            plt.title('ft_linear_regression')
            plt.title('Ayalman', loc="right")
            plt.xlabel('Km')
            plt.ylabel('Price')

            plt.plot(self.km, [(self.historyTheta0[i] + n * self.historyTheta1[i]) for n in self.km], color='red')
            plt.scatter(self.km, self.price)

            plt.axis([0, max(self.km) * 1.1, min(self.price) / 2, max(self.price) * 1.25])

            plt.legend(['\nWeights\nt0: '
                        + str(self.historyTheta0[i])[:6]
                        + ' | t1: ' + str(self.historyTheta1[i])[:6]
                        + '\nPres: %'
                        + str(self.historyPrecision[i])[2:4] + '.'
                        + str(self.historyPrecision[i])[4:6]])
            plt.pause(0.001)
        plt.show()


def main():

    try:
        data_file = input('\033[92m'+'Data file: '+'\033[0m')
        data = pd.read_csv(data_file)
        LinearRegression(data)
    except Exception as e:
        print('Exception:', '\033[91m', e, '\033[0m')


if __name__ == '__main__':
    main()
